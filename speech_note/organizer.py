"""Transcript cleanup: build the prompt from the ASR sources, call the model, sanity-check.

Only cleanup lives here. The pieces it used to carry moved out, because each is a
separate concern with separate reasons to change:

  chat.py         — the HTTP transport (auth, model fallback, response_format).
  llama_server.py — the optional local llama-server child process.
  annotate.py     — the uncertainty pass that runs *after* cleanup.

Cleanup and annotation are deliberately two calls: one model asked to preserve content,
strip disfluencies and judge its own confidence has the leeway to start summarizing, and
summarizing is the failure this pipeline exists to avoid. So this prompt does one job,
and annotate.py audits the result without being able to rewrite it.
"""

from __future__ import annotations

import dataclasses
import math
import shutil
import sys
import time
from typing import TYPE_CHECKING, Callable

from . import annotate as annotation
from .chat import ChatClient, request_timeout_seconds
from .config import (
    CLEANUP_MIN_LENGTH_RATIO,
    CLEANUP_TARGET_LENGTH_RATIO,
    DEFAULT_GGUF_MODEL,
    ORGANIZER_CONTEXT_SAFETY,
    ORGANIZER_MIN_OUTPUT_TOKENS,
)
from .llama_server import (
    LocalServerSupervisor,
    default_server_command,
    ensure_default_cleanup_model,
)
from .model import CleanupOutcome, Transcript
from .textproc import count_words, estimate_text_tokens, heuristic_cleanup

if TYPE_CHECKING:
    from .cli import Config


class PromptTooLargeError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class CleanupRequestPlan:
    user_prompt: str
    estimated_prompt_tokens: int
    largest_source_tokens: int
    requested_output_tokens: int
    token_budget: int
    output_cap_limited: bool
    # Length expectation, shared by the prompt and the post-hoc shortness check so
    # the two can never drift apart.
    reference_words: int
    minimum_words: int


SYSTEM_PROMPT = (
    "You are a transcript reconstruction engine. You are given several machine "
    "transcripts of the SAME recording, each from a different speech recognizer. The "
    "recognizers differ mainly in SENSITIVITY: some hear faint or fast speech that "
    "others miss entirely. A passage appearing in only one transcript almost always "
    "means the other recognizers failed to hear it — NOT that it is fake. Your job is "
    "to reconstruct the union of what was actually said: keep everything any source "
    "heard, use the other sources to fix wording and spelling, and drop only true "
    "non-speech (filler, recognition garbage, degenerate loops). The person "
    "who made the recording will read your output and can spot a wrong line at a glance, "
    "so losing real content is far worse than keeping a slightly uncertain line. "
    "You have two SEPARATE jobs that must not be confused: (1) KEEP all real content, "
    "including anything only one source heard — never drop a substantive word, phrase, or "
    "clause; (2) AGGRESSIVELY strip disfluencies — um, uh, er, filler 'like', 'you know', "
    "'I mean', false starts, and stutters — so the transcript reads cleanly. Removing a "
    "disfluency is good; removing real content is bad. Do both at once. Output only the "
    "final transcript text — no notes, no summary."
)


def _reference_words(sources: list[Transcript]) -> int:
    """Mean word count across non-empty sources, used to anchor the length expectation.

    The mean, not the max: a single filler-heavy source (a verbatim ASR that kept every
    um/uh and false start) inflates the max, which would demand a bloated output and
    false-flag a correctly-cleaned transcript as too short. The mean is robust to one
    long, noisy source while still rising when several sources heard a lot.
    """
    counts = [s.words for s in sources if s.words > 0]
    return round(sum(counts) / len(counts)) if counts else 0


def build_user_prompt(sources: list[Transcript]) -> str:
    """Build the cleanup prompt with named sources.

    Sources are identified by what produced them, with quality hints when we have them.
    The prompt reconstructs the UNION of what every source heard: source disagreement is
    treated as a difference in sensitivity (one recognizer missed something), not as a
    vote on whether a passage is real, so content carried by a single source is kept.
    """
    reference_length = _reference_words(sources)
    minimum_words = max(1, math.ceil(reference_length * CLEANUP_MIN_LENGTH_RATIO))
    target_words = max(minimum_words, math.ceil(reference_length * CLEANUP_TARGET_LENGTH_RATIO))
    source_blocks = []
    for index, source in enumerate(sources, start=1):
        header = f"Source {index} — {source.label} ({source.model})"
        if source.quality_hint:
            header += f" — {source.quality_hint}"
        header += f" — {len(source.text)} chars / {source.words} words"
        source_blocks.append(f"{header}:\n{source.text}")
    joined_sources = "\n\n".join(source_blocks)
    return (
        "Reconstruct one readable near-verbatim transcript that is the UNION of what the "
        "sources heard, with disfluencies cleaned out.\n\n"
        "Source handling:\n"
        "- Each source below is a fallible transcription of the same recording; its header "
        "says what produced it.\n"
        "- The sources DISAGREE mostly by sensitivity, not reliability. Where one source has "
        "content the others lack, assume the others simply missed it and KEEP that content.\n"
        "- Use agreement between sources to decide HOW a word was said (spelling, which "
        "homophone, a garbled name), never WHETHER a passage exists.\n"
        "- A passage carried by a single source SHOULD be kept unless it is clearly recognition "
        "garbage (random unconnected words, a degenerate repeated loop, an obvious mis-decode).\n"
        "- Align the sources chronologically and reconstruct the full spoken content that best "
        "explains all of them together.\n\n"
        "Length expectation:\n"
        f"On average the sources are {reference_length} words. Because you are keeping the "
        f"UNION of what every source heard, the result should land near {target_words} words "
        f"or more; well below {minimum_words} words means real content was dropped. MUST NOT "
        "pad with junk or repeated words to hit a length; coverage of real speech is the "
        "goal, not raw count.\n\n"
        "Editing contract:\n"
        "- MUST preserve chronological transcript form.\n"
        "- MUST keep all meaningful spoken content, INCLUDING content that only one source "
        "heard. Never drop a substantive word, phrase, clause, name, number, or example.\n"
        "- SHOULD aggressively remove disfluencies and verbal tics — um, uh, er, filler 'like', "
        "'you know', 'I mean', false starts, restarts, and repeated stutters — they are not "
        "content and they hurt readability. This is about HOW things are said, not WHAT was "
        "said.\n"
        "- MUST preserve concrete details, examples, caveats, corrections, asides, "
        "transitions, questions, answers, instructions, names, and numbers.\n"
        "- MUST NOT output a summary or outline, paraphrase away spoken content, or replace "
        "a stretch of speech with a shorter description.\n"
        "- MUST NOT invent content that no source supports.\n"
        "- MUST NOT write notes about the transcript or mention the editing process.\n"
        "- MUST remove long runs of repeated short acknowledgments (okay/yeah/yes) "
        "and degenerate repeated loops.\n"
        "- MAY fix punctuation, casing, repeated fragments, and clear mishearings.\n"
        "- If a phrase is repeated, keep the first clear instance unless later repetitions "
        "change the meaning or emphasis.\n"
        "- MAY merge adjacent fragments only when they are clearly part of the same spoken "
        "thought.\n"
        "- SHOULD put a blank line after every 3 to 6 sentences at natural pauses.\n"
        "- MAY add light structure (speaker labels, headings) only if it clarifies the "
        "transcript, and MUST NOT let formatting replace, shorten, reorder, or summarize "
        "spoken content.\n"
        "- MUST output only the final transcript.\n\n"
        f"{joined_sources}\n\n"
        "Output the reconstructed transcript now: keep all real content, strip the filler. "
        "Start at the beginning of the recording and continue linearly until the end."
    )


def cleanup_request_plan(
    sources: list[Transcript],
    *,
    context_tokens: int,
    max_output_tokens: int,
) -> CleanupRequestPlan:
    """Estimate the cleanup request exactly once, including output cap pressure."""
    sources = _dedupe_sources([source for source in sources if source.text.strip()])
    user_prompt = build_user_prompt(sources)
    estimated_prompt_tokens = (
        estimate_text_tokens(SYSTEM_PROMPT) + estimate_text_tokens(user_prompt) + 32
    )
    largest_source_tokens = max((estimate_text_tokens(s.text) for s in sources), default=0)
    uncapped_output_tokens = max(ORGANIZER_MIN_OUTPUT_TOKENS, largest_source_tokens * 2)
    requested_output_tokens = min(max_output_tokens, uncapped_output_tokens)
    token_budget = max(256, int(context_tokens * ORGANIZER_CONTEXT_SAFETY))
    reference_words = _reference_words(sources)
    return CleanupRequestPlan(
        user_prompt=user_prompt,
        estimated_prompt_tokens=estimated_prompt_tokens,
        largest_source_tokens=largest_source_tokens,
        requested_output_tokens=requested_output_tokens,
        token_budget=token_budget,
        output_cap_limited=uncapped_output_tokens >= max_output_tokens,
        reference_words=reference_words,
        minimum_words=max(1, math.ceil(reference_words * CLEANUP_MIN_LENGTH_RATIO)),
    )


class Organizer:
    def __init__(
        self,
        *,
        mode: str,
        client: ChatClient | None,
        supervisor: LocalServerSupervisor | None,
        context_tokens: int,
        max_output_tokens: int,
        annotate: bool = False,
    ) -> None:
        self.mode = mode
        self.client = client
        self.supervisor = supervisor
        self.context_tokens = context_tokens
        self.max_output_tokens = max_output_tokens
        self.annotate = annotate
        self.status_label_callback: Callable[[str], None] | None = None
        self.status_note_callback: Callable[[str], None] | None = None

    def close(self) -> None:
        """Stop the supervised local server, if this organizer owns one."""
        if self.supervisor is not None:
            self.supervisor.close()

    def cleanup(
        self, sources: list[Transcript], plan: CleanupRequestPlan | None = None
    ) -> CleanupOutcome:
        """Clean the sources; ``plan`` (if the caller already computed one for the
        same sources) avoids rebuilding the prompt over the full transcript text."""
        sources = [s for s in sources if s.text.strip()]
        sources = _dedupe_sources(sources)
        started = time.monotonic()
        outcome = self._cleanup_inner(sources, plan)
        # Annotate only a usable result: there is nothing to audit on an error or a
        # truncated transcript, and the sources must actually disagree to have a signal.
        if (
            self.annotate
            and self.mode == "llama"
            and outcome.text
            and not outcome.error
            and len(sources) > 1
        ):
            self._annotate(outcome, sources)
        outcome.seconds = round(time.monotonic() - started, 3)
        return outcome

    def _annotate(self, outcome: CleanupOutcome, sources: list[Transcript]) -> None:
        """Mark the claims the sources do not jointly support (see annotate.py).

        Best-effort by construction: the transcript is already complete and correct
        without markers, so a failure is recorded and the text left alone.
        """
        assert self.client is not None
        started = time.monotonic()
        result = annotation.annotate(
            self.client,
            cleaned_text=outcome.text,
            sources=sources,
            context_tokens=self.context_tokens,
        )
        outcome.annotation_seconds = round(time.monotonic() - started, 3)
        outcome.annotation_error = result.error
        outcome.annotation_rejected_citations = result.rejected_citations
        if result.total:
            outcome.text_before_annotation = outcome.text
            outcome.text = result.text
            outcome.annotation_count = result.total
            outcome.annotation_appendix_count = result.appended_count
            outcome.annotation_notes = result.notes

    def _cleanup_inner(
        self, sources: list[Transcript], plan: CleanupRequestPlan | None
    ) -> CleanupOutcome:
        if not sources:
            return CleanupOutcome(method="empty")
        if self.mode == "off":
            return CleanupOutcome(text=sources[0].text, method="off")
        if self.mode == "heuristic":
            return CleanupOutcome(text=heuristic_cleanup(sources[0].text), method="heuristic")
        try:
            return self._cleanup_with_llm(sources, plan)
        except PromptTooLargeError as exc:
            return CleanupOutcome(method="skipped-too-large", error=str(exc))
        except Exception as exc:
            return CleanupOutcome(method="error", error=f"cleanup request failed: {exc}")

    def _cleanup_with_llm(
        self, sources: list[Transcript], plan: CleanupRequestPlan | None
    ) -> CleanupOutcome:
        assert self.client is not None
        if self.supervisor is not None:
            self.supervisor.ensure_running()
        if plan is None:
            plan = cleanup_request_plan(
                sources,
                context_tokens=self.context_tokens,
                max_output_tokens=self.max_output_tokens,
            )
        if plan.estimated_prompt_tokens + plan.requested_output_tokens > plan.token_budget:
            raise PromptTooLargeError(
                "estimated cleanup prompt is too large "
                f"(~{plan.estimated_prompt_tokens} input + {plan.requested_output_tokens} "
                f"output tokens > budget {plan.token_budget} of context {self.context_tokens}); "
                "raise --organizer-context-tokens or use --organizer-provider openrouter"
            )
        timeout = request_timeout_seconds(
            configured_timeout=self.client.timeout,
            estimated_prompt_tokens=plan.estimated_prompt_tokens,
            requested_output_tokens=plan.requested_output_tokens,
        )
        label_callback = self.status_label_callback
        note_callback = self.status_note_callback
        response = self.client.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": plan.user_prompt},
            ],
            max_tokens=plan.requested_output_tokens,
            timeout=timeout,
            on_model_attempt=(
                # Just the model name — the status phase already says "Cleanup".
                (lambda model: label_callback(model))
                if label_callback
                else None
            ),
            on_model_failure=(
                (
                    lambda model, next_model: note_callback(
                        f"Cleanup LM failed: {model}; trying {next_model}"
                    )
                )
                if note_callback
                else None
            ),
        )
        outcome = CleanupOutcome(
            text=response.content.strip(),
            method="llama",
            served_model=response.served_model,
            finish_reason=response.finish_reason,
            estimated_prompt_tokens=plan.estimated_prompt_tokens,
            requested_output_tokens=plan.requested_output_tokens,
            request_timeout=round(timeout, 3),
        )
        if not outcome.text:
            outcome.method = "empty"
            outcome.error = "cleanup model returned empty output"
            return outcome
        if response.finish_reason == "length":
            outcome.flagged_short = True
            outcome.error = (
                "cleanup output was truncated at the output token limit "
                f"({plan.requested_output_tokens} tokens); the transcript is incomplete"
            )
            return outcome
        cleaned_words = count_words(outcome.text)
        flags: list[str] = []
        if cleaned_words < plan.minimum_words:
            flags.append(
                f"suspiciously short ({cleaned_words} words < {plan.minimum_words} expected "
                f"from the average source length)"
            )
        if _ends_mid_sentence(outcome.text):
            # finish_reason="length" (handled above) is the clean truncation signal, but
            # some providers report "stop" on a stream that was cut off anyway; a dangling
            # final word catches those.
            flags.append(
                f'ends mid-sentence ("...{_tail_snippet(outcome.text)}"), so the model may '
                f"have been cut off before finishing"
            )
        if flags:
            # A warning, not a gate: the output is still the best cleanup we have, so it is
            # kept and written — only flagged for review, never discarded.
            outcome.flagged_short = True
            outcome.warning = (
                "cleanup output flagged — " + "; ".join(flags) + "; review before trusting it"
            )
        return outcome


_SENTENCE_FINAL = ".?!…"
_TRAILING_CLOSERS = "\"')]}”’»"


def _ends_mid_sentence(text: str) -> bool:
    """True when cleaned text ends without sentence-final punctuation.

    A complete cleanup ends on '.', '?', '!' or '…' (optionally wrapped in a closing
    quote/bracket). A dangling word like "And" instead signals the model was cut off
    mid-stream — a truncation that finish_reason does not always report. A faithful
    trail-off the speaker actually made ("...the time is...") still ends in '.', so it
    is not flagged.
    """
    stripped = text.rstrip()
    while stripped and stripped[-1] in _TRAILING_CLOSERS:
        stripped = stripped[:-1].rstrip()
    if not stripped:
        return False
    return stripped[-1] not in _SENTENCE_FINAL


def _tail_snippet(text: str, words: int = 6) -> str:
    return " ".join(text.split()[-words:])


def _dedupe_sources(sources: list[Transcript]) -> list[Transcript]:
    """Drop sources whose text duplicates an earlier one (provenance is kept

    in the session transcript list; the prompt doesn't need the same text
    twice)."""
    seen: set[str] = set()
    unique: list[Transcript] = []
    for source in sources:
        key = source.text.strip()
        if key in seen:
            continue
        seen.add(key)
        unique.append(source)
    return unique


def build_organizer(config: "Config") -> Organizer:
    """Wire up the organizer stack for the resolved config.

    The returned Organizer owns its supervisor (if any); callers release the
    local server with organizer.close()."""
    if config.organizer.mode != "llama":
        return Organizer(
            mode=config.organizer.mode,
            client=None,
            supervisor=None,
            context_tokens=config.organizer.context_tokens,
            max_output_tokens=config.organizer.max_output_tokens,
        )
    client = ChatClient(
        api_base=config.organizer.api_base,
        models=config.organizer.models,
        timeout=config.organizer.timeout,
        auth_env=config.organizer.auth_env,
        reasoning_effort="none" if config.organizer.provider == "openrouter" else None,
    )
    annotate = config.organizer.annotate
    supervisor: LocalServerSupervisor | None = None
    if config.organizer.provider == "local":
        launch_command = config.organizer.server_command
        if launch_command is None:
            model_path = config.organizer.gguf or DEFAULT_GGUF_MODEL
            have_server = bool(shutil.which("llama-server"))
            # Auto-install the bundled default quant on a cache miss (consent-gated,
            # like the ASR models). A user-supplied --organizer-gguf is never fetched
            # — only the default has a known source.
            if have_server and config.organizer.gguf is None and not model_path.exists():
                ensure_default_cleanup_model(auto_yes=config.auto_download)
            launch_command = default_server_command(
                context_tokens=config.organizer.context_tokens,
                gpu_layers=config.organizer.gpu_layers,
                kv_offload=config.organizer.kv_offload,
                model_path=model_path,
            )
            if launch_command is None and have_server and not model_path.exists():
                # Reached only if auto-install was declined/unavailable or a custom
                # --organizer-gguf path is missing — point the user at the fix.
                print(
                    f"note: local cleanup model not found at {model_path}. "
                    "Place a GGUF there (or pass --organizer-gguf / --organizer-server-command), "
                    "or use --online-free / --online-paid for OpenRouter cleanup. Falling back "
                    "to the primary transcript for now.",
                    file=sys.stderr,
                )
        supervisor = LocalServerSupervisor(
            launch_command=launch_command,
            healthcheck_url=client.models_url(),
        )
    return Organizer(
        mode=config.organizer.mode,
        client=client,
        supervisor=supervisor,
        context_tokens=config.organizer.context_tokens,
        max_output_tokens=config.organizer.max_output_tokens,
        annotate=annotate,
    )
