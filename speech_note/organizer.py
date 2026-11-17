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
from .chat import ChatClient, ChatResponse, request_timeout_seconds
from .config import (
    CLEANUP_MIN_LENGTH_RATIO,
    DEFAULT_GGUF_MODEL,
    DEFAULT_LOCAL_API_BASE,
    OPENROUTER_ANNOTATION_MODELS,
    OPENROUTER_FREE_ANNOTATION_MODELS,
    ORGANIZER_CONTEXT_SAFETY,
    ORGANIZER_MIN_OUTPUT_TOKENS,
)
from .llama_server import (
    LocalServerSupervisor,
    default_server_command,
    ensure_default_cleanup_model,
    preferred_model_selected,
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
    # Used to check the returned transcript for suspiciously short output.
    reference_words: int
    minimum_words: int


SYSTEM_PROMPT = """Turn the supplied transcripts of the same recording into one readable, near-verbatim transcript. Someone reading it should be able to follow the whole conversation in the speakers’ own words, including the steps by which they reached a conclusion. Return only the finished transcript.

Reconstruct the conversation in order. Find where the sources describe the same moment, combine what they heard, and render that moment once. Recognizers differ in what they pick up. When one contains a passage and the others resume later, keep the passage if it makes sense in context. Agreement helps resolve competing readings of a phrase. Before returning the transcript, check each source for details that disappeared between shared passages. Even an incidental remark belongs if someone said it.

Where the sources disagree, consider the nearby exchange, usage elsewhere in the recording, and any source-specific reliability notes. A reply can clarify what the preceding question was. Prefer the wording that best accounts for the evidence, even when a different version reads more smoothly. Preserve a speaker’s factual mistakes, uncertainty, and changes of mind. Correct errors introduced by transcription. If none of the readings is intelligible, mark the uncertainty briefly rather than supplying a confident guess.

Edit speech into readable prose. Remove empty fillers, stutters, accidental duplicates, and false starts that the speaker immediately replaces. Keep a restart when it changes the claim, a repetition when it adds emphasis, and a sound when it conveys something the participants refer to. Preserve substantive details, qualifications, examples, side conversations, and the distinction between a question and its answer. Even in a long recording, give later passages the same care as the opening. Avoid padding the result with recognizer loops or compressing a stretch of dialogue into a description of its topic.

Assess who spoke separately from what was said. Keep trustworthy names, speaker IDs, and attribution notes even when only one source supplies them. Consider whether the labels consistently track people across the exchange. A recognizer can split one person’s sentence between several IDs or put two people under one label. Where you can resolve the attribution, use it. Where you cannot, keep the speech and leave the attribution open.

Make the structure of the speech visible on the page. Use section headings for distinct topics, lists when someone enumerates points, and emphasis where the delivery or wording warrants it. When someone reads from a slide or another text, let the layout show where the reading begins and their commentary resumes. Use your judgment about which choices help here. Keep the speaker’s words, sequence, and level of detail throughout."""


_EXAMPLE_MESSAGES = (
    ("user", """Source 1:
Lin: Did both sensors fail?
Jo: I thought so. But after we swapped the cables,
[Speaker 6] the left one came back. The right one still reads zero.
Lin: So the right sensor is broken?
Jo: Maybe. We haven't tested the spare cable.

Source 2:
Did both sensors fail? I thought so, but after we swapped the cables the left one came back. The right one still reads zero. So the right sensor is broken? Maybe, we haven't tested the spare cable."""),
    ("assistant", """Lin: Did both sensors fail?

Jo: I thought so. But after we swapped the cables, the left one came back. The right one still reads zero.

Lin: So the right sensor is broken?

Jo: Maybe. We haven't tested the spare cable."""),
    ("user", """Source 1:
there are two things for the meeting first um bring the green the green folder on Friday. Actually Thursday. The lease is in there too. Second call Sal before you leave. Before is the important part.

Source 2:
There are two things for the meeting. First bring the green holder on Friday. Actually Thursday. Second call Sal before you leave. Before is the important part."""),
    ("assistant", """There are two things for the meeting.

1. Bring the green folder on Friday. Actually, Thursday. The lease is in there too.
2. Call Sal **before** you leave. Before is the important part."""),
)


def cleanup_messages(plan: "CleanupRequestPlan") -> list[dict[str, str]]:
    """Messages shared by the live cleanup request and its exported prompt."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *({"role": role, "content": text} for role, text in _EXAMPLE_MESSAGES),
        {"role": "user", "content": plan.user_prompt},
    ]


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
    """Present each source with its provenance and original text."""
    source_blocks = []
    for index, source in enumerate(sources, start=1):
        header = f"Source {index} — {source.label} ({source.model})"
        if source.quality_hint:
            header += f" — {source.quality_hint}"
        header += f" — {len(source.text)} chars / {source.words} words"
        source_blocks.append(f"{header}:\n{source.text}")
    return "\n\n".join([*source_blocks, "Write the complete transcript. Carry through the details each source contributes, and use formatting to bring out the speaker’s structure and emphasis."])


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
        estimate_text_tokens(SYSTEM_PROMPT)
        + sum(estimate_text_tokens(text) for _, text in _EXAMPLE_MESSAGES)
        + estimate_text_tokens(user_prompt)
        + 32
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
        output_cap_limited=uncapped_output_tokens > max_output_tokens,
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
        annotation_client: ChatClient | None = None,
    ) -> None:
        self.mode = mode
        self.client = client
        self.supervisor = supervisor
        self.context_tokens = context_tokens
        self.max_output_tokens = max_output_tokens
        self.annotate = annotate
        # The audit runs on its own model chain when one is provided: the measured
        # best auditors are not the measured best cleaner, and a separate judge also
        # means cleanup never grades its own work.
        self.annotation_client = annotation_client or client
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
        for client in (self.client, self.annotation_client):
            if isinstance(client, ChatClient):
                client.last_attempts = []
        started = time.monotonic()
        outcome = self._cleanup_inner(sources, plan)
        if isinstance(self.client, ChatClient):
            outcome.attempts = list(self.client.last_attempts)
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

        Best-effort by construction: cleanup has passed its completion checks;
        an audit failure is recorded and the text left alone.
        """
        assert self.annotation_client is not None
        if self.status_label_callback is not None:
            # The audit runs inside the "Cleanup" status phase; without this the
            # spinner claims cleanup is still generating for the extra minute.
            self.status_label_callback("auditing uncertainty")
        started = time.monotonic()
        result = annotation.annotate(
            self.annotation_client,
            cleaned_text=outcome.text,
            sources=sources,
            context_tokens=self.context_tokens,
        )
        if isinstance(self.annotation_client, ChatClient):
            outcome.annotation_attempts = list(self.annotation_client.last_attempts)
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
            return CleanupOutcome(
                method="error", error=f"cleanup request failed: {exc}",
                flagged_short=any(term in str(exc) for term in ("truncated", "mid-sentence", "suspiciously short")),
            )

    def _cleanup_with_llm(
        self, sources: list[Transcript], plan: CleanupRequestPlan | None
    ) -> CleanupOutcome:
        assert self.client is not None
        if plan is None:
            plan = cleanup_request_plan(
                sources,
                context_tokens=self.context_tokens,
                max_output_tokens=self.max_output_tokens,
            )
        if plan.output_cap_limited:
            raise PromptTooLargeError("estimated cleanup output exceeds the configured output token cap")
        if plan.estimated_prompt_tokens + plan.requested_output_tokens > plan.token_budget:
            raise PromptTooLargeError(
                "estimated cleanup prompt is too large "
                f"(~{plan.estimated_prompt_tokens} input + {plan.requested_output_tokens} "
                f"output tokens > budget {plan.token_budget} of context {self.context_tokens}); "
                "raise --organizer-context-tokens or use --organizer-provider openrouter"
            )
        if self.supervisor is not None:
            self.supervisor.ensure_running()
        timeout = request_timeout_seconds(
            configured_timeout=self.client.timeout,
            estimated_prompt_tokens=plan.estimated_prompt_tokens,
            requested_output_tokens=plan.requested_output_tokens,
        )
        label_callback = self.status_label_callback
        note_callback = self.status_note_callback
        response = self.client.chat(
            cleanup_messages(plan),
            max_tokens=plan.requested_output_tokens,
            timeout=timeout,
            validate_response=lambda reply: cleanup_response_error(reply, plan),
            on_model_attempt=(
                # Just the model name — the status phase already says "Cleanup".
                (lambda model: label_callback(model))
                if label_callback
                else None
            ),
            on_model_failure=(
                (
                    lambda model, next_model: note_callback(
                        f"Cleanup LM failed: {model}: "
                        f"{self.client.last_attempts[-1].get('error', 'unusable response')}; "
                        f"trying {next_model}"
                    )
                )
                if note_callback
                else None
            ),
        )
        outcome = CleanupOutcome(
            text=response.content.strip(),
            method="llama",
            requested_model=response.requested_model,
            served_model=response.served_model,
            finish_reason=response.finish_reason,
            estimated_prompt_tokens=plan.estimated_prompt_tokens,
            requested_output_tokens=plan.requested_output_tokens,
            request_timeout=round(timeout, 3),
        )
        if self.supervisor is not None:
            outcome.checkpoint, outcome.checkpoint_source = self.supervisor.checkpoint_identity(response.served_model)
        # Also enforce the contract for alternate clients implementing chat().
        outcome.error = cleanup_response_error(response, plan)
        outcome.flagged_short = outcome.error is not None
        if outcome.error:
            outcome.text = ""
        return outcome


def cleanup_response_error(response: ChatResponse, plan: CleanupRequestPlan) -> str | None:
    """Reject incomplete cleanup inside the model fallback loop, before annotation."""
    failure = response.failure_reason()
    if failure:
        return failure
    text = response.content.strip()
    if not text:
        return "cleanup model returned empty output"
    flags = []
    words = count_words(text)
    if words < plan.minimum_words:
        flags.append(f"suspiciously short ({words} words < {plan.minimum_words} expected)")
    if _ends_mid_sentence(text):
        flags.append(f'ends mid-sentence ("...{_tail_snippet(text)}")')
    return "incomplete cleanup: " + "; ".join(flags) if flags else None


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
    """Drop sources whose text duplicates an earlier one.

    Provenance is kept in the session transcript list; the prompt does not need the
    same text twice.
    """
    seen: set[str] = set()
    unique: list[Transcript] = []
    for source in sources:
        key = source.text.strip()
        if key in seen:
            continue
        seen.add(key)
        unique.append(source)
    return unique


def annotation_models(cleanup_models: "tuple[str, ...]") -> tuple[str, ...]:
    """The auditor chain for a cleanup chain: free judges for an all-free run.

    Read off the cleanup chain rather than off the connectivity preset, so an
    explicit --organizer-model list of free endpoints gets the same treatment as
    --online-free. See the two catalog lists for what is in each chain and why.
    """
    if cleanup_models and all(model.endswith(":free") for model in cleanup_models):
        return tuple(OPENROUTER_FREE_ANNOTATION_MODELS)
    return tuple(OPENROUTER_ANNOTATION_MODELS)


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
    annotation_client: ChatClient | None = None
    if config.organizer.provider == "openrouter":
        # A dedicated auditor chain (see OPENROUTER_ANNOTATION_MODELS for the
        # measurements behind it). Local stays on the one loaded model.
        #
        # A run whose cleanup chain is entirely free audits on free models too. The
        # paid chain is the better judge, but billing a free-preset run for the audit
        # — and sending the transcript to endpoints its own privacy notice never
        # named — is not the preset's bargain.
        annotation_client = ChatClient(
            api_base=config.organizer.api_base,
            models=annotation_models(config.organizer.models),
            timeout=config.organizer.timeout,
            auth_env=config.organizer.auth_env,
            reasoning_effort="low",
        )
    annotate = config.organizer.annotate
    supervisor: LocalServerSupervisor | None = None
    if config.organizer.provider == "local":
        launch_command = config.organizer.server_command
        launch_checkpoint = None
        if launch_command is None and config.organizer.api_base == DEFAULT_LOCAL_API_BASE:
            model_path = config.organizer.gguf or DEFAULT_GGUF_MODEL
            have_server = bool(shutil.which("llama-server"))
            # Auto-install the bundled default quant on a cache miss (consent-gated,
            # like the ASR models). A user-supplied --organizer-gguf is never fetched
            # — only the default has a known source.
            if (
                have_server and config.organizer.gguf is None and not model_path.exists()
                and not preferred_model_selected(config.organizer.context_tokens)
            ):
                ensure_default_cleanup_model(
                    auto_yes=config.auto_download,
                    interactive=False if config.full_auto else None,
                )
            launch_command = default_server_command(
                context_tokens=config.organizer.context_tokens,
                gpu_layers=config.organizer.gpu_layers,
                kv_offload=config.organizer.kv_offload,
                model_path=config.organizer.gguf,
            )
            if launch_command:
                launch_checkpoint = launch_command[launch_command.index("--model") + 1]
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
            launch_checkpoint=launch_checkpoint,
            healthcheck_url=client.models_url(),
        )
    return Organizer(
        mode=config.organizer.mode,
        client=client,
        supervisor=supervisor,
        context_tokens=config.organizer.context_tokens,
        max_output_tokens=config.organizer.max_output_tokens,
        annotate=annotate,
        annotation_client=annotation_client,
    )
