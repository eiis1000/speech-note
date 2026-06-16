"""Transcript cleanup via an OpenAI-compatible chat endpoint.

Three separated concerns:
  LocalServerSupervisor — owns the optional llama-server child process.
  ChatClient            — HTTP: auth, model fallback, finish_reason, the model
                          the response says it served (not the one we asked for).
  Organizer             — builds the prompt from Transcript sources (named, with
                          quality hints) and applies the length sanity check as
                          a *warning*, never a gate.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import requests

from .config import (
    CLEANUP_MIN_LENGTH_RATIO,
    CLEANUP_TARGET_LENGTH_RATIO,
    DEFAULT_GGUF_MODEL,
    ORGANIZER_CONTEXT_SAFETY,
    ORGANIZER_MAX_REQUEST_TIMEOUT,
    ORGANIZER_MIN_REQUEST_TIMEOUT,
)
from .model import CleanupOutcome, Transcript
from .terminal import debug_log
from .textproc import count_words, estimate_text_tokens, heuristic_cleanup

if TYPE_CHECKING:
    from .cli import Config


class PromptTooLargeError(RuntimeError):
    pass


def default_server_command(
    *,
    context_tokens: int,
    gpu_layers: str = "auto",
    kv_offload: bool = True,
    model_path: Path | None = None,
) -> list[str] | None:
    model_path = model_path or DEFAULT_GGUF_MODEL
    llama_server = shutil.which("llama-server")
    if not llama_server or not model_path.exists():
        return None
    command = [
        llama_server,
        "--model", str(model_path),
        "--host", "127.0.0.1",
        "--port", "8011",
        "--parallel", "1",
        "--ctx-size", str(context_tokens),
        "--gpu-layers", str(gpu_layers),
        "--reasoning", "off",
        "--jinja",
        "--no-warmup",
    ]
    if not kv_offload:
        # Escape hatch for older llama.cpp/Vulkan stacks where allocating the
        # split SWA/shared KV cache on the GPU hung during slot init. Fixed by
        # b9190 (measured: KV on GPU is ~1.5x faster generation, no hang), so
        # KV offload is on by default; --no-organizer-kv-offload restores this.
        command.append("--no-kv-offload")
    return command


def request_timeout_seconds(
    *,
    configured_timeout: float,
    estimated_prompt_tokens: int,
    requested_output_tokens: int,
) -> float:
    """Scale the request timeout with the work being asked for."""
    prompt_seconds = estimated_prompt_tokens / 250.0
    output_seconds = requested_output_tokens / 12.0
    estimated = 15.0 + prompt_seconds + output_seconds
    return min(
        ORGANIZER_MAX_REQUEST_TIMEOUT,
        max(configured_timeout, ORGANIZER_MIN_REQUEST_TIMEOUT, estimated),
    )


class LocalServerSupervisor:
    """Launches and supervises a local llama-server when one isn't running."""

    def __init__(self, *, launch_command: list[str] | None, healthcheck_url: str) -> None:
        self.launch_command = launch_command
        self.healthcheck_url = healthcheck_url
        self.process: subprocess.Popen[str] | None = None
        self.log_path: Path | None = None
        self._log_handle = None
        self._lock = threading.Lock()

    def is_healthy(self) -> bool:
        try:
            return requests.get(self.healthcheck_url, timeout=1.5).ok
        except requests.RequestException:
            return False

    def ensure_running(self, *, startup_timeout: float = 45.0) -> None:
        with self._lock:
            if self.is_healthy():
                return
            if not self.launch_command:
                return
            if self.process is None or self.process.poll() is not None:
                fd, log_name = tempfile.mkstemp(prefix="speech-note-llama-server-", suffix=".log")
                os.close(fd)
                self.log_path = Path(log_name)
                self._log_handle = self.log_path.open("a", encoding="utf-8")
                self.process = subprocess.Popen(
                    self.launch_command,
                    stdout=self._log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    text=True,
                )
            deadline = time.monotonic() + startup_timeout
            while time.monotonic() < deadline:
                if self.process is not None and self.process.poll() is not None:
                    raise RuntimeError("organizer server exited during startup" + self._log_tail_suffix())
                if self.is_healthy():
                    return
                time.sleep(0.5)
            raise RuntimeError("organizer server did not become ready" + self._log_tail_suffix())

    def close(self) -> None:
        with self._lock:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None

    def log_tail(self, max_chars: int = 4000) -> str:
        if self._log_handle is not None:
            with contextlib.suppress(Exception):
                self._log_handle.flush()
        if self.log_path is None or not self.log_path.exists():
            return ""
        with contextlib.suppress(Exception):
            return self.log_path.read_text(encoding="utf-8", errors="replace")[-max_chars:].strip()
        return ""

    def _log_tail_suffix(self) -> str:
        tail = self.log_tail()
        return f"; llama-server log tail: {' '.join(tail.split())}" if tail else ""


@dataclasses.dataclass
class ChatResponse:
    content: str
    served_model: str | None
    finish_reason: str | None


class ChatClient:
    """OpenAI-compatible chat client with model fallback."""

    TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

    def __init__(
        self,
        *,
        api_base: str,
        models: list[str],
        timeout: float,
        auth_env: str | None = None,
    ) -> None:
        self.api_base = api_base
        self.configured_models = [m for m in models if m]
        self.timeout = timeout
        self.auth_env = auth_env
        self.last_served_model: str | None = None
        self._catalog_checked = False
        self._available_models: list[str] | None = None

    def models_url(self) -> str:
        if self.api_base.endswith("/chat/completions"):
            return self.api_base[: -len("/chat/completions")] + "/models"
        return self.api_base.rstrip("/") + "/models"

    def request_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.auth_env is not None:
            api_key = os.environ.get(self.auth_env, "").strip()
            if not api_key:
                raise RuntimeError(f"{self.auth_env} is required for this organizer provider")
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def candidate_models(self) -> list[str]:
        """Configured models filtered against the live catalog when possible.

        The configured list is a preference order, not a claim of existence:
        entries missing from the provider's /models catalog are dropped (and
        the drop is logged). If the catalog cannot be fetched, fall back to the
        configured list unfiltered.
        """
        if self._catalog_checked:
            return self._available_models or self.configured_models
        self._catalog_checked = True
        try:
            response = requests.get(self.models_url(), timeout=10.0, headers=self.request_headers())
            response.raise_for_status()
            catalog = {entry.get("id") for entry in response.json().get("data", [])}
        except Exception as exc:
            debug_log(f"model catalog check failed; using configured list: {exc}")
            return self.configured_models
        available = [model for model in self.configured_models if model in catalog]
        dropped = [model for model in self.configured_models if model not in catalog]
        if dropped:
            debug_log(f"models missing from provider catalog, skipped: {dropped}")
        # An empty intersection means our list is fully stale; trying the
        # configured names anyway gives clearer errors than failing silently.
        self._available_models = available or self.configured_models
        return self._available_models

    def _is_transient(self, response: requests.Response) -> bool:
        return response.status_code in self.TRANSIENT_STATUS or (
            # Model-specific 404s mean the model vanished from the catalog
            # between our check and the request; fall through to the next.
            self.auth_env is not None and response.status_code == 404
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        timeout: float,
        on_model_attempt: Callable[[str], None] | None = None,
        on_model_failure: Callable[[str, str], None] | None = None,
    ) -> ChatResponse:
        failures: list[str] = []
        models = self.candidate_models()
        for index, model in enumerate(models):
            if on_model_attempt is not None:
                on_model_attempt(model)
            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": max_tokens,
            }
            try:
                response = requests.post(
                    self.api_base,
                    timeout=timeout,
                    headers=self.request_headers(),
                    data=json.dumps(payload),
                )
                if not response.ok:
                    body = " ".join(response.text.split())[:500]
                    failures.append(f"{model}: HTTP {response.status_code} {body}")
                    if index + 1 < len(models) and self._is_transient(response):
                        if on_model_failure is not None:
                            on_model_failure(model, models[index + 1])
                        continue
                    raise RuntimeError("; ".join(failures))
                data = response.json()
                choice = data["choices"][0]
                # Record what the server says it served; for llama-server the
                # requested name is decorative, the response is authoritative.
                served = data.get("model") or model
                self.last_served_model = served
                return ChatResponse(
                    content=_content_to_text(choice.get("message", {}).get("content")),
                    served_model=served,
                    finish_reason=choice.get("finish_reason"),
                )
            except requests.RequestException as exc:
                failures.append(f"{model}: {exc}")
                if index + 1 < len(models):
                    if on_model_failure is not None:
                        on_model_failure(model, models[index + 1])
                    continue
                raise RuntimeError("; ".join(failures)) from exc
        raise RuntimeError("; ".join(failures) or "no organizer models configured")


def _content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return "\n".join(part for part in parts if part).strip()
    return ""


SYSTEM_PROMPT = (
    "You are a conservative transcript repair engine. The output must be a readable "
    "clean transcript, not notes and not a summary. Treat each input transcript as a "
    "fallible view of the same speech. Reconstruct the shared chronological speech by "
    "comparing all sources. Preserve meaningful content and the speaker's intent while "
    "fixing punctuation, casing, and clear ASR mistakes. Remove non-substantive mic checks, "
    "bare acknowledgments, duplicate stutters, repeated short confirmations, and obvious "
    "recognition garbage. When uncertain whether a sentence contains meaning, keep it. "
    "Return only the final transcript text."
)


def build_user_prompt(sources: list[Transcript]) -> str:
    """Build the cleanup prompt with named sources.

    Sources are identified by what produced them, with quality hints when we
    have them — the model gets everything we know, instead of being told a lie
    that all sources are equally reliable.
    """
    reference_length = min((s.words for s in sources if s.words > 0), default=0)
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
        "Reconstruct one readable near-verbatim transcript.\n\n"
        "Source handling:\n"
        "- Each source below is a fallible transcription of the same recording; its header "
        "says what produced it and how reliable it is likely to be.\n"
        "- MUST align the sources chronologically and reconstruct the spoken content that "
        "best explains all of them.\n"
        "- SHOULD prefer wording supported by multiple sources or by local context, weighing "
        "sources by their stated reliability.\n"
        "- MAY use a minority source if it clearly fixes an ASR error.\n"
        "- When uncertain, SHOULD keep imperfect source wording rather than dropping or "
        "inventing content.\n\n"
        "Length expectation:\n"
        f"The shortest source is {reference_length} words. The repaired transcript should "
        f"normally be near {target_words} words or longer, and below {minimum_words} words "
        "almost certainly means content was skipped. "
        "MUST NOT pad the transcript with junk or repeated words merely to satisfy length. "
        "Meaningful coverage matters more than raw word count.\n\n"
        "Editing contract:\n"
        "- MUST preserve chronological transcript form.\n"
        "- MUST preserve all meaningful spoken content unless it is pure filler, a duplicate "
        "stutter, setup chatter, repeated acknowledgments, or obvious ASR garbage.\n"
        "- MUST preserve concrete details, examples, caveats, corrections, asides, "
        "transitions, questions, answers, instructions, and repeated meaningful points.\n"
        "- MUST NOT output a summary or outline, paraphrase away spoken content, or replace "
        "a stretch of speech with a shorter description.\n"
        "- MUST NOT add facts not supported by the sources.\n"
        "- MUST NOT write notes about the transcript or mention the editing process.\n"
        "- MUST remove long runs of repeated short acknowledgments such as okay/yeah/yes "
        "when they do not add new meaning.\n"
        "- MUST remove or compress mic checks, audibility checks, room logistics, apologies, "
        "and other setup chatter when they do not affect the substantive content.\n"
        "- MUST remove hallucinated or degenerate loops, including repeated okay/yeah/yes/no "
        "strings and repeated partial sentences.\n"
        "- MAY remove bare filler like um/uh, repeated stutters, and obvious recognition "
        "garbage.\n"
        "- MAY fix punctuation, casing, repeated fragments, and clear mishearings.\n"
        "- If a phrase is repeated, keep the first clear instance unless later repetitions "
        "change the meaning or emphasis.\n"
        "- MAY merge adjacent fragments only when they are clearly part of the same spoken "
        "thought.\n"
        "- MUST output only the final transcript.\n"
        "- SHOULD put a blank line after every 3 to 6 sentences at natural pauses.\n"
        "- MAY add light structure (speaker labels, headings) only if it clarifies the "
        "transcript, and MUST NOT let formatting replace, shorten, reorder, or summarize "
        "spoken content.\n\n"
        f"{joined_sources}\n\n"
        "Output the repaired near-verbatim transcript now. Start at the beginning of the "
        "recording and continue linearly until the end."
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
    ) -> None:
        self.mode = mode
        self.client = client
        self.supervisor = supervisor
        self.context_tokens = context_tokens
        self.max_output_tokens = max_output_tokens
        self.status_label_callback: Callable[[str], None] | None = None
        self.status_note_callback: Callable[[str], None] | None = None

    def cleanup(self, sources: list[Transcript]) -> CleanupOutcome:
        sources = [s for s in sources if s.text.strip()]
        sources = _dedupe_sources(sources)
        started = time.monotonic()
        outcome = self._cleanup_inner(sources)
        outcome.seconds = round(time.monotonic() - started, 3)
        return outcome

    def _cleanup_inner(self, sources: list[Transcript]) -> CleanupOutcome:
        if not sources:
            return CleanupOutcome(method="empty")
        if self.mode == "off":
            return CleanupOutcome(text=sources[0].text, method="off")
        if self.mode == "heuristic":
            return CleanupOutcome(text=heuristic_cleanup(sources[0].text), method="heuristic")
        try:
            return self._cleanup_with_llm(sources)
        except PromptTooLargeError as exc:
            return CleanupOutcome(method="skipped-too-large", error=str(exc))
        except Exception as exc:
            return CleanupOutcome(method="error", error=f"cleanup request failed: {exc}")

    def _cleanup_with_llm(self, sources: list[Transcript]) -> CleanupOutcome:
        assert self.client is not None
        if self.supervisor is not None:
            self.supervisor.ensure_running()
        user_prompt = build_user_prompt(sources)
        estimated_prompt_tokens = (
            estimate_text_tokens(SYSTEM_PROMPT) + estimate_text_tokens(user_prompt) + 32
        )
        largest_source_tokens = max(estimate_text_tokens(s.text) for s in sources)
        requested_output_tokens = min(
            self.max_output_tokens,
            max(1_024, largest_source_tokens * 2),
        )
        token_budget = max(256, int(self.context_tokens * ORGANIZER_CONTEXT_SAFETY))
        if estimated_prompt_tokens + requested_output_tokens > token_budget:
            raise PromptTooLargeError(
                "estimated cleanup prompt is too large "
                f"(~{estimated_prompt_tokens} input + {requested_output_tokens} output tokens "
                f"> budget {token_budget} of context {self.context_tokens}); "
                "raise --organizer-context-tokens or use --organizer-provider openrouter"
            )
        timeout = request_timeout_seconds(
            configured_timeout=self.client.timeout,
            estimated_prompt_tokens=estimated_prompt_tokens,
            requested_output_tokens=requested_output_tokens,
        )
        label_callback = self.status_label_callback
        note_callback = self.status_note_callback
        response = self.client.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=requested_output_tokens,
            timeout=timeout,
            on_model_attempt=(
                (lambda model: label_callback(f"Running cleanup LM: {model}"))
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
            estimated_prompt_tokens=estimated_prompt_tokens,
            requested_output_tokens=requested_output_tokens,
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
                f"({requested_output_tokens} tokens); the transcript is incomplete"
            )
            return outcome
        reference_length = min((s.words for s in sources if s.words > 0), default=0)
        minimum_words = max(1, math.ceil(reference_length * CLEANUP_MIN_LENGTH_RATIO))
        cleaned_words = count_words(outcome.text)
        flags: list[str] = []
        if cleaned_words < minimum_words:
            flags.append(
                f"suspiciously short ({cleaned_words} words < {minimum_words} expected "
                f"from the shortest source)"
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


def build_organizer(config: "Config") -> tuple[Organizer, LocalServerSupervisor | None]:
    """Wire up the organizer stack for the resolved config."""
    if config.organizer_mode != "llama":
        return (
            Organizer(
                mode=config.organizer_mode,
                client=None,
                supervisor=None,
                context_tokens=config.organizer_context_tokens,
                max_output_tokens=config.organizer_max_output_tokens,
            ),
            None,
        )
    client = ChatClient(
        api_base=config.organizer_api_base,
        models=config.organizer_models,
        timeout=config.organizer_timeout,
        auth_env=config.organizer_auth_env,
    )
    supervisor: LocalServerSupervisor | None = None
    if config.organizer_provider == "local":
        launch_command = config.organizer_server_command
        if launch_command is None:
            model_path = config.organizer_gguf or DEFAULT_GGUF_MODEL
            launch_command = default_server_command(
                context_tokens=config.organizer_context_tokens,
                gpu_layers=config.organizer_gpu_layers,
                kv_offload=config.organizer_kv_offload,
                model_path=model_path,
            )
            if launch_command is None and shutil.which("llama-server") and not model_path.exists():
                # Unlike the ASR models, the cleanup GGUF has no pinned download
                # source, so we can't auto-install it — point the user at the fix.
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
    organizer = Organizer(
        mode=config.organizer_mode,
        client=client,
        supervisor=supervisor,
        context_tokens=config.organizer_context_tokens,
        max_output_tokens=config.organizer_max_output_tokens,
    )
    return organizer, supervisor
