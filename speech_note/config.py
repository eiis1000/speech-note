"""Static defaults and model tables.

Everything here is a default, not a runtime decision: runtime resolution
(per-backend model defaults, provider overrides, full-auto rerouting) lives in
cli.resolve_config so that provenance — did the user set this? — is tracked at
parse time instead of inferred from values later.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

# --- audio ---
SAMPLE_RATE = 16_000
FRAME_MS = 30
CHANNELS = 1
SAMPLE_WIDTH = 2  # bytes per sample, int16 PCM
VAD_SUPPORTED_SAMPLE_RATES = (48_000, 32_000, 16_000, 8_000)
QUIET_AUDIO_PEAK_DBFS = -35.0

# Gain normalization targets (matches the old pipeline's behavior).
NORMALIZE_TARGET_RMS_DBFS = -20.0
NORMALIZE_PEAK_CEILING_DBFS = -2.0
NORMALIZE_MAX_GAIN = 12.0

# --- ASR ---
# There is no "primary"/"secondary" ASR. A run transcribes the audio with an
# ordered *collection* of ASR sources and hands every transcript to the cleanup
# LM as a peer (in list order). Order is only a soft preference: it decides which
# transcript is surfaced as the raw output when cleanup is off or fails. Quality
# information is attached per-model and only when we have something to say (see
# ASR_MODEL_NOTES) — most sources carry no reliability claim at all.
LIVE_ASR_MODEL = "Systran/faster-whisper-tiny.en"
# The OpenRouter audio-LLM used as a network ASR source (the "openrouter" backend).
# Gemini 3 Flash accepts a whole long recording as chat input_audio and transcribes
# it in a single request — verified on a 44-min file — and, given the whole file, it
# annotates non-speech instead of confabulating it (the failure mode of per-chunk
# audio-LLM transcription). It is never used unless explicitly selected (--asr
# openrouter) or defaulted in by --online-paid.
OPENROUTER_ASR_MODEL = "google/gemini-3-flash-preview"
# Bounded but not artificially capped at 8 like the old code was.
DEFAULT_ASR_CPU_THREADS = max(2, min(16, os.cpu_count() or 2))
WHISPER_CPP_MODEL_DIR = Path.home() / ".cache/whisper.cpp"
WHISPER_CPP_MODEL_ALIASES = {
    "Systran/faster-whisper-tiny.en": "tiny.en",
    "Systran/faster-whisper-base.en": "base.en",
    "Systran/faster-whisper-small.en": "small.en",
    "Systran/faster-whisper-medium.en": "medium.en",
    "openai/whisper-tiny.en": "tiny.en",
    "openai/whisper-base.en": "base.en",
    "openai/whisper-small.en": "small.en",
    "openai/whisper-medium.en": "medium.en",
    "medium-q8": "medium-q8_0",
}


@dataclasses.dataclass(frozen=True)
class BackendSpec:
    """Static facts about an ASR backend, independent of any one run."""

    default_model: str
    in_process: bool  # False = runs as an external subprocess CLI (onnx-asr, crispasr)
    device_kind: str  # "gpu" | "cpu": the device this backend naturally uses by default


# The full backend registry. Each is just an ASR source; none is privileged.
# sherpa-onnx runs the Parakeet-TDT-0.6B-v2 int8 transducer decode loop in C++
# (~3x faster on CPU than onnx-asr's Python per-segment loop — 48-70x realtime,
# measured), with punctuation/casing and no length limit, so it is a natural CPU
# companion to a GPU Whisper source (the two devices don't contend → they overlap).
ASR_BACKENDS: dict[str, BackendSpec] = {
    "whisper-cpp": BackendSpec("medium-q8_0", in_process=True, device_kind="gpu"),
    "faster-whisper": BackendSpec("Systran/faster-whisper-medium.en", in_process=True, device_kind="cpu"),
    "sherpa": BackendSpec("csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8", in_process=True, device_kind="cpu"),
    "ctc": BackendSpec("nvidia/parakeet-ctc-0.6b", in_process=True, device_kind="cpu"),
    "onnx": BackendSpec("nemo-parakeet-tdt-0.6b-v2", in_process=False, device_kind="cpu"),
    "crispasr": BackendSpec("parakeet-tdt-0.6b-v2-q4_k.gguf", in_process=False, device_kind="gpu"),
    "pocketsphinx": BackendSpec("pocketsphinx-en-us", in_process=True, device_kind="cpu"),
    # A remote audio-LLM (OpenRouter chat). device_kind "net" is neither GPU nor CPU,
    # so it forms its own scheduling group and overlaps the local sources for free.
    "openrouter": BackendSpec(OPENROUTER_ASR_MODEL, in_process=True, device_kind="net"),
}


@dataclasses.dataclass(frozen=True)
class AsrSource:
    """One member of the ASR collection: a backend, a model, and a device.

    ``device`` is the literal value handed to the backend ("auto" resolves to the
    backend's natural device). ``model`` empty means the backend's default model.
    """

    backend: str
    model: str = ""
    device: str = "auto"

    def resolved(self) -> "AsrSource":
        spec = ASR_BACKENDS[self.backend]
        model = self.model or spec.default_model
        device = self.device if self.device and self.device != "auto" else _default_device(self.backend)
        return AsrSource(self.backend, model, device)

    @property
    def device_kind(self) -> str:
        """'gpu' or 'cpu' — for scheduling: same-device sources can't overlap."""
        d = self.device.lower()
        if d == "cpu":
            return "cpu"
        if d in {"gpu", "vulkan", "auto"} or d.startswith("cuda") or d.isdigit():
            return "gpu" if d != "auto" else ASR_BACKENDS[self.backend].device_kind
        return ASR_BACKENDS[self.backend].device_kind


def _default_device(backend: str) -> str:
    """Backend-native default device string for the backend's natural device kind."""
    kind = ASR_BACKENDS[backend].device_kind
    if kind == "cpu":
        return "cpu"
    if kind == "net":
        return "net"
    if backend == "whisper-cpp":
        return "0"  # first GPU index
    if backend == "crispasr":
        return "vulkan"
    return "auto"


# The bundled default collection: GPU Whisper + CPU sherpa-onnx Parakeet. They use
# different devices, so they run concurrently for free. Overridable per run via
# --asr or a user config file (see parse_asr_sources / load_user_asr_sources).
DEFAULT_ASR_SOURCES: tuple[AsrSource, ...] = (
    AsrSource("whisper-cpp", "medium-q8_0", "gpu"),
    AsrSource("sherpa", "", "cpu"),
)

# The default collection under --online-paid: lead with the OpenRouter Gemini
# full-file source (best on hard/quiet audio in testing — one request, no chunking,
# and it annotates silence instead of confabulating), then keep the local
# whisper+sherpa pair as free corroborating peers and a network-outage fallback.
# "net" + "gpu" + "cpu" are three device groups, so all three overlap. Overridable
# by --asr or the user ASR config file, like any default collection.
ONLINE_PAID_ASR_SOURCES: tuple[AsrSource, ...] = (
    AsrSource("openrouter", OPENROUTER_ASR_MODEL, "net"),
    *DEFAULT_ASR_SOURCES,
)


def parse_asr_source(token: str) -> AsrSource:
    """Parse one ``backend[:model][@device]`` spec into a resolved AsrSource."""
    token = token.strip()
    if not token:
        raise ValueError("empty ASR source spec")
    rest, _, device = token.partition("@")
    backend, _, model = rest.partition(":")
    backend = backend.strip()
    if backend not in ASR_BACKENDS:
        raise ValueError(
            f"unknown ASR backend {backend!r}; choose from {', '.join(sorted(ASR_BACKENDS))}"
        )
    return AsrSource(backend, model.strip(), device.strip() or "auto").resolved()


def parse_asr_sources(tokens: list[str]) -> tuple[AsrSource, ...]:
    """Parse repeatable --asr values (each itself comma-separated) into a collection."""
    sources: list[AsrSource] = []
    for raw in tokens:
        for piece in raw.split(","):
            piece = piece.strip()
            if piece:
                sources.append(parse_asr_source(piece))
    return tuple(sources)


# A user config file listing the default ASR collection, one `backend[:model][@device]`
# spec per line (blank lines and # comments ignored). Mirrors USER_ENV_FILE: it lets a
# user change the default collection without editing source. CLI --asr overrides it.
USER_ASR_FILE = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "speech-note" / "asr"


def load_user_asr_sources(path: Path | None = None) -> tuple[AsrSource, ...]:
    """Read the ASR collection from USER_ASR_FILE; empty tuple if absent/empty."""
    path = path or USER_ASR_FILE
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return ()
    tokens = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return parse_asr_sources(tokens)


# Per-model presentation for the cleanup prompt: a friendly display name and an
# optional reliability hint shown to the cleanup LM alongside the transcript.
# Keyed by model. Models not listed get their plain model name and NO hint — we
# only annotate a source when we actually have something to say about it.
ASR_MODEL_NOTES: dict[str, dict[str, str]] = {
    "large-v3-turbo-q5_k": {
        "display": "Whisper Large V3 Turbo Q5_K",
        "hint": (
            "Warning: this model has a tendency to repeat words and phrases; if "
            "other transcripts do not corroborate a repetition, ignore it"
        ),
    },
}


# Short, hand-picked names for the live status line. Auto-deriving them is useless
# ("csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8" tells you nothing at a
# glance and blows out the line width), so these are curated. ASR sources are named
# by backend; the cleanup LM is named by model id. Unknowns fall back to the leaf.
ASR_BACKEND_SHORT_NAMES: dict[str, str] = {
    "whisper-cpp": "whisper",
    "faster-whisper": "faster-whisper",
    "sherpa": "parakeet",
    "ctc": "parakeet-ctc",
    "onnx": "parakeet-onnx",
    "crispasr": "crispasr",
    "pocketsphinx": "sphinx",
    # "openrouter" is named by its model (see MODEL_SHORT_NAMES).
}

MODEL_SHORT_NAMES: dict[str, str] = {
    "google/gemini-3-flash-preview": "gemini",
    "google/gemini-3.1-flash-lite": "gemini-lite",
    "deepseek/deepseek-v3.2": "deepseek",
    "openai/gpt-oss-120b:free": "gpt-oss",
    "nvidia/nemotron-3-super-120b-a12b:free": "nemotron-super",
    "qwen/qwen3-next-80b-a3b-instruct:free": "qwen3-next",
    "meta-llama/llama-3.3-70b-instruct:free": "llama-3.3",
    "google/gemma-4-31b-it:free": "gemma-4",
    "nousresearch/hermes-3-llama-3.1-405b:free": "hermes-3",
    "nvidia/nemotron-3-ultra-550b-a55b:free": "nemotron-ultra",
    "local-gguf": "local",  # DEFAULT_LOCAL_MODEL_LABEL
}


def short_model_name(model: str) -> str:
    """A short status-line name for a cleanup-LM model id."""
    if model in MODEL_SHORT_NAMES:
        return MODEL_SHORT_NAMES[model]
    return model.split("/")[-1].split(":")[0]


def short_source_name(source: "AsrSource") -> str:
    """A short status-line name for an ASR source: by backend, or by model for the
    network audio-LLM backend."""
    if source.backend == "openrouter":
        return short_model_name(source.model)
    return ASR_BACKEND_SHORT_NAMES.get(source.backend, source.backend)


def asr_source_presentation(model: str, effective_model: str | None = None) -> tuple[str, str | None]:
    """(display name, optional cleanup-prompt hint) for an ASR transcript.

    effective_model is the file that actually ran (e.g. the resolved whisper.cpp
    ggml name); it is used as the display name when the model has no note.
    """
    note = ASR_MODEL_NOTES.get(model)
    if note is not None:
        return note["display"], note["hint"]
    return (effective_model or model), None


SECONDARY_OVERLAP_SECONDS = 5.0

# sherpa-onnx VAD: silero, reusing the same ONNX file onnx-asr already caches so
# there is no second VAD source. Segments are capped well under the ~400s Parakeet
# position cliff to keep each decode O(short).
SHERPA_VAD_REPO = "istupakov/silero-vad-onnx"
SHERPA_VAD_FILE = "silero_vad.onnx"
SHERPA_MAX_SEGMENT_SECONDS = 20.0

# Parakeet CTC 0.6B has config.max_position_embeddings=5000 (~400s at 12.5
# frames/s), so a single forward pass over longer audio raises. Rather than
# capping or skipping, the HF ASR pipeline transcribes the whole file in
# chunk_length_s windows with stride_length_s overlap on each side and merges at
# the logit level. 240s ≈ 3000 positions, comfortably under the cliff; the
# stride defaults to SECONDARY_OVERLAP_SECONDS. There is no length limit.
CTC_CHUNK_LENGTH_SECONDS = 240.0

# --- cleanup LM ---
DEFAULT_ORGANIZER_MODE = "llama"
DEFAULT_ORGANIZER_PROVIDER = "local"
DEFAULT_LOCAL_API_BASE = "http://127.0.0.1:8011/v1/chat/completions"
DEFAULT_LOCAL_MODEL_LABEL = "local-gguf"
DEFAULT_GGUF_MODEL = Path.home() / ".cache/huggingface/gguf/gemma-4-E2B-it-UD-Q4_K_XL.gguf"
# Hugging Face source for the bundled cleanup quant, so the default model can be
# auto-installed (consent-gated) like the ASR models instead of being placed by
# hand. Unsloth's dynamic (UD) GGUF repo; the file name matches DEFAULT_GGUF_MODEL.
# Only the default quant is fetchable — a user-supplied --organizer-gguf is not.
DEFAULT_GGUF_REPO = "unsloth/gemma-4-E2B-it-GGUF"
DEFAULT_GGUF_FILE = DEFAULT_GGUF_MODEL.name
DEFAULT_GGUF_SIZE_HINT = "2.5 GB"
DEFAULT_ORGANIZER_CONTEXT_TOKENS = 65_536
DEFAULT_ORGANIZER_MAX_OUTPUT_TOKENS = 16_384

DEFAULT_OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"
# Model lists for the connectivity presets (--online-paid / --online-free) and the
# default openrouter ordering. The live /models catalog is consulted at request time
# and missing entries are dropped, and the client falls through the list on transient
# errors, so order = preference and length = resilience.
#
# Free endpoints flap (rate limits, capacity), so OPENROUTER_FREE_MODELS is a long,
# provider-diverse fallback chain: if one provider's free tier is down, the next is a
# different provider. Lead is gpt-oss-120b (strongest open model here). The big free
# flagship nemotron-ultra-550b is deliberately last — in the cleanup bake-off it
# bloated output ~2x and missed the fix, so it's a last resort, not a preference.
OPENROUTER_FREE_MODELS = [
    "openai/gpt-oss-120b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-4-31b-it:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
]
# Paid leads with the cleanup bake-off winners (2026-06-15, reconciling whisper +
# sherpa + phone on a hard journal): deepseek-v3.2 recovered the correct reading,
# stayed near-verbatim, and is the cheapest (~pennies/file); gemini-3-flash-preview
# also got it right and is fast. It then falls through to the free chain so a paid
# outage still produces output. ASR: the dedicated OpenRouter transcription models
# (whisper-large-v3, chirp, gpt-4o-transcribe, …) reject a long single request and
# must be chunked, but an audio-LLM — Gemini 3 Flash — transcribes a whole 44-min
# recording in one chat request (verified 2026-06-28). So --online-paid also leads
# ASR with the "openrouter" Gemini source (see ONLINE_PAID_ASR_SOURCES); the local
# whisper+sherpa remain as corroborating peers and offline fallback.
OPENROUTER_PAID_MODELS = [
    "deepseek/deepseek-v3.2",
    "google/gemini-3-flash-preview",
    *OPENROUTER_FREE_MODELS,
]

# OpenRouter ASR request shaping (the "openrouter" backend). The whole recording is
# transcoded to mono mp3 (a 16 kHz wav of a long file is too large to base64 into a
# JSON body), sent as chat input_audio, and the reply is the transcript.
OPENROUTER_ASR_MP3_SAMPLE_RATE = 16_000
DEFAULT_OPENROUTER_ASR_MAX_OUTPUT_TOKENS = 16_384
# A floor; the per-request timeout scales up with audio duration in build_transcriber.
OPENROUTER_ASR_MIN_TIMEOUT = 120.0
# Default for a plain --organizer-provider openrouter run: same as paid (best models,
# then free fallback).
OPENROUTER_PREFERRED_MODELS = list(OPENROUTER_PAID_MODELS)
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
DEFAULT_OPENROUTER_CONTEXT_TOKENS = 262_144
DEFAULT_OPENROUTER_MAX_OUTPUT_TOKENS = 32_768

ORGANIZER_MIN_REQUEST_TIMEOUT = 45.0
ORGANIZER_MAX_REQUEST_TIMEOUT = 1_800.0
ORGANIZER_CONTEXT_SAFETY = 0.92
# Cleanup length is judged against the MEAN source length, not the shortest or the
# longest. Cleanup reconstructs the UNION of what every source heard, so anchoring to
# the shortest would license dropping the extra real content a more sensitive source
# (e.g. the Gemini audio-LLM) caught — but anchoring to the longest over-counts when
# that source is long because it kept every disfluency, demanding a bloated output. The
# mean is robust to one filler-heavy source. These are the warning floor / soft target
# as fractions of the mean source length; both are review hints, never gates.
CLEANUP_MIN_LENGTH_RATIO = 0.6
CLEANUP_TARGET_LENGTH_RATIO = 0.8

# --- archive mode ---
ARCHIVE_AUDIO_EXTENSIONS = {".m4a", ".mp3", ".wav", ".ogg", ".opus", ".flac", ".aac", ".webm", ".mp4"}
ARCHIVE_TRANSCRIPT_EXTENSIONS = {".txt", ".srt", ".vtt", ".json"}


# A KEY=VALUE file (e.g. OPENROUTER_API_KEY=...) read at startup so secrets travel
# with speech-note regardless of the working directory or shell — kept out of the
# Nix store / git, unlike a direnv .envrc which only loads inside the repo tree.
USER_ENV_FILE = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "speech-note" / "env"


def load_user_env(path: Path | None = None) -> None:
    """Populate os.environ from USER_ENV_FILE without overriding the live shell.

    setdefault, not assignment: an explicitly exported variable always wins, so this
    is a fallback for shells that don't have the key, never an override.
    """
    path = path or USER_ENV_FILE
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            os.environ.setdefault(key, value)
