"""Static defaults and model tables.

Everything here is a default, not a runtime decision: runtime resolution
(per-backend model defaults, provider overrides, full-auto rerouting) lives in
cli.resolve_config so that provenance — did the user set this? — is tracked at
parse time instead of inferred from values later.
"""

from __future__ import annotations

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

# --- primary ASR ---
DEFAULT_ASR_BACKEND = "whisper-cpp"
DEFAULT_ASR_MODEL = "medium-q8_0"
LIVE_ASR_MODEL = "Systran/faster-whisper-tiny.en"
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

# --- secondary ASR ---
SECONDARY_BACKENDS = ("sherpa", "onnx", "crispasr", "ctc", "pocketsphinx")
# Backends that run as an external subprocess (their runtimes cannot share the
# main process: onnx-asr and crispasr are CLIs).
SUBPROCESS_SECONDARY_BACKENDS = frozenset({"onnx", "crispasr"})
SECONDARY_DEFAULT_MODELS = {
    "sherpa": "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
    "onnx": "nemo-parakeet-tdt-0.6b-v2",
    "crispasr": "parakeet-tdt-0.6b-v2-q4_k.gguf",
    "ctc": "nvidia/parakeet-ctc-0.6b",
    "pocketsphinx": "pocketsphinx-en-us",
}
# sherpa-onnx is the default secondary: same Parakeet-TDT-0.6B-v2 int8 model as
# the onnx backend, but the transducer decode loop runs in C++ (~3x faster on CPU
# than onnx-asr's Python per-segment loop — 48-70x realtime vs 18x, measured),
# with punctuation/casing and no length limit (silero-VAD segments stay short, so
# the FastConformer encoder never sees the long sequence that OOMs a single pass).
# It runs on the CPU, overlapping the GPU Whisper primary.
DEFAULT_SECONDARY_ASR_BACKEND = "sherpa"
DEFAULT_SECONDARY_ASR_DEVICE = "auto"
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
# outage still produces output. ASR is unaffected — no OpenRouter ASR is usable via
# the chat API (transcription models reject chat audio input) and the audio-LLMs that
# accept it truncate long audio, so local whisper+sherpa stays in every mode.
OPENROUTER_PAID_MODELS = [
    "deepseek/deepseek-v3.2",
    "google/gemini-3-flash-preview",
    *OPENROUTER_FREE_MODELS,
]
# Default for a plain --organizer-provider openrouter run: same as paid (best models,
# then free fallback).
OPENROUTER_PREFERRED_MODELS = list(OPENROUTER_PAID_MODELS)
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
DEFAULT_OPENROUTER_CONTEXT_TOKENS = 262_144
DEFAULT_OPENROUTER_MAX_OUTPUT_TOKENS = 32_768

ORGANIZER_MIN_REQUEST_TIMEOUT = 45.0
ORGANIZER_MAX_REQUEST_TIMEOUT = 1_800.0
ORGANIZER_CONTEXT_SAFETY = 0.92
# Minimum acceptable cleanup length as a fraction of the shortest source.
CLEANUP_MIN_LENGTH_RATIO = 0.7
CLEANUP_TARGET_LENGTH_RATIO = 0.85

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
