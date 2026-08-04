"""Static defaults, model tables, and the ASR source type.

Everything here is a default, not a runtime decision: runtime resolution
(per-backend model defaults, provider overrides, full-auto rerouting) lives in
cli.resolve_config so that provenance — did the user set this? — is tracked at
parse time instead of inferred from values later. Parsing of user-supplied specs
lives in config.parsing; status-line display names live in config.display.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

__all__ = [
    "SAMPLE_RATE", "FRAME_MS", "CHANNELS", "SAMPLE_WIDTH", "VAD_SUPPORTED_SAMPLE_RATES",
    "QUIET_AUDIO_PEAK_DBFS", "NORMALIZE_TARGET_RMS_DBFS", "NORMALIZE_PEAK_CEILING_DBFS",
    "NORMALIZE_MAX_GAIN_DB", "NORMALIZE_MIN_GAIN_DB", "NORMALIZE_LEVEL_PERCENTILE",
    "NORMALIZE_LEVEL_WINDOW_SECONDS", "NORMALIZE_GAIN_SMOOTH_SECONDS",
    "NORMALIZE_COMPRESS_THRESHOLD_DBFS", "NORMALIZE_COMPRESS_RATIO",
    "NORMALIZE_COMPRESS_KNEE_DB", "NORMALIZE_COMPRESS_LOOKAHEAD_MS",
    "NORMALIZE_COMPRESS_RELEASE_MS",
    "LIVE_ASR_MODEL", "OPENROUTER_ASR_MODEL", "DEFAULT_ASR_CPU_THREADS",
    "OPENROUTER_STT_MODEL", "OPENROUTER_STT_SECOND_MODEL", "OPENROUTER_STT_API_BASE",
    "OPENROUTER_STT_MP3_SAMPLE_RATE", "OPENROUTER_STT_RESPONSE_FORMAT",
    "WHISPER_CPP_MODEL_DIR", "WHISPER_CPP_MODEL_ALIASES", "BackendSpec", "ASR_BACKENDS",
    "AsrSource", "DEFAULT_ASR_SOURCES", "ONLINE_PAID_ASR_SOURCES", "CTC_STRIDE_SECONDS",
    "SHERPA_VAD_REPO", "SHERPA_VAD_FILE", "SHERPA_MAX_SEGMENT_SECONDS", "CTC_CHUNK_LENGTH_SECONDS",
    "DEFAULT_ORGANIZER_MODE", "DEFAULT_ORGANIZER_PROVIDER", "DEFAULT_LOCAL_API_BASE",
    "DEFAULT_LOCAL_MODEL_LABEL", "DEFAULT_GGUF_MODEL", "DEFAULT_GGUF_REPO", "DEFAULT_GGUF_FILE",
    "DEFAULT_GGUF_SIZE_HINT", "DEFAULT_ORGANIZER_CONTEXT_TOKENS", "DEFAULT_ORGANIZER_MAX_OUTPUT_TOKENS",
    "DEFAULT_OPENROUTER_API_BASE", "OPENROUTER_FREE_MODELS", "OPENROUTER_PAID_MODELS",
    "OPENROUTER_ASR_MP3_SAMPLE_RATE", "DEFAULT_OPENROUTER_ASR_MAX_OUTPUT_TOKENS",
    "OPENROUTER_ASR_MIN_TIMEOUT", "OPENROUTER_PREFERRED_MODELS", "OPENROUTER_API_KEY_ENV",
    "DEFAULT_OPENROUTER_CONTEXT_TOKENS", "DEFAULT_OPENROUTER_MAX_OUTPUT_TOKENS",
    "ORGANIZER_MIN_OUTPUT_TOKENS", "ORGANIZER_MIN_REQUEST_TIMEOUT", "ORGANIZER_MAX_REQUEST_TIMEOUT",
    "ANNOTATION_MAX_NOTES", "ANNOTATION_MAX_QUOTE_CHARS", "ANNOTATION_MAX_ALTERNATIVES",
    "ANNOTATION_MAX_OUTPUT_TOKENS",
    "ORGANIZER_CONTEXT_SAFETY",
    "CLEANUP_MIN_LENGTH_RATIO", "CLEANUP_TARGET_LENGTH_RATIO",
    "ARCHIVE_AUDIO_EXTENSIONS", "ARCHIVE_TRANSCRIPT_EXTENSIONS",
]

# --- audio ---
SAMPLE_RATE = 16_000
FRAME_MS = 30
CHANNELS = 1
SAMPLE_WIDTH = 2  # bytes per sample, int16 PCM
VAD_SUPPORTED_SAMPLE_RATES = (48_000, 32_000, 16_000, 8_000)
QUIET_AUDIO_PEAK_DBFS = -35.0

# --- leveling ---
# Normalization is a slow automatic gain ride plus a peak compressor. Three principles,
# each learned the hard way:
#
# 1. NOTHING IS EVER CUT OR CLASSIFIED. There is no speech/silence gate. Speech is
#    routinely *quieter than the noise around it* — a speaker walking into a shower, a
#    phone in a pocket — and it is still perfectly recoverable, so any attempt to label
#    frames as "speech" or "not speech" gets that case exactly backwards: it would pick
#    the noise as the reference and leave the real speech further below target.
# 2. THE GAIN VARIES OVER MINUTES, NOT SECONDS. A sustained level change should be
#    corrected — a few minutes of quiet should end up at normal level — while a brief
#    dip should be ridden through untouched, because chasing it produces pumping and
#    tells the ASR feature extractor that a syllable was a sentence.
# 3. THE GAIN CURVE IS ZERO-PHASE. We hold the whole recording, so the envelope is
#    computed from a *centred* window and smoothed forwards and backwards. A causal
#    filter lags the audio it is describing, which means it turns the gain up after the
#    quiet part has already gone by.
#
# Peaks are handled by a look-ahead compressor, never by reducing the overall gain. The
# old code took its peak term from the absolute maximum sample, so one bump vetoed the
# whole file: measured on two real recordings, it applied gains of 0.86 and 0.79 —
# handing ASR audio *quieter* than the input, on exactly the quiet recordings that
# needed the gain most.
NORMALIZE_TARGET_RMS_DBFS = -20.0
# The level estimate is a high percentile of frame loudness inside the window, not a
# mean and not a maximum. A mean is dragged down by pauses (a window that is half
# silence reads as quiet, so the gain overshoots); a maximum is set by the single
# loudest click. A percentile is robust to both and needs no gate.
NORMALIZE_LEVEL_PERCENTILE = 75.0
# Centred window for that percentile. Long, so the estimate is stable across pauses.
NORMALIZE_LEVEL_WINDOW_SECONDS = 90.0
# Additional zero-phase smoothing of the gain curve, applied as two successive centred
# moving averages (a triangular kernel). This is what makes the ride slow: a level
# change has to persist on this timescale before the gain fully follows it.
NORMALIZE_GAIN_SMOOTH_SECONDS = 45.0
# Total gain bounds, in dB.
NORMALIZE_MAX_GAIN_DB = 34.0
NORMALIZE_MIN_GAIN_DB = -12.0
# Look-ahead soft-knee compressor, applied after the gain ride to tame transients that
# would otherwise clip. Look-ahead means the reduction starts *before* the transient, so
# there is no click on the attack.
NORMALIZE_COMPRESS_THRESHOLD_DBFS = -14.0
NORMALIZE_COMPRESS_RATIO = 4.0
NORMALIZE_COMPRESS_KNEE_DB = 6.0
NORMALIZE_COMPRESS_LOOKAHEAD_MS = 20.0
NORMALIZE_COMPRESS_RELEASE_MS = 250.0
# Final safety limiter. With the compressor in front of it this should barely engage.
NORMALIZE_PEAK_CEILING_DBFS = -2.0

# --- ASR ---
# There is no "primary"/"secondary" ASR. A run transcribes the audio with an
# ordered *collection* of ASR sources and hands every transcript to the cleanup
# LM as a peer (in list order). Order is only a soft preference: it decides which
# transcript is surfaced as the raw output when cleanup is off or fails. Quality
# information is attached per-model and only when we have something to say (see
# config.display.ASR_MODEL_NOTES) — most sources carry no reliability claim at all.
LIVE_ASR_MODEL = "Systran/faster-whisper-tiny.en"
# The OpenRouter audio-LLM used by the "openrouter" backend: a whole recording is sent
# as chat input_audio and the reply is the transcript.
#
# CAUTION — measured 2026-08-02/03, two failure modes, both severe:
#   * On *unintelligible* audio it does not annotate the gap. It invents fluent,
#     confident, specific prose, differently on every call, and the cleanup stage then
#     copies that verbatim into the output. Fluent invention is far worse than garbled
#     output because neither the reader nor any later stage can tell it from real speech.
#   * On a 65-minute recording it degenerated: ~9,600 consecutive repetitions of one
#     word, half of its 19k-word output, burning to the token cap in 348 s for $0.20 —
#     against 8.1k plausible words in 16 s for $0.04 from a dedicated STT model.
#
# The earlier claim that "given the whole file it annotates non-speech instead of
# confabulating" holds only for clear, shorter audio, which is all it had been tested on.
# Kept selectable (--asr openrouter) but no longer any default: see OPENROUTER_STT_MODEL.
OPENROUTER_ASR_MODEL = "google/gemini-3-flash-preview"

# --- dedicated speech-to-text over the network (the "openrouter-stt" backend) ---
# OpenRouter's /audio/transcriptions endpoint. This did not exist when the audio-LLM
# route was chosen, which is why the old comment claimed remote ASR had to be chunked;
# it accepts a whole 65-minute recording in one multipart request.
#
# Measured on a hard, low-SNR recording: all 12 working models emitted visibly garbled
# text where the audio was unintelligible and NOT ONE invented a narrative — fluent
# confabulation is an audio-LLM behaviour, not an ASR behaviour. 11 of the 12
# independently recovered a passage the audio-LLM had replaced with fiction. That is
# also what makes cross-source disagreement a *usable* uncertainty signal downstream.
#
# whisper-large-v3-turbo leads: cheapest of the set (~$0.04 for 65 minutes, a third of
# the audio-LLM), byte-identical across repeat calls, and never truncated in any trial.
# parakeet-tdt-0.6b-v3 is the natural second — different vendor and training lineage, so
# its errors are uncorrelated, and it is the v3 of the model the local sherpa source runs
# at v2.
OPENROUTER_STT_MODEL = "openai/whisper-large-v3-turbo"
OPENROUTER_STT_SECOND_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
OPENROUTER_STT_API_BASE = "https://openrouter.ai/api/v1/audio/transcriptions"
# Rejected after measurement: deepgram/nova-3 silently dropped a hard passage;
# x-ai/grok-stt-1.0 truncated to a fifth of the recording; google/chirp-3 returns HTTP
# 400 through this endpoint; openai/whisper-large-v3 gave different word counts across
# batches (provider routing); openai/gpt-audio-mini degenerates into a repeated clause.
#
# Uploaded as multipart form-data, not base64 JSON: a 65-minute mp3 is ~17 MB and base64
# inflates it to ~23 MB, which the gateway rejects with a 502 (15.7 MB base64 still
# passed; 23 MB did not). Multipart carries no such overhead.
OPENROUTER_STT_MP3_SAMPLE_RATE = 16_000
# Only the whisper-family models accept response_format=verbose_json (which carries
# segment timings and avg_logprob); parakeet, qwen3-asr and gpt-4o-*-transcribe reject
# it with HTTP 400. Plain "json" is the portable request, so that is what we send.
OPENROUTER_STT_RESPONSE_FORMAT = "json"
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
    # A remote *dedicated* speech recognizer (OpenRouter /audio/transcriptions).
    "openrouter-stt": BackendSpec(OPENROUTER_STT_MODEL, in_process=True, device_kind="net"),
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
# --asr or a user config file (see config.parsing).
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

# Overlap between adjacent CTC long-form windows (each side), merged at the
# logit level by the HF ASR pipeline.
CTC_STRIDE_SECONDS = 5.0

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
# stride defaults to CTC_STRIDE_SECONDS. There is no length limit.
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
DEFAULT_OPENROUTER_MAX_OUTPUT_TOKENS = 65_536

# OpenAI-compatible endpoints may count hidden reasoning against max_tokens. A
# transcript-sized allowance alone is therefore insufficient even for short notes.
ORGANIZER_MIN_OUTPUT_TOKENS = 4_096

# --- uncertainty annotation ---
# The annotation pass is answered under a JSON *schema*, not merely "please emit JSON".
# That matters for more than tidiness: llama.cpp compiles a schema into a GBNF grammar
# and constrains sampling with it, so a small local model physically cannot emit an
# unterminated string. Without it, the bundled gemma quant looped inside the first
# entry's string until it hit the token cap and the whole reply was unusable.
#
# The bounds below are part of that guarantee — they are what makes the worst-case reply
# a bounded length instead of "until the model stops", so truncation stops being possible
# rather than being something to recover from afterwards.
ANNOTATION_MAX_NOTES = 25
ANNOTATION_MAX_QUOTE_CHARS = 200
ANNOTATION_MAX_ALTERNATIVES = 3
# Worst case is ANNOTATION_MAX_NOTES * (1 + MAX_ALTERNATIVES) * MAX_QUOTE_CHARS chars of
# payload, ~5k tokens; this leaves room for that plus the JSON scaffolding.
ANNOTATION_MAX_OUTPUT_TOKENS = 8_192
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
