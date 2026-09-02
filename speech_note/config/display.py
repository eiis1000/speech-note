"""Presentation of ASR sources and cleanup models: short status-line names and
the per-model cleanup-prompt hints. Pure presentation, separate from the catalog
of what the sources actually are.
"""

from __future__ import annotations

from .catalog import AsrSource

__all__ = [
    "ASR_MODEL_NOTES",
    "ASR_BACKEND_SHORT_NAMES",
    "MODEL_SHORT_NAMES",
    "short_model_name",
    "short_source_name",
    "asr_source_presentation",
]


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
    # hosted ASR (the openrouter-stt backend)
    "openai/whisper-large-v3-turbo": "whisper-turbo",
    "nvidia/parakeet-tdt-0.6b-v3": "parakeet-v3",
    "microsoft/mai-transcribe-1.5": "mai",
    "openai/gpt-4o-mini-transcribe": "gpt4o-transcribe-mini",
    "openai/gpt-4o-transcribe": "gpt4o-transcribe",
    "qwen/qwen3-asr-flash-2026-02-10": "qwen3-asr",
    "google/gemini-3-flash-preview": "gemini",
    "deepseek/deepseek-v3.2": "deepseek",
    "anthropic/claude-haiku-4.5": "haiku",
    "openai/gpt-5.4-mini": "gpt-5.4-mini",
    "anthropic/claude-fable-5": "fable",
    "anthropic/claude-opus-4.6": "opus",
    "google/gemma-4-26b-a4b-it": "gemma-26b",
    "nvidia/nemotron-3-super-120b-a12b:free": "nemotron-super",
    "google/gemma-4-31b-it:free": "gemma-31b",
    "openai/gpt-oss-20b:free": "gpt-oss",
    "nvidia/nemotron-nano-9b-v2:free": "nemotron-nano",
    "inclusionai/ling-3.0-flash:free": "ling-flash",
    "nvidia/nemotron-3-ultra-550b-a55b:free": "nemotron-ultra",
    "local-gguf": "local",  # DEFAULT_LOCAL_MODEL_LABEL
}


def short_model_name(model: str) -> str:
    """A short status-line name for a cleanup-LM model id."""
    if model in MODEL_SHORT_NAMES:
        return MODEL_SHORT_NAMES[model]
    return model.split("/")[-1].split(":")[0]


def short_source_name(source: "AsrSource") -> str:
    """A short status-line name for an ASR source.

    Local backends are named by backend (one model each, in practice). The network
    backends are named by *model*, because the interesting distinction there is which
    hosted recognizer ran, not that it was hosted.
    """
    if source.backend in {"openrouter", "openrouter-stt"}:
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
