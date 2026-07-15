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
