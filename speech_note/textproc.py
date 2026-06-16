"""Text utilities shared by the pipeline and the subprocess helpers."""

from __future__ import annotations

import math
import re

SPACE_RE = re.compile(r"\s+")
PARAKEET_TIMESTAMP_RE = re.compile(r"\[\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*\]\s*:\s*")
FILLER_RE = re.compile(
    r"\b(?:um+|uh+|umm+|uhh+|erm|mm+|hmm+|ah+|eh+)\b",
    re.IGNORECASE,
)
PUNCT_END_RE = re.compile(r'[.!?]["\')\]]?\s*$')


def normalize_spacing(text: str) -> str:
    return SPACE_RE.sub(" ", text).strip()


def normalize_paragraphs(text: str) -> str:
    """Collapse intra-paragraph whitespace but keep paragraph breaks.

    Unlike blanket normalize_spacing this preserves the line structure ASR
    backends emit, which the cleanup LM uses as pause/section hints.
    """
    paragraphs = [normalize_spacing(part) for part in re.split(r"\n\s*\n", text)]
    return "\n\n".join(part for part in paragraphs if part)


def is_parakeet_model(model_name: str) -> bool:
    return "parakeet" in model_name.lower()


def strip_parakeet_timestamps(text: str) -> str:
    return normalize_spacing(PARAKEET_TIMESTAMP_RE.sub(" ", text))


def heuristic_cleanup(raw_text: str) -> str:
    """Cheap non-LM cleanup: drop fillers, fix spacing, terminal punctuation."""
    text = FILLER_RE.sub(" ", raw_text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = normalize_spacing(text)
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    if not PUNCT_END_RE.search(text):
        text += "."
    return text


def estimate_text_tokens(text: str) -> int:
    """Crude chars/4 token estimate.

    Only used for budget gating with a safety margin; never present this as an
    exact number.
    """
    return max(1, math.ceil(len(text) / 4))


def count_words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def sanitize_filename_stem(text: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9]+", "-", text.strip()).strip("-").lower()
    return stem or "speech-note"
