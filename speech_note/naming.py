"""Output naming for full-auto mode."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from .textproc import sanitize_filename_stem

if TYPE_CHECKING:
    from .cli import Config

SOURCE_WORDS = {
    "whisper",
    "parakeet",
    "nemo",
    "canary",
    "google",
    "recorder",
    "primary",
    "secondary",
    "raw",
    "transcript",
    "xscript",
    "xn",
}


def transcript_source_tag(path: Path, fallback: str) -> str:
    """Short tag describing a transcript file for output naming."""
    parts = [part for part in re.split(r"[^A-Za-z0-9]+", path.stem.lower()) if part]
    if not parts:
        return fallback
    for part in reversed(parts):
        if part in SOURCE_WORDS:
            return part
    return parts[-1]


def transcript_base_stem(path: Path) -> str:
    parts = [part for part in re.split(r"[^A-Za-z0-9]+", path.stem) if part]
    if len(parts) > 1 and transcript_source_tag(path, "") == parts[-1].lower():
        return "-".join(parts[:-1])
    return path.stem


def unique_output_path(directory: Path, stem: str, suffix: str) -> Path:
    candidate = directory / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate
    for index in range(2, 1000):
        candidate = directory / f"{stem}-{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not choose unused output filename for {stem}{suffix}")


def full_auto_source_stem(config: "Config") -> str:
    """Stem describing the input and its transcript sources."""
    extra_tags = [
        transcript_source_tag(Path(path), f"extra{index}")
        for index, path in enumerate(config.extra_transcripts, start=1)
    ]
    if config.input_archive is not None:
        stem = Path(config.input_archive).stem
        tags: list[str] = []
        for source in config.asr_sources:
            if source.backend in {"whisper-cpp", "faster-whisper"}:
                tags.append("whisper")
            else:
                tags.append(transcript_source_tag(Path(f"recording.{source.model}.txt"), source.backend))
        tags.extend(extra_tags)
        unique_tags = list(dict.fromkeys(tag for tag in tags if tag))
        return "-".join([stem, *unique_tags]) if unique_tags else stem
    for path in (config.input_file, config.replay_input_file):
        if path is not None:
            return Path(path).stem
    if config.primary_transcript is not None:
        primary_path = Path(config.primary_transcript)
        stem = transcript_base_stem(primary_path)
        tags = [transcript_source_tag(primary_path, "primary")]
        if config.secondary_transcript is not None:
            tags.append(transcript_source_tag(Path(config.secondary_transcript), "secondary"))
        tags.extend(extra_tags)
        unique_tags = list(dict.fromkeys(tag for tag in tags if tag))
        return "-".join([stem, *unique_tags]) if unique_tags else stem
    return "speech-note"


def full_auto_output_path(config: "Config", directory: Path) -> Path:
    stem = f"{sanitize_filename_stem(full_auto_source_stem(config))}-clean"
    return unique_output_path(directory, stem, ".txt")
