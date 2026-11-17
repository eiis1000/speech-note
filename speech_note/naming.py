"""Output naming for full-auto mode."""

from __future__ import annotations

import re
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from .textproc import sanitize_filename_stem

if TYPE_CHECKING:
    from .cli import Config
    from .session import Session

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
    if not candidate.exists() and not candidate.is_symlink():
        return candidate
    for index in range(2, 1000):
        candidate = directory / f"{stem}-{index}{suffix}"
        if not candidate.exists() and not candidate.is_symlink():
            return candidate
    raise RuntimeError(f"could not choose unused output filename for {stem}{suffix}")


def unique_directory_path(directory: Path) -> Path:
    if not directory.exists() and not directory.is_symlink():
        return directory
    for index in range(2, 1000):
        candidate = directory.with_name(f"{directory.name}-{index}")
        if not candidate.exists() and not candidate.is_symlink():
            return candidate
    raise RuntimeError(f"could not choose unused directory for {directory}")


def open_unique_output(path: Path) -> tuple[Path, TextIO]:
    """Claim a unique name atomically; the caller owns the returned handle."""
    for _ in range(1000):
        candidate = unique_output_path(path.parent, path.stem, path.suffix)
        try:
            handle = candidate.open("x", encoding="utf-8")
        except FileExistsError:
            continue  # another worker claimed it between the lookup and the open
        return candidate, handle
    raise RuntimeError(f"could not claim unused output filename for {path}")


def write_unique_output(path: Path, content: str) -> Path:
    """Claim and write an auto-named output without overwriting a concurrent run."""
    candidate, handle = open_unique_output(path)
    try:
        with handle:
            handle.write(content)
    except BaseException:
        candidate.unlink(missing_ok=True)
        raise
    return candidate


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
    if config.extra_transcripts:
        # Transcript-only run: name after the first transcript, tag the rest.
        first = Path(config.extra_transcripts[0])
        stem = transcript_base_stem(first)
        unique_tags = list(dict.fromkeys(tag for tag in extra_tags if tag))
        return "-".join([stem, *unique_tags]) if unique_tags else stem
    return "speech-note"


def full_auto_output_path(config: "Config", directory: Path) -> Path:
    stem = bounded_stem(f"{sanitize_filename_stem(full_auto_source_stem(config))}-clean")
    return unique_output_path(directory, stem, ".txt")


def model_filename_tag(model: str) -> str:
    """Keep model versions, unlike source-file tags that discard numeric suffixes."""
    leaf = model.rsplit("/", 1)[-1].removesuffix(":free").removesuffix(".bin")
    if leaf.startswith("ggml-"):
        leaf = "whisper-" + leaf.removeprefix("ggml-")
    leaf = leaf.removeprefix("sherpa-onnx-nemo-")
    return sanitize_filename_stem(leaf)


def bounded_stem(stem: str) -> str:
    """Leave room for diagnostics suffixes and collision counters on common filesystems."""
    stem = sanitize_filename_stem(stem)
    if len(stem) <= 210:
        return stem
    digest = hashlib.sha256(stem.encode()).hexdigest()[:12]
    return f"{stem[:95]}-{digest}-{stem[-95:]}"


def produced_transcript_path(config: "Config", session: "Session", *,
                             model: str | None = None, rejected: bool = False) -> Path:
    """Name generated outputs from actual ASR results and the model that produced this text.

    Called after cleanup: the configured first-choice model may have failed.
    Explicit output filenames remain the caller's choice for accepted transcripts.
    """
    assert config.output is not None
    if config.output_auto_named:
        source_path = config.input_archive or config.input_file or config.replay_input_file
        base = Path(source_path).stem if source_path else full_auto_source_stem(config)
        tags = []
        backends = {f"asr{index}": source.backend for index, source in enumerate(config.asr_sources, start=1)}
        for transcript in session.asr_transcripts():
            tag = model_filename_tag(transcript.model)
            # Local Whisper often reports only the quantized checkpoint name.
            if backends.get(transcript.label) in {"whisper-cpp", "faster-whisper"} and "whisper" not in tag:
                tag = "whisper-" + tag
            tags.append(tag)
        base = "-".join([base, *dict.fromkeys(tags)])
    else:
        base = config.output.stem.removesuffix("-clean")
    cleanup = session.cleanup
    producer = model or (cleanup.checkpoint or cleanup.served_model or cleanup.method if cleanup else "unknown-model")
    suffix = "too-short-clean" if rejected else "clean"
    stem = bounded_stem(f"{base}-{model_filename_tag(producer)}-{suffix}")
    return config.output.with_name(f"{stem}.txt")
