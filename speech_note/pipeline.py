"""Pipeline orchestration.

The flow for every entry point is: gather transcript sources -> cleanup ->
commit artifacts once -> report. Computation never prints; reporting never
computes.
"""

from __future__ import annotations

import contextlib
import re
import shutil
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

from .asr import build_transcribers, ensure_loaded, prepare_sources, run_asr_collection
from .audio import normalize_audio_for_asr, probe_duration_seconds
from .config import (
    ARCHIVE_AUDIO_EXTENSIONS,
    ARCHIVE_TRANSCRIPT_EXTENSIONS,
)
from .model import Transcript
from .naming import unique_output_path
from .organizer import Organizer, build_organizer
from .session import ArtifactStore, Session
from .terminal import (
    copy_with_wl_copy,
    print_side_by_side,
    read_single_choice,
    short_label,
    status_timer,
)
from .transcribers import WhisperCppTranscriber

if TYPE_CHECKING:
    from .asr import Transcriber
    from .cli import Config
    from .config import AsrSource


# --- building blocks ---


def run_final_asr(
    config: "Config",
    session: Session,
    final_audio_path: Path,
    *,
    built: list[tuple["AsrSource", "Transcriber | None"]],
) -> None:
    """Probe, prepare models, normalize, then run the ASR collection."""
    try:
        session.input_duration_seconds = round(probe_duration_seconds(final_audio_path), 3)
    except Exception as exc:
        session.add_error(f"could not determine audio duration: {exc}")
    duration = session.input_duration_seconds

    # Make every source's model present first, on this thread, before any spinner
    # starts — so download-consent prompts are visible and serial rather than
    # buried under a status line or racing across ASR worker threads.
    prepared = prepare_sources(built)

    # Preload in-process models (whose download already happened above) while we
    # normalize, so the heavy load overlaps audio I/O. A source that failed to
    # prepare has no usable transcriber to load.
    failed_labels = {label for label, _s, _t, prep_error in prepared if prep_error is not None}
    in_process = [
        transcriber
        for label, _source, transcriber, _prep_error in prepared
        if transcriber is not None and label not in failed_labels
    ]
    preload_thread: threading.Thread | None = None
    if in_process:

        def preload() -> None:
            started = time.monotonic()
            try:
                for transcriber in in_process:
                    ensure_loaded(transcriber)
            except Exception:
                pass  # a load failure surfaces (with its real error) when the source runs
            finally:
                session.note_timing("asr_preload_seconds", time.monotonic() - started)

        preload_thread = threading.Thread(target=preload, daemon=True)
        preload_thread.start()

    asr_audio_path = final_audio_path
    started = time.monotonic()
    with status_timer("Normalizing audio"):
        with contextlib.ExitStack() as stack:
            try:
                normalized_dir = stack.enter_context(
                    tempfile.TemporaryDirectory(prefix="speech-note-final-audio-")
                )
                normalized_path = Path(normalized_dir) / "normalized-final.wav"
                normalization = normalize_audio_for_asr(
                    final_audio_path, normalized_path, sample_rate=config.sample_rate
                )
                session.note_fact("normalization", normalization.as_dict())
                asr_audio_path = normalized_path
            except Exception as exc:
                session.add_error(f"audio normalization failed; using original audio: {exc}")
            session.note_timing("normalization_seconds", time.monotonic() - started)
            if preload_thread is not None:
                preload_thread.join()
            run_asr_collection(config, session, asr_audio_path, prepared=prepared, duration=duration)

    for _source, transcriber in built:
        if isinstance(transcriber, WhisperCppTranscriber):
            summary = transcriber.backend_summary()
            if summary:
                session.note_fact("whisper_cpp_backend", summary)
            break


# --- cleanup stage ---


def cleanup_sources(session: Session) -> list[Transcript]:
    """Sources for the cleanup LM: every final ASR transcript in collection order,
    the live transcript only as a fallback when no final ASR pass succeeded, then
    external inputs."""
    sources: list[Transcript] = list(session.asr_transcripts())
    if not sources:
        live = session.transcript_by_label("live")
        if live is not None:
            sources.append(live)
    included = {id(source) for source in sources}
    sources.extend(
        t
        for t in session.transcripts
        if t.kind in {"external", "user"} and id(t) not in included
    )
    return sources


def run_cleanup_stage(config: "Config", session: Session, organizer: Organizer) -> None:
    sources = cleanup_sources(session)
    if not sources:
        return
    if config.organizer_provider != "local" and organizer.mode == "llama":
        print(
            f"note: sending transcripts to {config.organizer_provider} for cleanup; "
            "free remote endpoints may log or train on inputs",
            file=sys.stderr,
        )
    timer_label = {
        "llama": "Starting cleanup LM",
        "heuristic": "Running heuristic cleanup",
    }.get(organizer.mode)
    if timer_label is None:
        session.cleanup = organizer.cleanup(sources)
    else:
        with status_timer(timer_label) as timer:
            organizer.status_label_callback = timer.set_label
            organizer.status_note_callback = timer.note
            try:
                session.cleanup = organizer.cleanup(sources)
            finally:
                organizer.status_label_callback = None
                organizer.status_note_callback = None
    cleanup = session.cleanup
    if cleanup.seconds is not None:
        session.note_timing("cleanup_seconds", cleanup.seconds)
    if cleanup.error:
        session.add_error(f"cleanup: {cleanup.error}")
    if cleanup.warning:
        session.add_event("cleanup-warning", message=cleanup.warning)


# --- finalize: commit + report ---


def write_output_file(config: "Config", session: Session) -> None:
    cleanup = session.cleanup
    if config.output is None or cleanup is None:
        return
    if config.full_auto and (not cleanup.text or cleanup.error):
        # Full-auto must never leave a file claiming to be a clean transcript
        # when cleanup failed or was truncated.
        return
    if not cleanup.text:
        return
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(cleanup.text + "\n", encoding="utf-8")
    session.paths["output"] = str(config.output)


def commit_artifacts(config: "Config", session: Session) -> None:
    """Single commit point for all artifacts, including full-auto diagnostics."""
    write_output_file(config, session)
    error_diag_path: Path | None = None
    if config.full_auto and session.errors:
        anchor = config.output if config.output is not None else Path.cwd() / "speech-note"
        error_diag_path = unique_output_path(anchor.parent, f"{anchor.stem}-diagnostics", ".json")
        session.paths["error_diagnostics"] = str(error_diag_path)
    store = ArtifactStore(config.artifacts_dir, config.archive_dir)
    store.commit(session)
    if error_diag_path is not None:
        shutil.copyfile(Path(session.paths["latest_diagnostics"]), error_diag_path)


def review_panels(config: "Config", session: Session) -> list[tuple[str, str]]:
    panels: list[tuple[str, str]] = []
    asr = session.asr_transcripts()
    if asr:
        panels.extend((t.model, t.text) for t in asr)
    else:
        live = session.transcript_by_label("live")
        if live is not None:
            panels.append((live.model, live.text))
        else:
            # No ASR/live source (e.g. --no-asr / --extra-transcript / --input-text):
            # show the supplied transcripts so the review isn't just the cleanup with
            # nothing to compare against.
            panels.extend(
                (t.model, t.text)
                for t in session.transcripts
                if t.kind in {"external", "user"}
            )
    cleanup = session.cleanup
    if cleanup is not None and cleanup.text:
        label = "clean"
        if cleanup.method == "llama" and cleanup.served_model:
            label = f"{short_label(cleanup.served_model)} clean"
        panels.append((label, cleanup.text))
    return panels


def print_review(panels: list[tuple[str, str]]) -> None:
    # Typical cases keep the familiar layout: two ASR panels side by side, or two
    # side by side plus the cleanup below. With more sources than that, stack them.
    if len(panels) == 2:
        print_side_by_side(panels[0][0], panels[0][1], panels[1][0], panels[1][1])
    elif len(panels) == 3:
        print_side_by_side(panels[0][0], panels[0][1], panels[1][0], panels[1][1])
        print()
        print(panels[2][0])
        print()
        print(panels[2][1])
        print()
    else:
        for title, text in panels:
            print(title)
            print()
            print(text)
            print()


def choose_and_copy(session: Session, panels: list[tuple[str, str]]) -> None:
    valid = {"n"} | {str(index) for index in range(1, len(panels) + 1)}
    # Only prompt when a human is at the terminal; piped/redirected runs copy
    # nothing (there is no longer a --copy-choice flag to script the answer).
    if not (sys.stdin and sys.stdin.isatty()):
        session.copy_choice = "n"
        session.clipboard_status = "skipped"
        return
    options = " ".join(
        f"[{index}] {short_label(title)}" for index, (title, _text) in enumerate(panels, start=1)
    )
    choice = read_single_choice(valid, f"Copy which? {options} [n] none", default_key="n")
    session.copy_choice = choice
    if choice == "n":
        session.clipboard_status = "skipped"
    else:
        session.clipboard_status = copy_with_wl_copy(panels[int(choice) - 1][1])
    print(f"clipboard: {session.clipboard_status}", file=sys.stderr)


def report(config: "Config", session: Session) -> None:
    err = sys.stderr
    if not session.produced_output:
        print("no speech captured", file=err)
    if config.full_auto:
        if "output" in session.paths:
            print(f"saved clean: {session.paths['output']}", file=err)
        else:
            print("no clean output", file=err)
        # A flagged-but-kept cleanup (short / cut off mid-sentence) is still written; the
        # warning must surface here or full-auto returns before the warning print below.
        cleanup = session.cleanup
        if cleanup is not None and cleanup.warning:
            print(f"warning: {cleanup.warning}", file=err)
        if "error_diagnostics" in session.paths:
            print(f"error diagnostics: {session.paths['error_diagnostics']}", file=err)
        return
    print(f"saved raw: {session.paths.get('latest_raw')} and {session.paths.get('archive_raw')}", file=err)
    print(
        f"saved clean: {session.paths.get('latest_clean')} and {session.paths.get('archive_clean')}",
        file=err,
    )
    if "output" in session.paths:
        print(f"saved output: {session.paths['output']}", file=err)
    print(
        f"saved diagnostics: {session.paths.get('latest_diagnostics')} "
        f"and {session.paths.get('archive_diagnostics')}",
        file=err,
    )
    level_warning = session.audio_level_warning()
    if level_warning is not None:
        print(level_warning, file=err)
    cleanup = session.cleanup
    if cleanup is not None and cleanup.warning:
        print(f"warning: {cleanup.warning}", file=err)
    if session.skips:
        for message in session.skips[-4:]:
            print(f"note: {message}", file=err)
    if session.errors:
        print("errors:", file=err)
        for message in session.errors[-8:]:
            print(f"- {message}", file=err)
    if not session.produced_output:
        return
    panels = review_panels(config, session)
    print_review(panels)
    if panels:
        started = time.monotonic()
        choose_and_copy(session, panels)
        session.note_timing("copy_choice_wait_seconds", time.monotonic() - started)


def finalize(config: "Config", session: Session, organizer: Organizer) -> None:
    run_cleanup_stage(config, session, organizer)
    if config.full_auto:
        session.copy_choice = "n"
        session.clipboard_status = "skipped"
    commit_artifacts(config, session)
    report(config, session)
    cleanup = session.cleanup
    session.run_failed = not session.produced_output or (
        config.full_auto
        and config.organizer_mode == "llama"
        and (cleanup is None or not cleanup.text or bool(cleanup.error))
    )


# --- transcript file inputs ---


_SRT_INDEX_RE = re.compile(r"^\d+$")
_CUE_TIME_RE = re.compile(r"-->")


def extract_subtitle_text(raw: str) -> str:
    """Pull spoken text out of SRT/VTT content, dropping cues and timestamps."""
    lines: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped == "WEBVTT":
            continue
        if _SRT_INDEX_RE.match(stripped) or _CUE_TIME_RE.search(stripped):
            continue
        lines.append(stripped)
    return "\n".join(lines)


def read_extra_transcript(path: Path) -> Transcript | None:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise SystemExit(f"failed to read extra transcript {path}: {exc}") from exc
    suffix = path.suffix.lower()
    if suffix in {".srt", ".vtt"}:
        text = extract_subtitle_text(raw)
    elif suffix == ".json":
        # No universal transcript-JSON schema; pass the raw content through and
        # say so, instead of pretending it was parsed.
        text = raw.strip()
    else:
        text = raw.strip()
    if not text:
        return None
    return Transcript(
        label=f"extra:{path.name}",
        model=path.name,
        kind="external",
        text=text,
        quality_hint="externally provided transcript",
    )


def load_extra_transcripts(config: "Config", session: Session) -> None:
    for path in config.extra_transcripts:
        transcript = read_extra_transcript(Path(path))
        if transcript is not None:
            session.add_transcript(transcript)


# --- entry points ---


def run_file_pipeline(config: "Config") -> Session:
    assert config.input_file is not None
    session = Session(config)
    organizer, supervisor = build_organizer(config)
    load_extra_transcripts(config, session)
    # --no-asr: don't transcribe the audio, just clean up the provided transcript(s).
    built = [] if config.no_asr else build_transcribers(config)
    prewarm = start_organizer_prewarm(config, organizer)
    try:
        if not config.no_asr:
            run_final_asr(config, session, config.input_file, built=built)
        finalize(config, session, organizer)
    finally:
        if prewarm is not None:
            prewarm.join(timeout=0.1)
        if supervisor is not None:
            supervisor.close()
    return session


def start_organizer_prewarm(config: "Config", organizer: Organizer) -> threading.Thread | None:
    if config.organizer_mode != "llama" or organizer.supervisor is None:
        return None
    if not config.organizer_prewarm:
        # Sequencing escape hatch (--organizer-no-prewarm): don't load the cleanup
        # LM concurrently with ASR; it starts lazily at the cleanup stage instead,
        # after ASR has released the (possibly shared) GPU.
        return None
    print("Loading cleanup LM in background", file=sys.stderr, flush=True)

    def prewarm() -> None:
        with contextlib.suppress(Exception):
            organizer.supervisor.ensure_running()

    thread = threading.Thread(target=prewarm, daemon=True)
    thread.start()
    return thread


def discover_archive_inputs(extract_dir: Path) -> tuple[Path, list[Path]]:
    files = [path for path in extract_dir.rglob("*") if path.is_file()]
    audio_files = sorted(
        (path for path in files if path.suffix.lower() in ARCHIVE_AUDIO_EXTENSIONS),
        key=lambda path: (-path.stat().st_size, str(path)),
    )
    if not audio_files:
        raise SystemExit("archive contains no supported audio file")
    if len(audio_files) > 1:
        names = ", ".join(str(path.relative_to(extract_dir)) for path in audio_files[:8])
        raise SystemExit(f"archive contains multiple audio files; pass one directly instead: {names}")
    audio_path = audio_files[0]
    transcript_paths = sorted(
        path
        for path in files
        if path != audio_path and path.suffix.lower() in ARCHIVE_TRANSCRIPT_EXTENSIONS
    )
    return audio_path, transcript_paths


def extract_zip_safely(archive_path: Path, extract_dir: Path) -> None:
    extract_root = extract_dir.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target_path = (extract_dir / member.filename).resolve()
            if target_path != extract_root and extract_root not in target_path.parents:
                raise SystemExit(f"archive member escapes extraction directory: {member.filename}")
        archive.extractall(extract_dir)


def run_archive_pipeline(config: "Config") -> Session:
    assert config.input_archive is not None
    with tempfile.TemporaryDirectory(prefix="speech-note-archive-") as tmp_dir:
        extract_dir = Path(tmp_dir)
        extract_zip_safely(config.input_archive, extract_dir)
        audio_path, transcript_paths = discover_archive_inputs(extract_dir)
        # Derive a per-run config rather than mutating the one we were given.
        run_config = config.with_archive_contents(audio_path, transcript_paths)
        return run_file_pipeline(run_config)


def run_transcript_pipeline(config: "Config") -> Session:
    """Transcript-only run: no audio, just clean up the provided --extra-transcript
    file(s) as equal peers (the cleanup LM reconciles them)."""
    assert config.extra_transcripts
    session = Session(config)
    organizer, supervisor = build_organizer(config)
    load_extra_transcripts(config, session)
    try:
        finalize(config, session, organizer)
    finally:
        if supervisor is not None:
            supervisor.close()
    return session


def run_dry_text_pipeline(config: "Config") -> Session:
    assert config.dry_run_text is not None
    session = Session(config)
    organizer, supervisor = build_organizer(config)
    chunks = [chunk.strip() for chunk in config.dry_run_text.split("||") if chunk.strip()]
    if chunks:
        session.add_transcript(
            Transcript(
                label="primary",
                model="dry-run-text",
                kind="user",
                text="\n".join(chunks),
            )
        )
    load_extra_transcripts(config, session)
    try:
        finalize(config, session, organizer)
    finally:
        if supervisor is not None:
            supervisor.close()
    return session
