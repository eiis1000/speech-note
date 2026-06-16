"""Pipeline orchestration.

The flow for every entry point is: gather transcript sources -> cleanup ->
commit artifacts once -> report. Computation never prints; reporting never
computes.
"""

from __future__ import annotations

import concurrent.futures
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

from .audio import normalize_audio_for_asr, probe_duration_seconds
from .config import (
    ARCHIVE_AUDIO_EXTENSIONS,
    ARCHIVE_TRANSCRIPT_EXTENSIONS,
    SUBPROCESS_SECONDARY_BACKENDS,
    primary_model_presentation,
)
from .model import AsrOutcome, Transcript
from .naming import unique_output_path
from .organizer import Organizer, build_organizer
from .secondary import (
    build_local_secondary_transcriber,
    run_secondary,
)
from .session import ArtifactStore, Session
from .terminal import (
    copy_with_wl_copy,
    print_side_by_side,
    read_single_choice,
    short_label,
    status_timer,
)
from .transcribers import (
    FasterWhisperTranscriber,
    PrimaryTranscriber,
    WhisperCppTranscriber,
)

if TYPE_CHECKING:
    from .cli import Config
    from .transcribers import LocalSecondaryTranscriber


# --- building blocks ---


def build_primary_transcriber(config: "Config", *, live: bool = False) -> PrimaryTranscriber:
    if live:
        # Live preview always uses a small resident faster-whisper model: a
        # per-segment whisper.cpp subprocess would reload the model and
        # re-initialize the GPU for every utterance.
        from .config import LIVE_ASR_MODEL

        return FasterWhisperTranscriber(
            model_name=LIVE_ASR_MODEL,
            device="cpu",
            compute_type="int8",
            cpu_threads=config.live_asr_cpu_threads,
            download_root=config.download_root,
        )
    if config.asr_backend == "whisper-cpp":
        return WhisperCppTranscriber(
            model_name=config.asr_model,
            model_path=config.whisper_cpp_model,
            binary=config.whisper_cpp_binary,
            cpu_threads=config.asr_cpu_threads,
            device=config.whisper_cpp_device,
            use_gpu=config.whisper_cpp_gpu,
            auto_download=config.auto_download,
        )
    return FasterWhisperTranscriber(
        model_name=config.asr_model,
        device=config.asr_device,
        compute_type=config.asr_compute_type,
        cpu_threads=config.asr_cpu_threads,
        download_root=config.download_root,
    )


def primary_device_kind(config: "Config") -> str:
    if config.asr_backend == "whisper-cpp":
        return "gpu" if config.whisper_cpp_gpu else "cpu"
    return "gpu" if config.asr_device.lower().startswith("cuda") else "cpu"


def secondary_device_kind(config: "Config") -> str:
    backend = config.secondary_asr_backend
    device = config.secondary_asr_device.lower()
    if backend in {"onnx", "pocketsphinx"}:
        return "cpu"
    if device == "cpu":
        return "cpu"
    return "gpu"


def should_parallelize_asr(config: "Config") -> bool:
    """Run primary and secondary concurrently when they use different devices.

    --parallel-final-asr / --no-parallel-final-asr overrides; the default is
    automatic because e.g. GPU whisper.cpp and CPU ONNX Parakeet do not contend.
    """
    if config.parallel_final_asr is not None:
        return config.parallel_final_asr
    if not config.secondary_asr_enabled:
        return False
    return primary_device_kind(config) != secondary_device_kind(config)


def run_primary_pass(
    config: "Config",
    transcriber: PrimaryTranscriber,
    audio_path: Path,
    *,
    duration_seconds: float | None,
) -> AsrOutcome:
    outcome = AsrOutcome(name="primary")
    started = time.monotonic()
    try:
        text = transcriber.transcribe_file(audio_path, config.language)
    except Exception as exc:
        outcome.error = str(exc)
        outcome.seconds = time.monotonic() - started
        return outcome
    outcome.seconds = time.monotonic() - started
    if duration_seconds and outcome.seconds:
        outcome.realtime_factor = round(duration_seconds / outcome.seconds, 2)
    if not text:
        outcome.error = "primary ASR produced no text"
        return outcome
    effective = (
        transcriber.effective_model
        if isinstance(transcriber, WhisperCppTranscriber)
        else config.asr_model
    )
    display_model, quality_hint = primary_model_presentation(config.asr_model, effective)
    outcome.transcript = Transcript(
        label="primary",
        model=display_model,
        kind="asr-final",
        text=text,
        quality_hint=quality_hint,
    )
    return outcome


def run_final_asr(
    config: "Config",
    session: Session,
    final_audio_path: Path,
    *,
    primary_transcriber: PrimaryTranscriber | None,
    secondary_transcriber: "LocalSecondaryTranscriber | None",
) -> None:
    """Probe, normalize, then run the enabled ASR passes."""
    try:
        session.input_duration_seconds = round(probe_duration_seconds(final_audio_path), 3)
    except Exception as exc:
        session.add_error(f"could not determine audio duration: {exc}")
    duration = session.input_duration_seconds

    run_secondary_pass = config.secondary_asr_enabled

    preload_thread: threading.Thread | None = None
    if run_secondary_pass and secondary_transcriber is not None:
        preload_errors: list[str] = []

        def preload() -> None:
            started = time.monotonic()
            try:
                secondary_transcriber.ensure_loaded()
            except Exception as exc:
                preload_errors.append(str(exc))
            finally:
                session.note_timing("secondary_preload_seconds", time.monotonic() - started)

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

            def primary_job() -> AsrOutcome:
                assert primary_transcriber is not None
                return run_primary_pass(
                    config, primary_transcriber, asr_audio_path, duration_seconds=duration
                )

            def secondary_job() -> AsrOutcome:
                return run_secondary(
                    config,
                    asr_audio_path,
                    duration_seconds=duration,
                    transcriber=secondary_transcriber,
                )

            jobs: list[tuple[str, object]] = []
            if primary_transcriber is not None:
                jobs.append((config.asr_model, primary_job))
            if run_secondary_pass:
                jobs.append((config.secondary_asr_model, secondary_job))
            if not jobs:
                return

            if len(jobs) > 1 and should_parallelize_asr(config):
                labels = " + ".join(short_label(name) for name, _job in jobs)
                with status_timer(f"Running ASR passes in parallel: {labels}"):
                    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as executor:
                        futures = [executor.submit(job) for _name, job in jobs]  # type: ignore[arg-type]
                        for future in concurrent.futures.as_completed(futures):
                            session.record_asr_outcome(future.result())
            else:
                for index, (name, job) in enumerate(jobs, start=1):
                    with status_timer(f"Running ASR pass {index}/{len(jobs)}: {short_label(name)}"):
                        session.record_asr_outcome(job())  # type: ignore[operator]

    if isinstance(primary_transcriber, WhisperCppTranscriber):
        summary = primary_transcriber.backend_summary()
        if summary:
            session.note_fact("whisper_cpp_backend", summary)


# --- cleanup stage ---


def cleanup_sources(session: Session) -> list[Transcript]:
    """Sources for the cleanup LM: final ASR results, the live transcript only
    as a fallback when the primary final pass failed, then external inputs."""
    sources: list[Transcript] = []
    primary = session.transcript_by_label("primary")
    live = session.transcript_by_label("live")
    if primary is not None:
        sources.append(primary)
    elif live is not None:
        sources.append(live)
    secondary = session.transcript_by_label("secondary")
    if secondary is not None:
        sources.append(secondary)
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
    primary = session.transcript_by_label("primary") or session.transcript_by_label("live")
    if primary is not None:
        panels.append((primary.model, primary.text))
    secondary = session.transcript_by_label("secondary")
    if secondary is not None:
        panels.append((secondary.model, secondary.text))
    cleanup = session.cleanup
    if cleanup is not None and cleanup.text:
        label = "clean"
        if cleanup.method == "llama" and cleanup.served_model:
            label = f"{short_label(cleanup.served_model)} clean"
        panels.append((label, cleanup.text))
    return panels


def print_review(panels: list[tuple[str, str]]) -> None:
    if len(panels) >= 3:
        print_side_by_side(panels[0][0], panels[0][1], panels[1][0], panels[1][1])
        print()
        print(panels[2][0])
        print()
        print(panels[2][1])
        print()
    elif len(panels) == 2:
        print_side_by_side(panels[0][0], panels[0][1], panels[1][0], panels[1][1])


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


def read_transcript_file(path: Path, *, label: str) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError as exc:
        raise SystemExit(f"failed to read {label} transcript {path}: {exc}") from exc
    if not text:
        raise SystemExit(f"{label} transcript is empty: {path}")
    return text


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
    primary_transcriber = build_primary_transcriber(config)
    secondary_transcriber = maybe_local_secondary(config)
    prewarm = start_organizer_prewarm(config, organizer)
    try:
        run_final_asr(
            config,
            session,
            config.input_file,
            primary_transcriber=primary_transcriber,
            secondary_transcriber=secondary_transcriber,
        )
        finalize(config, session, organizer)
    finally:
        if prewarm is not None:
            prewarm.join(timeout=0.1)
        if supervisor is not None:
            supervisor.close()
    return session


def maybe_local_secondary(config: "Config") -> "LocalSecondaryTranscriber | None":
    if not config.secondary_asr_enabled:
        return None
    if config.secondary_asr_backend in SUBPROCESS_SECONDARY_BACKENDS:
        return None
    return build_local_secondary_transcriber(config)


def start_organizer_prewarm(config: "Config", organizer: Organizer) -> threading.Thread | None:
    if config.organizer_mode != "llama" or organizer.supervisor is None:
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
    assert config.primary_transcript is not None
    session = Session(config)
    organizer, supervisor = build_organizer(config)
    session.add_transcript(
        Transcript(
            label="primary",
            model=config.primary_transcript.name,
            kind="external",
            text=read_transcript_file(config.primary_transcript, label="primary"),
        )
    )
    if config.secondary_transcript is not None:
        session.add_transcript(
            Transcript(
                label="secondary",
                model=config.secondary_transcript.name,
                kind="external",
                text=read_transcript_file(config.secondary_transcript, label="secondary"),
            )
        )
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


def run_secondary_only(config: "Config") -> Session:
    """Run only the secondary backend on a file and print its transcript."""
    assert config.input_file is not None
    session = Session(config)
    try:
        session.input_duration_seconds = round(probe_duration_seconds(config.input_file), 3)
    except Exception as exc:
        session.add_error(f"could not determine audio duration: {exc}")
    with tempfile.TemporaryDirectory(prefix="speech-note-secondary-only-") as tmp_dir:
        audio_path = config.input_file
        try:
            normalized = Path(tmp_dir) / "normalized.wav"
            normalize_audio_for_asr(config.input_file, normalized, sample_rate=config.sample_rate)
            audio_path = normalized
        except Exception as exc:
            session.add_error(f"audio normalization failed; using original audio: {exc}")
        outcome = run_secondary(
            config, audio_path, duration_seconds=session.input_duration_seconds
        )
    session.record_asr_outcome(outcome)
    if outcome.transcript is not None:
        print(outcome.transcript.text)
    elif outcome.skip_reason:
        print(outcome.skip_reason, file=sys.stderr)
    else:
        print(f"secondary ASR failed: {outcome.error}", file=sys.stderr)
    session.run_failed = outcome.error is not None
    return session
