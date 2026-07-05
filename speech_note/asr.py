"""ASR execution: build and run the collection of ASR sources.

There is no primary/secondary distinction. A run transcribes the audio with an
ordered collection of AsrSource entries and produces one Transcript per source,
in list order, all handed to the cleanup LM as peers. Sources on different
devices (e.g. GPU Whisper + CPU Parakeet) run concurrently; same-device sources
run sequentially within their device group so they don't contend. Each backend
is built lazily and downloads its model (consent-gated) only when it actually runs.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

from .config import (
    ASR_BACKENDS,
    CTC_CHUNK_LENGTH_SECONDS,
    DEFAULT_OPENROUTER_API_BASE,
    DEFAULT_OPENROUTER_ASR_MAX_OUTPUT_TOKENS,
    LIVE_ASR_MODEL,
    OPENROUTER_API_KEY_ENV,
    CTC_STRIDE_SECONDS,
    OPENROUTER_ASR_MIN_TIMEOUT,
    OPENROUTER_ASR_MP3_SAMPLE_RATE,
    AsrSource,
    asr_source_presentation,
    short_source_name,
)
from .model import AsrOutcome, Transcript
from .terminal import debug_log, status_phase
from .textproc import is_parakeet_model, normalize_spacing, strip_parakeet_timestamps
from .transcribers import (
    CTCTranscriber,
    FasterWhisperTranscriber,
    OpenRouterTranscriber,
    PocketSphinxTranscriber,
    SherpaTranscriber,
    Transcriber,
    WhisperCppTranscriber,
    thread_env,
)

if TYPE_CHECKING:
    from .cli import Config
    from .session import Session


def ctc_chunk_config() -> tuple[float, float]:
    """(chunk_length_seconds, stride_seconds) for the CTC backend's long-form
    striding: it transcribes any length in overlapping chunk_length windows that
    merge at the logit level, staying under the model's ~400s position cliff."""
    return CTC_CHUNK_LENGTH_SECONDS, CTC_STRIDE_SECONDS


def crispasr_gpu_backend(device: str) -> str:
    lowered = device.lower()
    if lowered == "cpu":
        return "cpu"
    if lowered in {"vulkan", "gpu", "auto"}:
        return "vulkan"
    return lowered


def offline_env(*, auto_download: bool, cpu_threads: int) -> dict[str, str]:
    env = thread_env(cpu_threads)
    cache_base = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    env.setdefault("PIP_CACHE_DIR", str(cache_base / "speech-note" / "pip"))
    if not auto_download:
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
    return env


def build_live_transcriber(config: "Config") -> FasterWhisperTranscriber:
    """The live-preview model: a small resident faster-whisper, separate from the
    final ASR collection (a per-segment whisper.cpp subprocess would reload the
    model and re-init the GPU for every utterance).

    auto_download is forced on: this tiny model is intrinsic to interactive
    capture and loads in a background thread mid-recording, where a consent
    prompt could neither be shown cleanly nor answered. The final ASR sources
    (including any faster-whisper source) are gated normally."""
    return FasterWhisperTranscriber(
        model_name=LIVE_ASR_MODEL,
        device="cpu",
        compute_type="int8",
        cpu_threads=config.live_asr_cpu_threads,
        download_root=config.download_root,
        auto_download=True,
    )


def build_transcriber(config: "Config", source: AsrSource) -> Transcriber | None:
    """In-process transcriber for a source, or None if the backend is a subprocess CLI."""
    if not ASR_BACKENDS[source.backend].in_process:
        return None
    backend = source.backend
    if backend == "whisper-cpp":
        device_index = int(source.device) if source.device.isdigit() else 0
        return WhisperCppTranscriber(
            model_name=source.model,
            model_path=config.whisper_cpp_model,
            binary=config.whisper_cpp_binary,
            cpu_threads=config.asr_cpu_threads,
            device=device_index,
            use_gpu=source.device_kind == "gpu",
            auto_download=config.auto_download,
        )
    if backend == "faster-whisper":
        return FasterWhisperTranscriber(
            model_name=source.model,
            device=source.device,
            compute_type=config.asr_compute_type,
            cpu_threads=config.asr_cpu_threads,
            download_root=config.download_root,
        )
    if backend == "sherpa":
        return SherpaTranscriber(
            model_name=source.model,
            device=source.device,
            download_root=config.download_root,
            num_threads=config.asr_cpu_threads,
            auto_download=config.auto_download,
        )
    if backend == "ctc":
        chunk_length_seconds, stride_seconds = ctc_chunk_config()
        return CTCTranscriber(
            model_name=source.model,
            device=source.device,
            download_root=config.download_root,
            chunk_length_seconds=chunk_length_seconds,
            stride_seconds=stride_seconds,
            auto_download=config.auto_download,
        )
    if backend == "pocketsphinx":
        return PocketSphinxTranscriber(model_name=source.model, sample_rate=config.sample_rate)
    if backend == "openrouter":
        return OpenRouterTranscriber(
            model_name=source.model,
            api_base=DEFAULT_OPENROUTER_API_BASE,
            auth_env=OPENROUTER_API_KEY_ENV,
            timeout=OPENROUTER_ASR_MIN_TIMEOUT,
            mp3_sample_rate=OPENROUTER_ASR_MP3_SAMPLE_RATE,
            max_output_tokens=DEFAULT_OPENROUTER_ASR_MAX_OUTPUT_TOKENS,
        )
    raise ValueError(f"backend {backend!r} has no in-process transcriber")


def build_transcribers(config: "Config") -> list[tuple[AsrSource, Transcriber | None]]:
    """Build a transcriber for every source (None for subprocess backends).

    Construction is cheap and never touches the network for any backend: every
    transcriber loads (and downloads, consent-gated) lazily. The model is fetched
    by prepare_sources / when the source runs, so building sources that may not
    run (or may be skipped) is free.
    """
    return [(source, build_transcriber(config, source)) for source in config.asr_sources]


def build_subprocess_command(config: "Config", source: AsrSource, audio_path: Path) -> list[str]:
    backend = source.backend
    if backend == "onnx":
        return ["onnx-asr", "--vad", "silero", "-q", "int8", source.model, str(audio_path)]
    if backend == "crispasr":
        command = [
            "crispasr",
            "--backend", "parakeet",
            "--gpu-backend", crispasr_gpu_backend(source.device),
            "-t", str(config.asr_cpu_threads),
            "-m", source.model,
            "-f", str(audio_path),
            "-nt",
            "--no-prints",
        ]
        if config.auto_download:
            command.append("--auto-download")
        return command
    raise ValueError(f"backend {backend!r} does not run as a subprocess")


def _run_subprocess_backend(config: "Config", source: AsrSource, audio_path: Path) -> str:
    """Run an external backend (onnx-asr, crispasr). These CLIs print only the
    transcript on stdout by contract."""
    env = offline_env(auto_download=config.auto_download, cpu_threads=config.asr_cpu_threads)
    command = build_subprocess_command(config, source, audio_path)
    debug_log(f"asr subprocess start command={command!r}")
    result = subprocess.run(command, check=False, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        )
    return result.stdout


def run_source(
    config: "Config",
    source: AsrSource,
    audio_path: Path,
    *,
    label: str,
    duration: float | None,
    transcriber: Transcriber | None,
) -> AsrOutcome:
    """Run one ASR source; never raises."""
    outcome = AsrOutcome(name=label)
    started = time.monotonic()
    try:
        if not ASR_BACKENDS[source.backend].in_process:
            text = _run_subprocess_backend(config, source, audio_path)
            effective = source.model
        else:
            if transcriber is None:
                transcriber = build_transcriber(config, source)
            assert transcriber is not None
            text = transcriber.transcribe_file(
                audio_path, config.language, duration_seconds=duration
            )
            effective = (
                transcriber.effective_model
                if isinstance(transcriber, WhisperCppTranscriber)
                else source.model
            )
    except Exception as exc:
        outcome.error = str(exc)
        outcome.seconds = time.monotonic() - started
        return outcome
    outcome.seconds = time.monotonic() - started
    if duration and outcome.seconds:
        outcome.realtime_factor = round(duration / outcome.seconds, 2)
    text = normalize_spacing(text)
    if config.strip_asr_timestamps and is_parakeet_model(source.model):
        text = strip_parakeet_timestamps(text)
    if text:
        display, hint = asr_source_presentation(source.model, effective)
        outcome.transcript = Transcript(
            label=label, model=display, kind="asr-final", text=text, quality_hint=hint
        )
    else:
        # The backend ran fine but found nothing (e.g. its VAD saw no speech) — a
        # skip, not a failure.
        outcome.skip_reason = f"{label} ASR skipped: backend produced no text (no speech detected?)"
    return outcome


@dataclasses.dataclass(frozen=True)
class PreparedSource:
    """One source ready to run: a stable label, its transcriber (None for
    subprocess backends), and the prep error if its model could not be made present."""

    label: str
    source: AsrSource
    transcriber: Transcriber | None
    error: str | None = None


def prepare_sources(
    built: list[tuple[AsrSource, Transcriber | None]],
) -> list[PreparedSource]:
    """Make every source's model present, up front and on the calling thread.

    This is where model-download consent happens: on the main thread, serially,
    with no status spinner running and no worker threads contending for stdin —
    so the "[y/N]" prompt is actually visible and a single answer applies to one
    model at a time. A backend whose model is already cached prompts nothing; a
    subprocess backend (transcriber None) is left for its own run to fetch.

    A declined or failed download is captured as a per-source prep error and the
    source is *not* retried later under a spinner — it simply runs as a failure.
    Labels (asr1, asr2, …) are assigned here, once, over the full collection.
    """
    prepared: list[PreparedSource] = []
    for index, (source, transcriber) in enumerate(built, start=1):
        prep_error: str | None = None
        if transcriber is not None:
            try:
                transcriber.ensure_downloaded()
            except Exception as exc:  # noqa: BLE001 — record and skip, don't abort the run
                prep_error = str(exc)
        prepared.append(PreparedSource(f"asr{index}", source, transcriber, prep_error))
    return prepared


def run_asr_collection(
    config: "Config",
    session: "Session",
    audio_path: Path,
    *,
    prepared: list[PreparedSource],
    duration: float | None,
) -> None:
    """Run every prepared source and record its outcome, in collection order.

    Sources are grouped by device kind: each group runs sequentially (same-device
    sources contend), the groups run concurrently (different devices overlap). So a
    GPU Whisper source and a CPU Parakeet source finish in max(GPU, CPU) wall-clock.
    Outcomes are recorded in the original collection order regardless of which
    finishes first, so list order is preserved for the raw/fallback transcript.
    """
    if not prepared:
        return

    groups: dict[str, list[PreparedSource]] = {}
    for job in prepared:
        groups.setdefault(job.source.device_kind, []).append(job)
    display_names = _unique_display_names(prepared)

    results: dict[str, AsrOutcome] = {}
    # One "ASR" phase; every source — across both device groups — is a task on the
    # single status line, showing live elapsed and flipping to ✓/✗ as it finishes.
    with status_phase("ASR") as display:

        def run_group(group: list[PreparedSource]) -> list[tuple[str, AsrOutcome]]:
            out: list[tuple[str, AsrOutcome]] = []
            for job in group:
                name = display_names[job.label]
                if job.error is not None:
                    # Model could not be made present (declined/failed in prepare_sources);
                    # show it as a failed task and don't re-attempt or re-prompt.
                    display.start_task(name)
                    display.finish_task(name, error=True)
                    out.append((job.label, AsrOutcome(name=job.label, error=job.error)))
                    continue
                display.start_task(name)
                outcome = run_source(
                    config, job.source, audio_path,
                    label=job.label, duration=duration, transcriber=job.transcriber,
                )
                # run_source never raises — derive the task's ✓/✗ from the outcome.
                display.finish_task(name, error=not outcome.ok)
                out.append((job.label, outcome))
            return out

        if len(groups) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(groups)) as executor:
                futures = [executor.submit(run_group, group) for group in groups.values()]
                # Collect in the main thread (worker threads never touch `results`).
                for future in concurrent.futures.as_completed(futures):
                    results.update(future.result())
        else:
            results.update(run_group(next(iter(groups.values()))))

    for job in prepared:
        if job.label in results:
            session.record_asr_outcome(results[job.label])


def _unique_display_names(prepared: list[PreparedSource]) -> dict[str, str]:
    """Status-line name per label; duplicates get a counter ("whisper 2") so two
    sources of the same backend don't share one task slot on the status line."""
    names: dict[str, str] = {}
    seen: dict[str, int] = {}
    for job in prepared:
        name = short_source_name(job.source)
        seen[name] = seen.get(name, 0) + 1
        names[job.label] = name if seen[name] == 1 else f"{name} {seen[name]}"
    return names
