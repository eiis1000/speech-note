"""Secondary ASR execution.

Subprocess backends (onnx-asr, crispasr) and in-process backends (sherpa, CTC,
PocketSphinx) share one entry point, run_secondary, which returns a typed
AsrOutcome — a transcript, a skip reason, or an error, never a string protocol.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

from .config import (
    CTC_CHUNK_LENGTH_SECONDS,
    SECONDARY_OVERLAP_SECONDS,
    SUBPROCESS_SECONDARY_BACKENDS,
)
from .model import AsrOutcome, Transcript
from .terminal import debug_log
from .textproc import is_parakeet_model, normalize_spacing, strip_parakeet_timestamps
from .transcribers import CTCTranscriber, PocketSphinxTranscriber, SherpaTranscriber, thread_env

if TYPE_CHECKING:
    from .cli import Config
    from .transcribers import LocalSecondaryTranscriber


def ctc_chunk_config() -> tuple[float, float]:
    """(chunk_length_seconds, stride_seconds) for the CTC backend's long-form
    striding. CTC transcribes any length in chunk_length windows that overlap by
    the stride and merge at the logit level, so there is no length limit; the
    window stays under the model's ~400s position cliff."""
    return CTC_CHUNK_LENGTH_SECONDS, SECONDARY_OVERLAP_SECONDS


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


def build_subprocess_command(config: "Config", audio_path: Path) -> list[str]:
    backend = config.secondary_asr_backend
    if backend == "onnx":
        return [
            "onnx-asr",
            "--vad", "silero",
            "-q", "int8",
            config.secondary_asr_model,
            str(audio_path),
        ]
    if backend == "crispasr":
        command = [
            "crispasr",
            "--backend", "parakeet",
            "--gpu-backend", crispasr_gpu_backend(config.secondary_asr_device),
            "-t", str(config.asr_cpu_threads),
            "-m", config.secondary_asr_model,
            "-f", str(audio_path),
            "-nt",
            "--no-prints",
        ]
        if config.auto_download:
            command.append("--auto-download")
        return command
    raise ValueError(f"backend {backend!r} does not run as a subprocess")


def _run_subprocess_backend(config: "Config", audio_path: Path) -> str:
    """Run an external backend (onnx-asr, crispasr) and return its transcript.

    These CLIs print only the transcript on stdout by contract.
    """
    env = offline_env(auto_download=config.auto_download, cpu_threads=config.asr_cpu_threads)
    command = build_subprocess_command(config, audio_path)
    debug_log(f"secondary subprocess start command={command!r}")
    result = subprocess.run(command, check=False, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        )
    return result.stdout


def run_secondary(
    config: "Config",
    audio_path: Path,
    *,
    duration_seconds: float | None,
    transcriber: "LocalSecondaryTranscriber | None" = None,
) -> AsrOutcome:
    """Run the configured secondary backend; never raises."""
    outcome = AsrOutcome(name="secondary")
    started = time.monotonic()
    try:
        if config.secondary_asr_backend in SUBPROCESS_SECONDARY_BACKENDS:
            text = _run_subprocess_backend(config, audio_path)
        else:
            if transcriber is None:
                transcriber = build_local_secondary_transcriber(config)
            text = transcriber.transcribe_file(
                audio_path, config.language, duration_seconds=duration_seconds
            )
    except Exception as exc:
        outcome.error = str(exc)
        outcome.seconds = time.monotonic() - started
        return outcome
    outcome.seconds = time.monotonic() - started
    if duration_seconds and outcome.seconds:
        outcome.realtime_factor = round(duration_seconds / outcome.seconds, 2)
    text = normalize_spacing(text)
    if config.secondary_asr_strip_times and is_parakeet_model(config.secondary_asr_model):
        text = strip_parakeet_timestamps(text)
    if text:
        outcome.transcript = Transcript(
            label="secondary",
            model=config.secondary_asr_model,
            kind="asr-final",
            text=text,
            quality_hint=f"secondary ASR pass ({config.secondary_asr_backend} backend)",
        )
    else:
        # The backend ran fine but found nothing to transcribe (e.g. its VAD
        # saw no speech) — that is a skip, not a failure.
        outcome.skip_reason = "secondary ASR skipped: backend produced no text (no speech detected?)"
    return outcome


def build_local_secondary_transcriber(config: "Config") -> "LocalSecondaryTranscriber":
    backend = config.secondary_asr_backend
    if backend == "sherpa":
        return SherpaTranscriber(
            model_name=config.secondary_asr_model,
            device=config.secondary_asr_device,
            download_root=config.download_root,
            num_threads=config.asr_cpu_threads,
            auto_download=config.auto_download,
        )
    if backend == "ctc":
        chunk_length_seconds, stride_seconds = ctc_chunk_config()
        return CTCTranscriber(
            model_name=config.secondary_asr_model,
            device=config.secondary_asr_device,
            download_root=config.download_root,
            chunk_length_seconds=chunk_length_seconds,
            stride_seconds=stride_seconds,
            auto_download=config.auto_download,
        )
    if backend == "pocketsphinx":
        return PocketSphinxTranscriber(
            model_name=config.secondary_asr_model,
            sample_rate=config.sample_rate,
        )
    raise ValueError(f"backend {backend!r} is not an in-process backend")
