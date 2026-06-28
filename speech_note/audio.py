"""Audio measurement and conversion.

All PCM math is numpy; ffmpeg/ffprobe do decoding. Duration is probed once per
input and passed around as a value instead of being re-measured by every layer.
"""

from __future__ import annotations

import dataclasses
import math
import subprocess
import wave
from pathlib import Path

import numpy as np

from .config import (
    CHANNELS,
    NORMALIZE_MAX_GAIN,
    NORMALIZE_PEAK_CEILING_DBFS,
    NORMALIZE_TARGET_RMS_DBFS,
    SAMPLE_WIDTH,
)

INT16_FULL_SCALE = 32767.0


def dbfs(amplitude: float) -> float | None:
    if amplitude <= 0:
        return None
    return round(20.0 * math.log10(amplitude / INT16_FULL_SCALE), 2)


@dataclasses.dataclass(frozen=True)
class PcmStats:
    peak_abs: int
    rms_abs: float

    @property
    def peak_dbfs(self) -> float | None:
        return dbfs(self.peak_abs)

    @property
    def rms_dbfs(self) -> float | None:
        return dbfs(self.rms_abs)

    def as_dict(self) -> dict[str, object]:
        return {
            "peak_abs": self.peak_abs,
            "peak_dbfs": self.peak_dbfs,
            "rms_abs": round(self.rms_abs, 2),
            "rms_dbfs": self.rms_dbfs,
        }


def pcm_stats(pcm: bytes) -> PcmStats:
    if not pcm:
        return PcmStats(peak_abs=0, rms_abs=0.0)
    samples = np.frombuffer(pcm, dtype="<i2")
    if samples.size == 0:
        return PcmStats(peak_abs=0, rms_abs=0.0)
    values = samples.astype(np.float64)
    return PcmStats(
        peak_abs=int(np.max(np.abs(values))),
        rms_abs=float(np.sqrt(np.mean(np.square(values)))),
    )


def pcm_duration_seconds(num_bytes: int, sample_rate: int) -> float:
    return num_bytes / (sample_rate * SAMPLE_WIDTH * CHANNELS)


def write_wav(path: Path, pcm_data: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(CHANNELS)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm_data)


def _run_ffmpeg(command: list[str], *, action: str) -> None:
    result = subprocess.run(
        command,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip().splitlines()
        tail = detail[-1] if detail else f"exit code {result.returncode}"
        raise RuntimeError(f"ffmpeg {action} failed: {tail}")


def convert_to_pcm_wav(source: Path, target: Path, *, sample_rate: int) -> None:
    _run_ffmpeg(
        [
            "ffmpeg", "-nostdin", "-y",
            "-i", str(source),
            "-ac", str(CHANNELS),
            "-ar", str(sample_rate),
            "-f", "wav",
            str(target),
        ],
        action="conversion",
    )


def encode_to_mp3(source: Path, target: Path, *, sample_rate: int) -> None:
    """Transcode to a compact mono mp3 for upload to a remote ASR API.

    A long recording as 16 kHz mono wav is tens of MB — too large to base64 into a
    JSON request body — so the network ASR backend sends mp3 instead.
    """
    _run_ffmpeg(
        [
            "ffmpeg", "-nostdin", "-y",
            "-i", str(source),
            "-ac", str(CHANNELS),
            "-ar", str(sample_rate),
            "-c:a", "libmp3lame", "-q:a", "4",
            str(target),
        ],
        action="mp3 encode",
    )


def extract_audio_chunk(
    source: Path,
    target: Path,
    *,
    start_seconds: float,
    duration_seconds: float,
    sample_rate: int,
) -> None:
    _run_ffmpeg(
        [
            "ffmpeg", "-nostdin", "-y",
            "-i", str(source),
            "-ss", str(start_seconds),
            "-t", str(duration_seconds),
            "-ac", str(CHANNELS),
            "-ar", str(sample_rate),
            str(target),
        ],
        action="chunk extraction",
    )


def probe_duration_seconds(path: Path) -> float:
    command = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nk=1:nw=1",
        str(path),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe duration probe failed with code {result.returncode}")
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"ffprobe returned invalid duration for {path}") from exc


@dataclasses.dataclass(frozen=True)
class NormalizationResult:
    gain: float
    input_stats: PcmStats
    applied: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "gain": round(self.gain, 4),
            "applied": self.applied,
            "input_peak_dbfs": self.input_stats.peak_dbfs,
            "input_rms_dbfs": self.input_stats.rms_dbfs,
        }


def normalize_pcm_wav(source: Path, target: Path) -> NormalizationResult:
    """Apply gain so RMS approaches the target level without clipping."""
    target_rms = INT16_FULL_SCALE * 10 ** (NORMALIZE_TARGET_RMS_DBFS / 20.0)
    peak_ceiling = INT16_FULL_SCALE * 10 ** (NORMALIZE_PEAK_CEILING_DBFS / 20.0)
    with wave.open(str(source), "rb") as handle:
        params = handle.getparams()
        if handle.getnchannels() != CHANNELS or handle.getsampwidth() != SAMPLE_WIDTH:
            raise RuntimeError("normalization input must be mono 16-bit PCM")
        frames = handle.readframes(handle.getnframes())

    samples = np.frombuffer(frames, dtype="<i2").astype(np.float64)
    stats = pcm_stats(frames)
    if samples.size == 0 or stats.peak_abs <= 0 or stats.rms_abs <= 0:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        return NormalizationResult(gain=1.0, input_stats=stats, applied=False)

    gain = min(target_rms / stats.rms_abs, peak_ceiling / stats.peak_abs, NORMALIZE_MAX_GAIN)
    scaled = np.clip(np.rint(samples * gain), -32768, 32767).astype("<i2")
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target), "wb") as out:
        out.setparams(params)
        out.writeframes(scaled.tobytes())
    return NormalizationResult(gain=float(gain), input_stats=stats, applied=True)


def normalize_audio_for_asr(source: Path, target: Path, *, sample_rate: int) -> NormalizationResult:
    """Decode any input to mono PCM at the pipeline rate, then gain-normalize."""
    import tempfile

    with tempfile.TemporaryDirectory(prefix="speech-note-normalize-") as tmp_dir:
        decoded = Path(tmp_dir) / "decoded.wav"
        convert_to_pcm_wav(source, decoded, sample_rate=sample_rate)
        return normalize_pcm_wav(decoded, target)
