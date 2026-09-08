"""Audio measurement and conversion.

All PCM math is numpy; ffmpeg/ffprobe do decoding. Duration is probed once per
input and passed around as a value instead of being re-measured by every layer.
"""

from __future__ import annotations

import dataclasses
import math
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

from .config import (
    CHANNELS,
    NORMALIZE_COMPRESS_KNEE_DB,
    NORMALIZE_COMPRESS_LOOKAHEAD_MS,
    NORMALIZE_COMPRESS_RATIO,
    NORMALIZE_COMPRESS_RELEASE_MS,
    NORMALIZE_COMPRESS_THRESHOLD_DBFS,
    NORMALIZE_GAIN_SMOOTH_SECONDS,
    NORMALIZE_LEVEL_PERCENTILE,
    NORMALIZE_LEVEL_WINDOW_SECONDS,
    NORMALIZE_MAX_GAIN_DB,
    NORMALIZE_MIN_GAIN_DB,
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
            "-vn", "-c:a", "pcm_s16le",
            "-ac", str(CHANNELS),
            "-ar", str(sample_rate),
            "-f", "wav",
            str(target),
        ],
        action="conversion",
    )


def encode_to_mp3(
    source: Path,
    target: Path,
    *,
    sample_rate: int,
    bitrate_kbps: int | None = None,
) -> None:
    """Transcode to a compact mono mp3 for upload to a remote ASR API.

    A long recording as 16 kHz mono wav is tens of MB — too large to base64 into a
    JSON request body — so the network ASR backend sends mp3 instead.
    """
    quality = ["-b:a", f"{bitrate_kbps}k"] if bitrate_kbps else ["-q:a", "4"]
    _run_ffmpeg(
        [
            "ffmpeg", "-nostdin", "-y",
            "-i", str(source),
            "-vn",
            "-ac", str(CHANNELS),
            "-ar", str(sample_rate),
            "-c:a", "libmp3lame", *quality,
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
    gain: float  # median of the gain ride; the ride is per-frame, never one constant
    input_stats: PcmStats
    applied: bool
    # Range the ride covered. A wide range means the recording's level really moved.
    gain_db_min: float | None = None
    gain_db_max: float | None = None
    speech_dbfs_before: float | None = None
    speech_dbfs_after: float | None = None
    # p90-p10 of frame loudness over the louder half: how *uneven* the recording is.
    # Falling is the point of leveling; the absolute level alone doesn't show that.
    spread_db_before: float | None = None
    spread_db_after: float | None = None
    compressed_db: float = 0.0  # peak gain reduction the compressor applied
    limited_blocks: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "gain": round(self.gain, 4),
            "applied": self.applied,
            "gain_db_min": self.gain_db_min,
            "gain_db_max": self.gain_db_max,
            "input_peak_dbfs": self.input_stats.peak_dbfs,
            "input_rms_dbfs": self.input_stats.rms_dbfs,
            "speech_dbfs_before": self.speech_dbfs_before,
            "speech_dbfs_after": self.speech_dbfs_after,
            "spread_db_before": self.spread_db_before,
            "spread_db_after": self.spread_db_after,
            "compressed_db": self.compressed_db,
            "limited_blocks": self.limited_blocks,
        }


# --- leveling: slow zero-phase gain ride + look-ahead compressor ---
#
# There is deliberately no speech/silence detection anywhere in here. See
# config.catalog for why: speech is often quieter than the noise around it, so any
# frame classification picks the wrong reference and buries the speech further.

_FRAME_HOP = 320  # 20 ms at 16 kHz — the grid the level envelope lives on
_FRAMES_PER_SEC = 50
_BLOCK = 160  # 10 ms — the grid the compressor and limiter work on


def _frame_dbfs(samples: np.ndarray, *, hop: int = _FRAME_HOP) -> tuple[np.ndarray, int]:
    """Per-frame loudness in dBFS over a 20 ms grid, plus the usable sample count."""
    if len(samples) == 0:
        return np.empty(0), 0
    starts = np.arange(0, len(samples), hop)
    counts = np.minimum(hop, len(samples) - starts)
    rms = np.sqrt(np.add.reduceat(samples ** 2, starts) / counts)
    return 20.0 * np.log10(np.maximum(rms / INT16_FULL_SCALE, 1e-9)), len(samples)


def _centred_percentile(values: np.ndarray, *, window: int, percentile: float) -> np.ndarray:
    """Rolling percentile over a *centred* window — zero-phase by construction.

    Evaluated on a coarse grid and interpolated. The envelope this describes moves over
    tens of seconds, so sampling it every half second loses nothing, and a true
    per-frame percentile over a 90 s window is by far the most expensive thing in the
    normalizer (~10x everything else on a 65-minute file).
    """
    count = len(values)
    if count == 0:
        return np.empty(0)
    half = max(1, window // 2)
    stride = max(1, _FRAMES_PER_SEC // 2)
    anchors = np.arange(0, count, stride)
    sampled = np.array(
        [
            np.percentile(values[max(0, a - half) : min(count, a + half + 1)], percentile)
            for a in anchors
        ]
    )
    if len(anchors) == 1:
        return np.full(count, float(sampled[0]))
    return np.interp(np.arange(count), anchors, sampled)


def _centred_moving_average(values: np.ndarray, width: int) -> np.ndarray:
    """Centred box average via cumsum, with edges averaged over what exists.

    Centred, so it introduces no delay: the smoothed gain still lines up with the audio
    it was derived from. A causal filter would turn the gain up only after the quiet
    passage had gone by.
    """
    count = len(values)
    if count == 0 or width <= 1:
        return values.astype(np.float64)
    width = min(width, count)
    cumulative = np.concatenate([[0.0], np.cumsum(values, dtype=np.float64)])
    half = width // 2
    starts = np.clip(np.arange(count) - half, 0, count)
    ends = np.clip(np.arange(count) - half + width, 0, count)
    return (cumulative[ends] - cumulative[starts]) / np.maximum(ends - starts, 1)


def _gain_curve_db(frame_db: np.ndarray) -> np.ndarray:
    """The slow gain ride, in dB per 20 ms frame.

    Level estimate: a high percentile of frame loudness in a long centred window. No
    gate, no classification — a percentile is already robust to pauses (which a mean is
    not) and to clicks (which a maximum is not).

    Then two successive centred moving averages, which is a triangular kernel: the ride
    only follows a level change that persists on that timescale. A few minutes of quiet
    ends up fully corrected; a couple of seconds of quiet is ridden through untouched.
    Both stages are zero-phase, so the gain never lags the audio.
    """
    level = _centred_percentile(
        frame_db,
        window=int(NORMALIZE_LEVEL_WINDOW_SECONDS * _FRAMES_PER_SEC),
        percentile=NORMALIZE_LEVEL_PERCENTILE,
    )
    wanted = np.clip(
        NORMALIZE_TARGET_RMS_DBFS - level, NORMALIZE_MIN_GAIN_DB, NORMALIZE_MAX_GAIN_DB
    )
    width = max(1, int(NORMALIZE_GAIN_SMOOTH_SECONDS * _FRAMES_PER_SEC))
    smoothed = _centred_moving_average(_centred_moving_average(wanted, width), width)
    # Smoothing can only pull the curve back toward the middle, never past the bounds.
    return np.clip(smoothed, NORMALIZE_MIN_GAIN_DB, NORMALIZE_MAX_GAIN_DB)


def _expand_to_samples(per_frame: np.ndarray, total: int, *, hop: int) -> np.ndarray:
    """Interpolate a per-frame curve to per-sample, so gain changes are continuous.

    Applying a per-frame gain as a step would put a discontinuity at every frame edge.
    These curves move over tens of seconds, so linear interpolation is inaudible and
    leaves nothing for the limiter to clean up.
    """
    if len(per_frame) == 0:
        return np.ones(total)
    centres = np.arange(len(per_frame)) * hop + hop / 2.0
    return np.interp(np.arange(total), centres, per_frame)


def _compress_peaks(samples: np.ndarray, *, block: int = _BLOCK) -> tuple[np.ndarray, float]:
    """Look-ahead soft-knee compressor. Returns (compressed, max reduction in dB).

    This is what handles transients, instead of letting the loudest sample decide the
    gain for the whole recording. Reduction is computed per 10 ms block, then a running
    minimum over the look-ahead window pulls the gain down *before* the transient
    arrives, so the attack has no click. Release is slow enough not to pump.
    """
    if len(samples) == 0:
        return samples, 0.0
    peaks = np.maximum.reduceat(np.abs(samples), np.arange(0, len(samples), block))
    peak_db = 20.0 * np.log10(
        np.maximum(peaks / INT16_FULL_SCALE, 1e-9)
    )

    # Soft knee: no reduction below the knee, full ratio above it, quadratic between.
    over = peak_db - NORMALIZE_COMPRESS_THRESHOLD_DBFS
    knee = max(1e-6, NORMALIZE_COMPRESS_KNEE_DB)
    slope = 1.0 - 1.0 / NORMALIZE_COMPRESS_RATIO
    reduction = np.where(
        over <= -knee / 2,
        0.0,
        np.where(
            over >= knee / 2,
            slope * over,
            slope * (over + knee / 2) ** 2 / (2 * knee),
        ),
    )

    # Look-ahead: each block also obeys the largest reduction coming up shortly.
    ahead = max(1, int(NORMALIZE_COMPRESS_LOOKAHEAD_MS / 10.0))
    padded = np.concatenate([reduction, np.zeros(ahead)])
    windows = np.lib.stride_tricks.sliding_window_view(padded, ahead + 1)
    reduction = windows.max(axis=1)[: len(reduction)]

    # Release: let the reduction decay gradually once the transient has passed.
    release = np.exp(-10.0 / max(1.0, NORMALIZE_COMPRESS_RELEASE_MS))
    held = np.empty_like(reduction)
    current = 0.0
    for index, value in enumerate(reduction):
        current = max(value, current * release)
        held[index] = current

    gain = _expand_to_samples(10 ** (-held / 20.0), len(samples), hop=block)
    return samples * gain, float(held.max())


def _shift(values: np.ndarray, offset: int, fill: float) -> np.ndarray:
    """``values`` shifted by ``offset`` blocks, with ``fill`` past the edges.

    Not np.roll: rolling wraps, which for a gain curve means the end of the
    recording reaches around and attenuates the beginning of it. The edges have no
    neighbour, so they get the neutral value instead of the far end's.
    """
    out = np.full_like(values, fill)
    if offset > 0:
        out[offset:] = values[:-offset]
    elif offset < 0:
        out[:offset] = values[-offset:]
    else:
        out[:] = values
    return out


def _limit(samples: np.ndarray, *, block: int = _BLOCK) -> tuple[np.ndarray, int]:
    """Final safety limiter. With the compressor in front, it should barely engage."""
    ceiling = INT16_FULL_SCALE * 10 ** (NORMALIZE_PEAK_CEILING_DBFS / 20.0)
    if len(samples) == 0:
        return samples, 0
    peaks = np.maximum.reduceat(np.abs(samples), np.arange(0, len(samples), block))
    reduce = np.minimum(1.0, ceiling / np.maximum(peaks, 1e-9))
    # Each block also obeys its neighbours', so the reduction eases in and out
    # rather than stepping at a block edge. 1.0 past the ends: no neighbour there.
    reduce = np.minimum.reduce([reduce, *(_shift(reduce, s, 1.0) for s in (-2, -1, 1, 2))])
    limited = samples * np.repeat(reduce, block)[:len(samples)]
    return limited, int((reduce < 1.0).sum())


def _level_summary(samples: np.ndarray, *, hop: int = _FRAME_HOP) -> tuple[float | None, float | None]:
    """(median loudness, p90-p10 spread) in dBFS over the loud half of the recording.

    Restricted to frames above the median so that long silences don't dominate the
    numbers — this is a report, not a gate, and nothing is removed from the audio on the
    strength of it.
    """
    frame_db, usable = _frame_dbfs(samples, hop=hop)
    if usable == 0:
        return None, None
    loud = frame_db[frame_db >= np.median(frame_db)]
    if loud.size == 0:
        loud = frame_db
    return (
        round(float(np.median(loud)), 2),
        round(float(np.percentile(loud, 90) - np.percentile(loud, 10)), 2),
    )


def normalize_pcm_wav(source: Path, target: Path) -> NormalizationResult:
    """Level the recording and tame its peaks.

    A slow, zero-phase gain ride derived from a long centred percentile of loudness,
    then a look-ahead compressor, then a safety limiter. Nothing is gated, classified,
    or removed. See config.catalog for the reasoning behind each stage.
    """
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

    # Keep the 20 ms level / 10 ms peak grids in time at every supported rate.
    hop = max(1, params.framerate // _FRAMES_PER_SEC)
    block = max(1, params.framerate // 100)
    before_level, before_spread = _level_summary(samples, hop=hop)
    frame_db, _usable = _frame_dbfs(samples, hop=hop)
    gain_db = _gain_curve_db(frame_db)
    leveled = samples * _expand_to_samples(
        10 ** (gain_db / 20.0), len(samples), hop=hop
    )
    leveled, compressed_db = _compress_peaks(leveled, block=block)
    leveled, limited_blocks = _limit(leveled, block=block)
    after_level, after_spread = _level_summary(leveled, hop=hop)

    scaled = np.clip(np.rint(leveled), -32768, 32767).astype("<i2")
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target), "wb") as out:
        out.setparams(params)
        out.writeframes(scaled.tobytes())
    return NormalizationResult(
        gain=float(10 ** (float(np.median(gain_db)) / 20.0)),
        input_stats=stats,
        applied=True,
        gain_db_min=round(float(gain_db.min()), 2) if gain_db.size else None,
        gain_db_max=round(float(gain_db.max()), 2) if gain_db.size else None,
        speech_dbfs_before=before_level,
        speech_dbfs_after=after_level,
        spread_db_before=before_spread,
        spread_db_after=after_spread,
        compressed_db=round(compressed_db, 2),
        limited_blocks=limited_blocks,
    )


def normalize_audio_for_asr(source: Path, target: Path, *, sample_rate: int) -> NormalizationResult:
    """Decode any input to mono PCM at the pipeline rate, then gain-normalize."""
    with tempfile.TemporaryDirectory(prefix="speech-note-normalize-") as tmp_dir:
        decoded = Path(tmp_dir) / "decoded.wav"
        convert_to_pcm_wav(source, decoded, sample_rate=sample_rate)
        return normalize_pcm_wav(decoded, target)
