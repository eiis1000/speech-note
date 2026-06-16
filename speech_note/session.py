"""Run record and artifact persistence.

A Session accumulates facts about one run: transcripts (with provenance),
outcomes, timings, and audio measurements. Events are stamped with seconds
since session start (the same monotonic clock the timings use) plus the wall
start time, so the timeline is reconstructible from one epoch.

Artifacts are committed exactly once, at the end of the run, by ArtifactStore.
"""

from __future__ import annotations

import dataclasses
import json
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from . import __version__
from .audio import PcmStats, pcm_duration_seconds, pcm_stats, write_wav
from .config import QUIET_AUDIO_PEAK_DBFS

if TYPE_CHECKING:
    from .cli import Config
    from .model import AsrOutcome, CleanupOutcome, Transcript


class Session:
    def __init__(self, config: "Config") -> None:
        self.config = config
        self.started_at = datetime.now().astimezone()
        self._monotonic_start = time.monotonic()

        self.transcripts: list[Transcript] = []
        self.asr_outcomes: list[AsrOutcome] = []
        self.cleanup: CleanupOutcome | None = None
        self.errors: list[str] = []
        self.skips: list[str] = []
        self.events: list[dict[str, object]] = []
        self.timings: dict[str, float] = {}
        self.facts: dict[str, object] = {}

        # Audio measurement (live capture only; audio_measured says so).
        self.audio_measured = False
        self.audio_peak_abs_max = 0
        self.audio_rms_abs_max = 0.0
        self.audio_segments: list[dict[str, object]] = []
        self.recorded_audio = bytearray()
        self.recording_sample_rate = config.sample_rate
        self.input_duration_seconds: float | None = None
        self.selected_input_device: dict[str, object] | None = None
        self.available_input_devices: list[dict[str, object]] = []
        self.stream_status_counts: dict[str, int] = {}
        self.audio_queue_high_watermark = 0
        self.audio_queue_full_count = 0

        self.copy_choice: str | None = None
        self.clipboard_status: str | None = None
        self.paths: dict[str, str] = {}
        self.run_failed = False

    # --- clock ---

    def elapsed(self) -> float:
        return time.monotonic() - self._monotonic_start

    def add_event(self, event_type: str, /, **fields: object) -> None:
        self.events.append({"t": round(self.elapsed(), 3), "type": event_type, **fields})

    # --- facts ---

    def add_transcript(self, transcript: "Transcript") -> None:
        self.transcripts.append(transcript)
        self.add_event(
            "transcript",
            label=transcript.label,
            model=transcript.model,
            kind=transcript.kind,
            chars=len(transcript.text),
            words=transcript.words,
        )

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.add_event("error", message=message)

    def add_skip(self, message: str) -> None:
        self.skips.append(message)
        self.add_event("skip", message=message)

    def record_asr_outcome(self, outcome: "AsrOutcome") -> None:
        self.asr_outcomes.append(outcome)
        if outcome.seconds is not None:
            self.note_timing(f"{outcome.name}_asr_seconds", outcome.seconds)
        if outcome.transcript is not None:
            self.add_transcript(outcome.transcript)
        elif outcome.skip_reason:
            self.add_skip(outcome.skip_reason)
        elif outcome.error:
            self.add_error(f"{outcome.name} ASR failed: {outcome.error}")

    def note_timing(self, name: str, seconds: float) -> None:
        self.timings[name] = round(seconds, 3)

    def note_fact(self, name: str, value: object) -> None:
        self.facts[name] = value

    # --- audio measurement (live capture path) ---

    def observe_audio_frame(self, pcm_data: bytes) -> None:
        self.audio_measured = True
        stats = pcm_stats(pcm_data)
        self.audio_peak_abs_max = max(self.audio_peak_abs_max, stats.peak_abs)
        self.audio_rms_abs_max = max(self.audio_rms_abs_max, stats.rms_abs)
        self.recorded_audio.extend(pcm_data)

    def add_audio_segment(self, pcm_data: bytes, *, sample_rate: int) -> None:
        stats = pcm_stats(pcm_data).as_dict()
        stats["duration_seconds"] = round(pcm_duration_seconds(len(pcm_data), sample_rate), 3)
        self.audio_segments.append(stats)

    def note_stream_status(self, status: object, *, queue_depth: int) -> None:
        key = str(status)
        self.stream_status_counts[key] = self.stream_status_counts.get(key, 0) + 1
        self.add_event("stream-status", status=key, queue_depth=queue_depth)

    def note_audio_queue_depth(self, depth: int) -> None:
        self.audio_queue_high_watermark = max(self.audio_queue_high_watermark, depth)

    def note_audio_queue_full(self) -> None:
        self.audio_queue_full_count += 1

    def audio_level_warning(self) -> str | None:
        if not self.audio_measured:
            return None
        peak_dbfs = PcmStats(self.audio_peak_abs_max, self.audio_rms_abs_max).peak_dbfs
        if peak_dbfs is None:
            return "audio level warning: no usable input signal detected"
        if peak_dbfs <= QUIET_AUDIO_PEAK_DBFS:
            return f"audio level warning: peak {peak_dbfs} dBFS; input is very quiet"
        return None

    # --- derived views ---

    def transcript_by_label(self, label: str) -> "Transcript | None":
        for transcript in self.transcripts:
            if transcript.label == label:
                return transcript
        return None

    def raw_text(self) -> str:
        primary = self.transcript_by_label("primary") or self.transcript_by_label("live")
        if primary is not None:
            return primary.text
        for transcript in self.transcripts:
            if transcript.kind in {"asr-final", "asr-live", "user"}:
                return transcript.text
        return ""

    def clean_text(self) -> str:
        return self.cleanup.text if self.cleanup is not None else ""

    @property
    def produced_output(self) -> bool:
        return bool(self.raw_text() or self.clean_text())

    def timestamp_slug(self) -> str:
        return self.started_at.strftime("%Y%m%d-%H%M%S-%f")


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    return value


class ArtifactStore:
    """Writes all run artifacts once, at commit time.

    Layout under artifacts_dir:
      raw.latest, clean.latest, recording.latest.wav, diagnostics.latest.json
      logs/<timestamp>-{raw,clean,recording.wav,diagnostics.json}
    """

    def __init__(self, artifacts_dir: Path, archive_dir: Path) -> None:
        self.artifacts_dir = artifacts_dir
        self.archive_dir = archive_dir

    def commit(self, session: Session) -> None:
        slug = session.timestamp_slug()
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

        def write_pair(name: str, archive_suffix: str, text: str) -> None:
            content = text + ("\n" if text else "")
            latest = self.artifacts_dir / name
            archived = self.archive_dir / f"{slug}-{archive_suffix}"
            latest.write_text(content, encoding="utf-8")
            archived.write_text(content, encoding="utf-8")
            session.paths[f"latest_{archive_suffix}"] = str(latest)
            session.paths[f"archive_{archive_suffix}"] = str(archived)

        write_pair("raw.latest", "raw", session.raw_text())
        write_pair("clean.latest", "clean", session.clean_text())

        if session.recorded_audio:
            latest_wav = self.artifacts_dir / "recording.latest.wav"
            archive_wav = self.archive_dir / f"{slug}-recording.wav"
            write_wav(latest_wav, bytes(session.recorded_audio), session.recording_sample_rate)
            write_wav(archive_wav, bytes(session.recorded_audio), session.recording_sample_rate)
            session.paths["latest_recording"] = str(latest_wav)
            session.paths["archive_recording"] = str(archive_wav)

        latest_diag = self.artifacts_dir / "diagnostics.latest.json"
        archive_diag = self.archive_dir / f"{slug}-diagnostics.json"
        session.paths["latest_diagnostics"] = str(latest_diag)
        session.paths["archive_diagnostics"] = str(archive_diag)
        payload = json.dumps(self.diagnostics_payload(session), indent=2, ensure_ascii=True) + "\n"
        latest_diag.write_text(payload, encoding="utf-8")
        archive_diag.write_text(payload, encoding="utf-8")

    @staticmethod
    def diagnostics_payload(session: Session) -> dict[str, object]:
        config = session.config
        cleanup = session.cleanup
        return {
            "schema": 2,
            "version": __version__,
            "started_at": session.started_at.isoformat(),
            "duration_seconds": round(session.elapsed(), 3),
            "config": _jsonable(dataclasses.asdict(config)),
            "input": {
                "duration_seconds": session.input_duration_seconds,
                "selected_input_device": session.selected_input_device,
                "available_input_devices": session.available_input_devices,
                "recording_sample_rate": session.recording_sample_rate,
            },
            "transcripts": [
                {
                    "label": t.label,
                    "model": t.model,
                    "kind": t.kind,
                    "chars": len(t.text),
                    "words": t.words,
                    "text": t.text,
                }
                for t in session.transcripts
            ],
            "asr": [
                {
                    "name": outcome.name,
                    "ok": outcome.ok,
                    "skip_reason": outcome.skip_reason,
                    "error": outcome.error,
                    "seconds": outcome.seconds,
                    "realtime_factor": outcome.realtime_factor,
                }
                for outcome in session.asr_outcomes
            ],
            "cleanup": None
            if cleanup is None
            else {
                "method": cleanup.method,
                "served_model": cleanup.served_model,
                "finish_reason": cleanup.finish_reason,
                "flagged_short": cleanup.flagged_short,
                "error": cleanup.error,
                "warning": cleanup.warning,
                "estimated_prompt_tokens": cleanup.estimated_prompt_tokens,
                "requested_output_tokens": cleanup.requested_output_tokens,
                "request_timeout": cleanup.request_timeout,
                "seconds": cleanup.seconds,
            },
            "audio_levels": {
                "measured": session.audio_measured,
                "recording_duration_seconds": round(
                    pcm_duration_seconds(len(session.recorded_audio), session.recording_sample_rate), 3
                ),
                "max_peak_abs": session.audio_peak_abs_max,
                "max_peak_dbfs": PcmStats(session.audio_peak_abs_max, session.audio_rms_abs_max).peak_dbfs,
                "max_rms_abs": round(session.audio_rms_abs_max, 2),
                "max_rms_dbfs": PcmStats(session.audio_peak_abs_max, session.audio_rms_abs_max).rms_dbfs,
                "segments": session.audio_segments,
            },
            "stream": {
                "status_counts": session.stream_status_counts,
                "audio_queue_high_watermark": session.audio_queue_high_watermark,
                "audio_queue_full_count": session.audio_queue_full_count,
            },
            "facts": _jsonable(session.facts),
            "timings": session.timings,
            "errors": session.errors,
            "skips": session.skips,
            "copy_choice": session.copy_choice,
            "clipboard_status": session.clipboard_status,
            "paths": session.paths,
            "events": session.events,
        }
