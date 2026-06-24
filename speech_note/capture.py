"""Live microphone / replay capture.

CaptureRunner owns the capture-phase lifecycle: the audio stream (or replay
feeder), the VAD segmenter, the live-preview transcriber, and the prewarm
threads. It hands back to the pipeline once capture ends; signal handlers are
restored before finalization so Ctrl-C still works during the slow stages.
"""

from __future__ import annotations

import contextlib
import os
import queue
import select
import shutil
import signal
import sys
import tempfile
import threading
import time
import wave
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, cast

import warnings

import sounddevice as sd

# webrtcvad still imports pkg_resources; not actionable from here.
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)
import webrtcvad  # noqa: E402

from .audio import convert_to_pcm_wav, write_wav
from .config import LIVE_ASR_MODEL, SAMPLE_WIDTH
from .asr import build_live_transcriber, build_transcriber, build_transcribers
from .devices import choose_live_capture_sample_rate, choose_replay_capture_sample_rate
from .model import Transcript
from .organizer import build_organizer
from .pipeline import (
    finalize,
    load_extra_transcripts,
    run_final_asr,
    start_organizer_prewarm,
)
from .session import Session
from .terminal import cbreak_stdin, status_timer
from .textproc import format_elapsed, normalize_spacing
from .transcribers import FasterWhisperTranscriber, WhisperCppTranscriber

if TYPE_CHECKING:
    from .cli import Config

STOP_DRAIN_SECONDS = 0.7
AUDIO_QUEUE_MAX = 512


class SegmentCollector:
    """webrtcvad-based utterance segmenter over fixed-size PCM frames."""

    def __init__(
        self,
        *,
        sample_rate: int,
        frame_ms: int,
        vad_mode: int,
        start_padding_ms: int,
        end_silence_ms: int,
        min_speech_ms: int,
        max_segment_seconds: float,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_bytes = sample_rate * frame_ms // 1000 * SAMPLE_WIDTH
        self.vad = webrtcvad.Vad(vad_mode)
        self.pre_roll: deque[bytes] = deque(maxlen=max(1, start_padding_ms // frame_ms))
        self.end_silence_frames = max(1, end_silence_ms // frame_ms)
        self.min_speech_frames = max(1, min_speech_ms // frame_ms)
        self.max_segment_frames = max(1, int(max_segment_seconds * 1000 / frame_ms))
        self.current_frames: list[bytes] = []
        self.speaking = False
        self.voiced_frames = 0
        self.unvoiced_frames = 0

    def _reset(self) -> bytes:
        segment = b"".join(self.current_frames)
        self.current_frames = []
        self.speaking = False
        self.voiced_frames = 0
        self.unvoiced_frames = 0
        return segment

    def process_frame(self, frame: bytes) -> list[bytes]:
        if len(frame) != self.frame_bytes:
            return []
        voiced = self.vad.is_speech(frame, self.sample_rate)
        if not self.speaking:
            self.pre_roll.append(frame)
            if voiced:
                self.speaking = True
                self.current_frames = list(self.pre_roll)
                self.pre_roll.clear()
                self.voiced_frames = 1
                self.unvoiced_frames = 0
            return []

        self.current_frames.append(frame)
        if voiced:
            self.voiced_frames += 1
            self.unvoiced_frames = 0
        else:
            self.unvoiced_frames += 1

        should_finalize = (
            self.voiced_frames >= self.min_speech_frames
            and self.unvoiced_frames >= self.end_silence_frames
        ) or len(self.current_frames) >= self.max_segment_frames
        if not should_finalize:
            if (
                self.unvoiced_frames >= self.end_silence_frames
                and self.voiced_frames < self.min_speech_frames
            ):
                # A blip shorter than min-speech followed by silence: discard it
                # instead of accumulating silence until the max-segment cap.
                self._reset()
            return []
        return [self._reset()]

    def flush(self) -> list[bytes]:
        if not self.current_frames or self.voiced_frames < self.min_speech_frames:
            self._reset()
            return []
        return [self._reset()]


class CaptureRunner:
    def __init__(self, config: "Config", session: Session) -> None:
        self.config = config
        self.session = session
        self.replay_mode = config.replay_input_file is not None
        self.sample_rate = (
            choose_replay_capture_sample_rate(config.sample_rate)
            if self.replay_mode
            else choose_live_capture_sample_rate(config.input_device, config.sample_rate)
        )
        session.recording_sample_rate = self.sample_rate
        self.collector = SegmentCollector(
            sample_rate=self.sample_rate,
            frame_ms=config.frame_ms,
            vad_mode=config.vad_mode,
            start_padding_ms=config.start_padding_ms,
            end_silence_ms=config.end_silence_ms,
            min_speech_ms=config.min_speech_ms,
            max_segment_seconds=config.max_segment_seconds,
        )
        self.audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=AUDIO_QUEUE_MAX)
        self.segment_queue: queue.Queue[Path | None] = queue.Queue()
        self.temp_dir = Path(tempfile.mkdtemp(prefix="speech-note-capture-"))
        self.frame_buffer = bytearray()
        self.stop_event = threading.Event()
        self.stop_requested_at: float | None = None
        self.replay_finished = threading.Event()
        self.live_chunks: list[str] = []
        self.live_transcriber_ready = threading.Event()
        self.live_transcriber: FasterWhisperTranscriber | None = None
        self.recording_started_at: float | None = None
        self._terminal_lock = threading.Lock()
        self._status_width = shutil.get_terminal_size((120, 40)).columns
        self._last_warning_at = 0.0
        self._threads: list[threading.Thread] = []

    # --- terminal ---

    def _print_status(self, message: str) -> None:
        padded = message[: self._status_width].ljust(self._status_width)
        with self._terminal_lock:
            print(f"\r{padded}", end="", file=sys.stderr, flush=True)

    def _print_event(self, message: str) -> None:
        with self._terminal_lock:
            print("\r" + " " * self._status_width + "\r", end="", file=sys.stderr, flush=True)
            print(message, file=sys.stderr, flush=True)

    def _request_stop(self, message: str) -> None:
        if self.stop_requested_at is not None:
            return
        self.stop_requested_at = time.monotonic()
        self.stop_event.set()
        with self._terminal_lock:
            print(file=sys.stderr, flush=True)
        print(message, file=sys.stderr, flush=True)

    # --- workers ---

    def _spawn(self, target, name: str) -> None:
        thread = threading.Thread(target=target, name=name, daemon=True)
        thread.start()
        self._threads.append(thread)

    def _timer_worker(self) -> None:
        while not self.stop_event.wait(1.0):
            if self.recording_started_at is None:
                continue
            elapsed = format_elapsed(time.monotonic() - self.recording_started_at)
            if self.replay_mode:
                self._print_status(f"Replaying {elapsed}. Press Enter to stop early.")
            else:
                self._print_status(f"Listening {elapsed}. Press Enter when done.")

    def _load_live_transcriber(self) -> None:
        started = time.monotonic()
        try:
            transcriber = build_live_transcriber(self.config)
            # Force the (now lazy) load here in the background so the model is
            # warm before the first segment arrives, and so a load failure is
            # reported once — not per segment in the transcribe worker.
            transcriber.ensure_loaded()
            self.live_transcriber = transcriber
        except Exception as exc:
            self.session.add_error(f"live preview ASR failed to initialize: {exc}")
        finally:
            self.session.note_timing("live_model_load_seconds", time.monotonic() - started)
            self.live_transcriber_ready.set()

    def _prewarm_primary(self) -> None:
        """Warm a final whisper.cpp ASR source while recording: page-cache its ggml
        file (the whisper.cpp subprocess cannot stay resident)."""
        whisper_source = next(
            (s for s in self.config.asr_sources if s.backend == "whisper-cpp"), None
        )
        if whisper_source is None:
            return
        try:
            transcriber = build_transcriber(self.config, whisper_source)
        except Exception:
            return
        if isinstance(transcriber, WhisperCppTranscriber):
            started = time.monotonic()
            transcriber.warm_model_cache()
            self.session.note_timing("primary_model_cache_warm_seconds", time.monotonic() - started)

    def _transcribe_worker(self) -> None:
        while True:
            try:
                item = self.segment_queue.get(timeout=0.5)
            except queue.Empty:
                if self.stop_event.is_set():
                    return
                continue
            if item is None:
                return
            try:
                self.live_transcriber_ready.wait(timeout=60.0)
                transcriber = self.live_transcriber
                if transcriber is None:
                    continue
                text = normalize_spacing(
                    transcriber.transcribe_file(item, language=self.config.language)
                )
                if text:
                    self.live_chunks.append(text)
                    self._print_event(text)
            except Exception as exc:
                self.session.add_event("live-transcribe-error", message=str(exc))
            finally:
                with contextlib.suppress(FileNotFoundError):
                    item.unlink()

    def _replay_feeder(self, replay_wav: Path) -> None:
        frames_per_chunk = self.sample_rate * self.config.frame_ms // 1000
        sleep_seconds = (self.config.frame_ms / 1000.0) / max(self.config.replay_speed, 0.01)
        next_emit_at = time.monotonic()
        with wave.open(str(replay_wav), "rb") as handle:
            while not self.stop_event.is_set():
                chunk = handle.readframes(frames_per_chunk)
                if not chunk:
                    break
                try:
                    self.audio_queue.put_nowait(chunk)
                    self.session.note_audio_queue_depth(self.audio_queue.qsize())
                except queue.Full:
                    self.session.note_audio_queue_full()
                next_emit_at += sleep_seconds
                delay = next_emit_at - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
        self.replay_finished.set()

    def _stream_callback(self, indata, frames, stream_time, status) -> None:
        del frames, stream_time
        if status:
            self.session.note_stream_status(status, queue_depth=self.audio_queue.qsize())
            now = time.monotonic()
            if now - self._last_warning_at >= 5.0:
                self._print_event(f"audio-status: {status}")
                self._last_warning_at = now
        try:
            self.audio_queue.put_nowait(bytes(indata))
            self.session.note_audio_queue_depth(self.audio_queue.qsize())
        except queue.Full:
            self.session.note_audio_queue_full()
            now = time.monotonic()
            if now - self._last_warning_at >= 5.0:
                self._print_event("audio queue full; dropping frames")
                self._last_warning_at = now

    # --- frame handling ---

    def _handle_frame(self, frame: bytes) -> None:
        self.session.observe_audio_frame(frame)
        self.frame_buffer.extend(frame)
        while len(self.frame_buffer) >= self.collector.frame_bytes:
            pcm = bytes(self.frame_buffer[: self.collector.frame_bytes])
            del self.frame_buffer[: self.collector.frame_bytes]
            for segment in self.collector.process_frame(pcm):
                self._enqueue_segment(segment)

    def _enqueue_segment(self, segment: bytes) -> None:
        self.session.add_audio_segment(segment, sample_rate=self.sample_rate)
        segment_path = self.temp_dir / f"segment-{time.time_ns()}.wav"
        write_wav(segment_path, segment, self.sample_rate)
        self.segment_queue.put(segment_path)

    def _check_stdin_for_stop(self) -> None:
        if not sys.stdin or not sys.stdin.isatty():
            return
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            return
        char = os.read(sys.stdin.fileno(), 1).decode(errors="ignore")
        if char == "\x03":
            raise KeyboardInterrupt
        if char in {"\r", "\n"}:
            self._request_stop("\nStopping capture. Finishing transcription...")

    def _consume_until_stopped(self) -> None:
        with cbreak_stdin():
            while True:
                self._check_stdin_for_stop()
                if self.replay_mode and self.replay_finished.is_set() and self.stop_requested_at is None:
                    self._request_stop("\nReplay finished. Finishing transcription...")
                if self.stop_event.is_set() and self.stop_requested_at is None:
                    # Stop came from a signal handler.
                    self.stop_requested_at = time.monotonic()
                try:
                    frame = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    if (
                        self.stop_requested_at is not None
                        and time.monotonic() - self.stop_requested_at >= STOP_DRAIN_SECONDS
                    ):
                        return
                    continue
                self._handle_frame(frame)
                if (
                    self.stop_requested_at is not None
                    and time.monotonic() - self.stop_requested_at >= STOP_DRAIN_SECONDS
                    and self.audio_queue.empty()
                ):
                    return

    # --- main entry ---

    def run(self) -> Path | None:
        """Capture audio; returns the recorded session wav (or None)."""
        config = self.config
        session = self.session
        if self.replay_mode:
            session.selected_input_device = {
                "name": "replay-input-file",
                "source_file": str(config.replay_input_file),
                "capture_samplerate": self.sample_rate,
            }
        else:
            with contextlib.suppress(Exception):
                info = cast("dict[str, object]", sd.query_devices(config.input_device, "input"))
                session.selected_input_device = {
                    "name": info["name"],
                    "max_input_channels": info["max_input_channels"],
                    "default_samplerate": info["default_samplerate"],
                    "capture_samplerate": self.sample_rate,
                }

        self._spawn(self._transcribe_worker, "live-transcribe")
        previous_sigint = signal.signal(signal.SIGINT, self._signal_stop)
        previous_sigterm = signal.signal(signal.SIGTERM, self._signal_stop)
        try:
            if self.replay_mode:
                replay_wav = self.temp_dir / "replay-input.wav"
                convert_to_pcm_wav(config.replay_input_file, replay_wav, sample_rate=self.sample_rate)
                self._start_capture_threads()
                self._spawn(lambda: self._replay_feeder(replay_wav), "replay-feeder")
                self._consume_until_stopped()
            else:
                with sd.RawInputStream(
                    samplerate=self.sample_rate,
                    blocksize=0,
                    dtype="int16",
                    channels=1,
                    device=config.input_device,
                    latency="high",
                    callback=self._stream_callback,
                ):
                    self._start_capture_threads()
                    self._consume_until_stopped()
        finally:
            self.stop_event.set()
            signal.signal(signal.SIGINT, previous_sigint)
            signal.signal(signal.SIGTERM, previous_sigterm)
        return self._drain_and_collect()

    def _signal_stop(self, signum: int, frame: object) -> None:
        del signum, frame
        self.stop_event.set()

    def _start_capture_threads(self) -> None:
        config = self.config
        self.recording_started_at = time.monotonic()
        if self.replay_mode:
            print(
                f"Replaying {config.replay_input_file}. Press Enter to stop early.",
                file=sys.stderr,
                flush=True,
            )
        else:
            print("Listening. Press Enter when done. Ctrl-C also stops.", file=sys.stderr, flush=True)
        if self.sample_rate != config.sample_rate:
            print(
                f"Live capture sample rate: {self.sample_rate} Hz (requested {config.sample_rate} Hz).",
                file=sys.stderr,
                flush=True,
            )
        print("Raw transcript will print live below.", file=sys.stderr, flush=True)
        self._spawn(self._timer_worker, "status-timer")
        self._spawn(self._load_live_transcriber, "live-model-load")
        self._spawn(self._prewarm_primary, "primary-prewarm")

    def _drain_and_collect(self) -> Path | None:
        # Drain whatever the callback enqueued before the stream closed.
        while True:
            try:
                frame = self.audio_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_frame(frame)
        for segment in self.collector.flush():
            self._enqueue_segment(segment)
        self.segment_queue.put(None)
        for thread in self._threads:
            if thread.name == "live-transcribe":
                thread.join(timeout=120)
                if thread.is_alive():
                    self.session.add_event("live-transcribe-stalled")
        if self.live_chunks:
            self.session.add_transcript(
                Transcript(
                    label="live",
                    model=LIVE_ASR_MODEL,
                    kind="asr-live",
                    text="\n".join(self.live_chunks),
                    quality_hint=(
                        "live preview from a small fast model on VAD segments; "
                        "lowest-quality source"
                    ),
                )
            )
        if not self.session.recorded_audio:
            return None
        recording_path = self.temp_dir / "session-recording.wav"
        write_wav(recording_path, bytes(self.session.recorded_audio), self.sample_rate)
        return recording_path

    def cleanup(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)


def run_capture_pipeline(config: "Config") -> Session:
    session = Session(config)
    organizer, supervisor = build_organizer(config)
    load_extra_transcripts(config, session)
    runner = CaptureRunner(config, session)
    prewarm = start_organizer_prewarm(config, organizer)
    try:
        try:
            recording_path = runner.run()
        except Exception as exc:
            session.add_error(f"audio input failed: {exc}")
            raise SystemExit(f"audio input failed: {exc}") from None
        if recording_path is not None:
            with status_timer("Loading ASR models"):
                built = build_transcribers(config)
            run_final_asr(config, session, recording_path, built=built)
        finalize(config, session, organizer)
    finally:
        if prewarm is not None:
            prewarm.join(timeout=0.1)
        if supervisor is not None:
            supervisor.close()
        runner.cleanup()
    return session
