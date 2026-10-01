"""Phonon adapter: consent on the caller, inference in its own interpreter."""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import wave
from pathlib import Path

from .install import require_consent
from .terminal import debug_log
from .transcribers import Transcriber, thread_env


def validate_result(result: dict, duration: float | None) -> str:
    """Reject incomplete window coverage before handing text to cleanup."""
    def number(value: object) -> float:
        if type(value) not in (float, int) or not math.isfinite(value):
            raise ValueError("Phonon returned an invalid time")
        return float(value)

    text = result.get("text")
    if not isinstance(text, str) or result.get("truncated") is not False:
        raise ValueError("Phonon returned missing or truncated text")
    seconds = number(result.get("audio_seconds"))
    if seconds < 0 or (duration is not None and abs(seconds - duration) > 0.01):
        raise ValueError("Phonon did not process the full audio duration")
    segments = result.get("segments")
    if not isinstance(segments, list) or (text.strip() and not segments):
        raise ValueError("Phonon returned no window coverage")
    end = 0.0
    parts = []
    for segment in segments:
        if not isinstance(segment, dict) or not isinstance(segment.get("text"), str):
            raise ValueError("Phonon returned an invalid window")
        start, stop = number(segment.get("start")), number(segment.get("end"))
        if abs(start - end) > 0.002 or stop <= start:
            raise ValueError("Phonon returned a gap or overlap in window coverage")
        end = stop
        parts.append(segment["text"])
    if segments and abs(end - seconds) > 0.002:
        raise ValueError("Phonon window coverage ended before the audio")
    if " ".join(text.split()) != " ".join(" ".join(parts).split()):
        raise ValueError("Phonon text disagrees with decoded windows")
    return text


class PhononTranscriber(Transcriber):
    def __init__(self, *, cpu_threads: int, download_root: Path | None,
                 auto_download: bool = False) -> None:
        self.cpu_threads = cpu_threads
        self.download_root = download_root
        self.auto_download = auto_download
        self._prepared = False

    def _invoke(self, operation: str, path: Path | None = None) -> dict:
        binary = shutil.which("speech-note-phonon")
        if binary is None:
            raise RuntimeError("speech-note-phonon not found; enter the Nix dev shell")
        env = thread_env(self.cpu_threads)
        env["FERMION_CPU_THREADS"] = str(self.cpu_threads)
        # Match the measured runtime: idle Torch workers must not spin while
        # the native encoder/decoder owns the CPU. Preserve an explicit override.
        env.setdefault("OMP_WAIT_POLICY", "PASSIVE")
        if self.download_root is not None:
            env["FERMION_CACHE_DIR"] = str(self.download_root / "phonon")
        command = [binary, operation]
        if path is not None:
            command.append(str(path.resolve()))
        result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                                text=True, env=env, check=False)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or f"Phonon exited {result.returncode}")
        try:
            payload = json.loads(result.stdout)
        except ValueError as exc:
            raise RuntimeError("Phonon returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Phonon returned an invalid response")
        return payload

    def ensure_downloaded(self) -> None:
        if self._prepared:
            return
        state = self._invoke("check")
        if state.get("ready") is not True:
            require_consent("Phonon 2", state.get("path", "Phonon cache"),
                            size_hint="164 MB", auto_yes=self.auto_download)
            if self._invoke("download").get("ready") is not True:
                raise RuntimeError("Phonon model download is incomplete")
        self._prepared = True

    def transcribe_file(self, path: Path, language: str, *,
                        duration_seconds: float | None = None) -> str:
        if language.lower() not in {"en", "english"}:
            raise ValueError("Phonon 2 supports English only")
        self.ensure_downloaded()
        # The caller's duration may describe the original MP3 container, whose
        # padding differs from the normalized PCM that we actually decode.
        try:
            with wave.open(str(path), "rb") as audio:
                duration_seconds = audio.getnframes() / audio.getframerate()
        except (wave.Error, EOFError):
            pass
        result = self._invoke("transcribe", path)
        text = validate_result(result, duration_seconds)
        debug_log(f"phonon runtime={result.get('runtime')!r} "
                  f"audio_seconds={result['audio_seconds']} windows={len(result['segments'])}")
        return text
