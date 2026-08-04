"""The optional local llama-server child process.

Launching it, waiting for health, tearing it down, and installing the bundled cleanup
GGUF it needs. Separate from the chat transport: a remote provider uses none of this.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Sequence
from pathlib import Path

import requests

from .config import (
    DEFAULT_GGUF_FILE,
    DEFAULT_GGUF_MODEL,
    DEFAULT_GGUF_REPO,
    DEFAULT_GGUF_SIZE_HINT,
)


def default_server_command(
    *,
    context_tokens: int,
    gpu_layers: str = "auto",
    kv_offload: bool = True,
    model_path: Path | None = None,
) -> list[str] | None:
    model_path = model_path or DEFAULT_GGUF_MODEL
    llama_server = shutil.which("llama-server")
    if not llama_server or not model_path.exists():
        return None
    command = [
        llama_server,
        "--model", str(model_path),
        "--host", "127.0.0.1",
        "--port", "8011",
        "--parallel", "1",
        "--ctx-size", str(context_tokens),
        "--gpu-layers", str(gpu_layers),
        "--reasoning", "off",
        "--jinja",
        "--no-warmup",
    ]
    if not kv_offload:
        # Escape hatch for older llama.cpp/Vulkan stacks where allocating the
        # split SWA/shared KV cache on the GPU hung during slot init. Fixed by
        # b9190 (measured: KV on GPU is ~1.5x faster generation, no hang), so
        # KV offload is on by default; --no-organizer-kv-offload restores this.
        command.append("--no-kv-offload")
    return command


def ensure_default_cleanup_model(*, auto_yes: bool, interactive: bool | None = None) -> bool:
    """Install the bundled cleanup GGUF if it's missing. Returns True if present.

    Consent-gated like the ASR models (``--auto-download`` answers yes). Only the
    default quant has a known source — a user-supplied ``--organizer-gguf`` path is
    never fetched here. A download failure (offline, wrong repo) is non-fatal: it
    returns False so the caller falls back to its existing "place a GGUF" notice.
    """
    if DEFAULT_GGUF_MODEL.exists():
        return True
    from .install import ModelInstallDeclined, require_consent

    try:
        require_consent(
            f"cleanup LM '{DEFAULT_GGUF_FILE}' ({DEFAULT_GGUF_REPO})",
            str(DEFAULT_GGUF_MODEL.parent),
            size_hint=DEFAULT_GGUF_SIZE_HINT,
            auto_yes=auto_yes,
            interactive=interactive,
        )
    except ModelInstallDeclined:
        return False
    try:
        from huggingface_hub import hf_hub_download

        DEFAULT_GGUF_MODEL.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            DEFAULT_GGUF_REPO,
            DEFAULT_GGUF_FILE,
            local_dir=str(DEFAULT_GGUF_MODEL.parent),
        )
    except Exception as exc:  # noqa: BLE001 — any fetch failure should fall back, not crash
        print(f"note: could not download cleanup model: {exc}", file=sys.stderr)
        return False
    return DEFAULT_GGUF_MODEL.exists()


class LocalServerSupervisor:
    """Launches and supervises a local llama-server when one isn't running."""

    def __init__(self, *, launch_command: Sequence[str] | None, healthcheck_url: str) -> None:
        self.launch_command = launch_command
        self.healthcheck_url = healthcheck_url
        self.process: subprocess.Popen[str] | None = None
        self.log_path: Path | None = None
        self._log_handle = None
        self._lock = threading.Lock()

    def is_healthy(self) -> bool:
        try:
            return requests.get(self.healthcheck_url, timeout=1.5).ok
        except requests.RequestException:
            return False

    def ensure_running(self, *, startup_timeout: float = 45.0) -> None:
        with self._lock:
            if self.is_healthy():
                return
            if not self.launch_command:
                return
            if self.process is None or self.process.poll() is not None:
                fd, log_name = tempfile.mkstemp(prefix="speech-note-llama-server-", suffix=".log")
                os.close(fd)
                self.log_path = Path(log_name)
                self._log_handle = self.log_path.open("a", encoding="utf-8")
                self.process = subprocess.Popen(
                    self.launch_command,
                    stdout=self._log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    text=True,
                )
            deadline = time.monotonic() + startup_timeout
            while time.monotonic() < deadline:
                if self.process is not None and self.process.poll() is not None:
                    raise RuntimeError("organizer server exited during startup" + self._log_tail_suffix())
                if self.is_healthy():
                    return
                time.sleep(0.5)
            raise RuntimeError("organizer server did not become ready" + self._log_tail_suffix())

    def close(self) -> None:
        with self._lock:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None

    def log_tail(self, max_chars: int = 4000) -> str:
        if self._log_handle is not None:
            with contextlib.suppress(Exception):
                self._log_handle.flush()
        if self.log_path is None or not self.log_path.exists():
            return ""
        with contextlib.suppress(Exception):
            return self.log_path.read_text(encoding="utf-8", errors="replace")[-max_chars:].strip()
        return ""

    def _log_tail_suffix(self) -> str:
        tail = self.log_tail()
        return f"; llama-server log tail: {' '.join(tail.split())}" if tail else ""
