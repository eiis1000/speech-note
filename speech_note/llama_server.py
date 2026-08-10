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
    PREFERRED_GGUF_MODEL,
)


# RAM guard. The iGPU allocates from system RAM, so a model that does not fit in
# MemAvailable does not fail cleanly — it swap-thrashes or wedges the compositor.
# The estimate is deliberately crude (weights + a per-token KV allowance + fixed
# overhead); it is a guard against launching something hopeless, not an allocator.
_KV_BYTES_PER_CTX_TOKEN = 32 * 1024
_RAM_OVERHEAD_BYTES = 1_500 * 1024 * 1024


def available_ram_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def model_ram_shortfall(model_path: Path, context_tokens: int) -> int | None:
    """Bytes MISSING to run this model comfortably, or None when it fits (or when
    either quantity is unknowable, in which case the guard stays out of the way)."""
    available = available_ram_bytes()
    if available is None:
        return None
    try:
        weights = model_path.stat().st_size
    except OSError:
        return None
    need = weights + context_tokens * _KV_BYTES_PER_CTX_TOKEN + _RAM_OVERHEAD_BYTES
    return None if need <= available else need - available


def default_server_command(
    *,
    context_tokens: int,
    gpu_layers: str = "auto",
    kv_offload: bool = True,
    model_path: Path | None = None,
) -> list[str] | None:
    explicit = model_path is not None
    if model_path is None:
        # Best installed model that fits wins. The preferred quant beat the bundled
        # E2B decisively on both roles (cleanup keeps and annotation recall, measured
        # in evals/); E2B remains the fallback because it always fits and auto-installs.
        model_path = DEFAULT_GGUF_MODEL
        if PREFERRED_GGUF_MODEL.exists():
            if model_ram_shortfall(PREFERRED_GGUF_MODEL, context_tokens) is None:
                model_path = PREFERRED_GGUF_MODEL
            else:
                print(
                    f"NOTE: {PREFERRED_GGUF_MODEL.name} is installed but does not fit "
                    f"in free RAM at {context_tokens} context tokens; using "
                    f"{DEFAULT_GGUF_MODEL.name} instead.",
                    file=sys.stderr,
                )
    llama_server = shutil.which("llama-server")
    if not llama_server or not model_path.exists():
        return None
    shortfall = model_ram_shortfall(model_path, context_tokens)
    if shortfall is not None:
        gib = shortfall / 1024**3
        if explicit:
            # The user picked this file; proceed, but say what they are in for.
            print(
                f"WARNING: {model_path.name} likely needs ~{gib:.1f} GiB more RAM than "
                "is available; expect heavy swapping or an allocation failure.",
                file=sys.stderr,
            )
        else:
            print(
                "\n".join(
                    (
                        "=" * 72,
                        f"WARNING: not enough free RAM for the local cleanup model "
                        f"({model_path.name} is short ~{gib:.1f} GiB).",
                        "Local cleanup is DISABLED for this run so the machine does not "
                        "swap-thrash.",
                        "Free some memory, lower --organizer-context-tokens, or force a "
                        "model with --organizer-gguf.",
                        "=" * 72,
                    )
                ),
                file=sys.stderr,
            )
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
        # huggingface_hub defaults both of these network operations to 10 seconds.
        # That is too short for a large Xet-backed file on a slow or intermittent
        # connection: even resolving the CDN redirect can exceed it.  The Hub
        # downloader keeps an ``.incomplete`` file and resumes it on the next call,
        # so combine generous per-request timeouts with retries of the whole fetch.
        # Disable Xet for this single large file: its reconstruction buffer can hold
        # downloaded data without advancing the partial file, which defeats reliable
        # byte-range resume and progress detection on an intermittent connection.
        os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        from huggingface_hub import constants as hub_constants
        from huggingface_hub import hf_hub_download

        # The package may already have been imported while loading an ASR model,
        # in which case changing the environment alone is too late.
        hub_constants.HF_HUB_ETAG_TIMEOUT = max(
            hub_constants.HF_HUB_ETAG_TIMEOUT, 60
        )
        hub_constants.HF_HUB_DOWNLOAD_TIMEOUT = max(
            hub_constants.HF_HUB_DOWNLOAD_TIMEOUT, 300
        )
        hub_constants.HF_HUB_DISABLE_XET = True

        DEFAULT_GGUF_MODEL.parent.mkdir(parents=True, exist_ok=True)
        stalled_failures = 0
        while stalled_failures < 10:
            partial_dir = DEFAULT_GGUF_MODEL.parent / ".cache/huggingface/download"
            bytes_before = sum(
                path.stat().st_size for path in partial_dir.glob("*.incomplete")
            ) if partial_dir.exists() else 0
            try:
                hf_hub_download(
                    DEFAULT_GGUF_REPO,
                    DEFAULT_GGUF_FILE,
                    local_dir=str(DEFAULT_GGUF_MODEL.parent),
                )
                break
            except Exception:  # noqa: BLE001 — retry resumable network failures
                bytes_after = sum(
                    path.stat().st_size for path in partial_dir.glob("*.incomplete")
                ) if partial_dir.exists() else 0
                if bytes_after > bytes_before:
                    stalled_failures = 0
                else:
                    stalled_failures += 1
                if stalled_failures == 10:
                    raise
                delay = min(5 * max(stalled_failures, 1), 30)
                print(
                    f"note: cleanup-model download interrupted; resuming in {delay}s "
                    f"({stalled_failures}/10 consecutive attempts made no progress)",
                    file=sys.stderr,
                )
                time.sleep(delay)
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
