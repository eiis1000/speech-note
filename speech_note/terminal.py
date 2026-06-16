"""Terminal interaction: status lines, single-key prompts, review panels.

All user-facing presentation lives here or in pipeline reporting — never inside
the compute path.
"""

from __future__ import annotations

import contextlib
import os
import select
import shutil
import subprocess
import sys
import termios
import textwrap
import threading
import time
import tty
from collections.abc import Generator
from datetime import datetime
from pathlib import Path

from .textproc import format_elapsed


def debug_log(message: str) -> None:
    debug_to_file = os.environ.get("SPEECH_NOTE_DEBUG_LOG")
    debug_to_stderr = os.environ.get("SPEECH_NOTE_DEBUG_STDERR")
    if not debug_to_file and not debug_to_stderr:
        return
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    line = f"[speech-note {timestamp}] {message}"
    if debug_to_stderr:
        print(line, file=sys.stderr, flush=True)
    if debug_to_file:
        try:
            path = Path(debug_to_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            pass


class StatusTimer:
    """A single self-updating `label MM:SS` line on stderr."""

    def __init__(self, label: str) -> None:
        self._label = label
        self._stop_event = threading.Event()
        self._started_at = time.monotonic()
        self._lock = threading.Lock()
        self._last_width = 0
        self._print_status()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _print_status(self) -> None:
        text = f"{self._label} {format_elapsed(time.monotonic() - self._started_at)}   "
        padding = " " * max(0, self._last_width - len(text))
        self._last_width = len(text)
        print(f"\r{text}{padding}", end="", file=sys.stderr, flush=True)

    def _worker(self) -> None:
        while not self._stop_event.wait(1.0):
            with self._lock:
                self._print_status()

    def set_label(self, label: str) -> None:
        with self._lock:
            self._label = label
            self._print_status()

    def note(self, message: str) -> None:
        """Print a permanent line above the status line."""
        with self._lock:
            print("\r" + " " * self._last_width + f"\r{message}", file=sys.stderr, flush=True)
            self._last_width = 0
            self._print_status()

    def stop(self) -> None:
        self._stop_event.set()
        print(file=sys.stderr, flush=True)


@contextlib.contextmanager
def status_timer(label: str) -> Generator[StatusTimer]:
    timer = StatusTimer(label)
    try:
        yield timer
    finally:
        timer.stop()


@contextlib.contextmanager
def cbreak_stdin() -> Generator[None]:
    if not sys.stdin or not sys.stdin.isatty():
        yield
        return
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)


def read_single_choice(
    valid_keys: set[str],
    prompt: str,
    *,
    default_key: str | None = None,
) -> str:
    """Read one key from the user; EOF and Enter resolve to the default.

    On a closed/non-interactive stdin with no default this exits with a clear
    message instead of an EOFError traceback.
    """
    lowered = {key.lower() for key in valid_keys}
    default_choice = default_key.lower() if default_key is not None else None
    if default_choice is not None and default_choice not in lowered:
        raise ValueError("default choice must be one of the valid keys")
    if not sys.stdin or not sys.stdin.isatty():
        while True:
            print(prompt)
            try:
                choice = input("> ").strip().lower()
            except EOFError:
                if default_choice is not None:
                    return default_choice
                raise SystemExit(
                    "stdin is not interactive and no default exists for this prompt; "
                    "pass the relevant flag instead"
                ) from None
            if not choice and default_choice is not None:
                return default_choice
            if choice in lowered:
                return choice
            print("Invalid choice.")
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    try:
        print(prompt)
        tty.setraw(fd)
        while True:
            ready, _, _ = select.select([fd], [], [])
            if not ready:
                continue
            char = os.read(fd, 1).decode(errors="ignore").lower()
            if char == "\x03":
                raise KeyboardInterrupt
            if char in {"\r", "\n"} and default_choice is not None:
                print()
                return default_choice
            if char in lowered:
                print(char)
                return char
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)


def wrap_blocks(text: str, width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        if not paragraph.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(paragraph, width=width) or [""])
    return lines or [""]


def print_side_by_side(left_title: str, left_text: str, right_title: str, right_text: str) -> None:
    total_width = shutil.get_terminal_size((160, 40)).columns
    gap = 3
    column_width = max(30, (total_width - gap) // 2)
    left_lines = [left_title] + wrap_blocks(left_text, column_width)
    right_lines = [right_title] + wrap_blocks(right_text, column_width)
    row_count = max(len(left_lines), len(right_lines))
    print()
    print("=" * min(total_width, column_width * 2 + gap))
    for idx in range(row_count):
        left = left_lines[idx] if idx < len(left_lines) else ""
        right = right_lines[idx] if idx < len(right_lines) else ""
        print(f"{left:<{column_width}} | {right:<{column_width}}")
    print("=" * min(total_width, column_width * 2 + gap))


def short_label(title: str) -> str:
    text = title.strip()
    if "," in text:
        text = text.split(",", 1)[0].strip()
    if len(text) <= 32:
        return text
    return text.split("/")[-1]


def copy_with_wl_copy(text: str) -> str:
    if not shutil.which("wl-copy"):
        return "wl-copy not found"
    result = subprocess.run(["wl-copy"], input=text, check=False, text=True)
    if result.returncode != 0:
        return f"wl-copy failed with code {result.returncode}"
    return "ok"
