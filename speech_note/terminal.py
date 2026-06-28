"""Terminal interaction: status lines, single-key prompts, review panels.

All user-facing presentation lives here or in pipeline reporting — never inside
the compute path.
"""

from __future__ import annotations

import contextlib
import dataclasses
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


@dataclasses.dataclass
class StatusTask:
    """One unit of work within a phase: a name, a start, and (when finished) an end."""

    name: str
    started_at: float
    done_at: float | None = None
    error: bool = False

    def elapsed(self, now: float) -> float:
        return (self.done_at if self.done_at is not None else now) - self.started_at


SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
SPINNER_DONE = "✓"


def render_status_line(
    phase: str,
    tasks: list[StatusTask],
    *,
    phase_started: float,
    now: float,
    width: int,
    spinner: str,
) -> str:
    """Build the one-line status string for a phase and its tasks (pure, testable).

    Layout: ``{spinner} {phase}   t1 · t2 · …<pad>{total}`` — a leading spinner glyph
    (resolves to ✓ when the phase ends), the phase label, middot-separated tasks each
    showing live elapsed (✓ done, ✗ errored), and the phase total right-aligned to the
    terminal edge.

    Degrades on a narrow terminal instead of wrapping: the left anchor (spinner +
    phase + start of the task list) and the right total are always kept; the middle is
    elided with … . Never returns more than ``width-1`` columns, so it never triggers a
    line wrap."""
    total = format_elapsed(now - phase_started)
    if tasks:
        parts: list[str] = []
        for task in tasks:
            elapsed = format_elapsed(task.elapsed(now))
            mark = "✗" if task.error else ("✓" if task.done_at is not None else "")
            parts.append(f"{task.name} {mark}{elapsed}")
        left = f"{spinner} {phase}   " + " · ".join(parts)
    else:
        left = f"{spinner} {phase}"
    return _compose_status(left, total, width)


def _compose_status(left: str, total: str, width: int) -> str:
    """Right-align ``total`` against ``left`` within ``width``, keeping both ends
    visible when the terminal is too thin (truncate the left's tail, never the total)."""
    if width <= 0:
        return f"{left}   {total}"
    usable = width - 1  # leave the last column empty so terminals don't auto-wrap
    left_budget = usable - len(total) - 1  # one space minimum before the total
    if left_budget < 1:
        # Pathologically narrow: the time is the must-keep; show its tail.
        return total[-usable:]
    if len(left) <= left_budget:
        pad = usable - len(left) - len(total)
        return left + " " * max(1, pad) + total
    return left[: left_budget - 1] + "…" + " " + total


class StatusDisplay:
    """Single owner of the stderr status line.

    On a TTY it renders the active phase and its (possibly concurrent) tasks as one
    self-updating line, repainted ~once a second by one background thread. On a
    non-TTY (redirected stderr, --full-auto logs) it prints a single milestone line
    when a phase ends instead — so output never fills with carriage returns. Only
    one phase is active at a time, but a phase may hold several concurrent tasks, and
    tasks may be opened/closed from worker threads (the parallel ASR sources do
    exactly this), so every mutation takes the lock."""

    def __init__(self, *, tick: float = 0.1) -> None:
        self._tick = tick  # ~10 fps so the spinner animates smoothly
        self._is_tty = False
        self._lock = threading.RLock()
        self._phase: str | None = None
        self._phase_started = 0.0
        self._tasks: dict[str, StatusTask] = {}
        self._last_width = 0
        self._frame = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def begin_phase(self, label: str) -> None:
        with self._lock:
            self._phase = label
            self._phase_started = time.monotonic()
            self._tasks = {}
            self._is_tty = bool(getattr(sys.stderr, "isatty", lambda: False)())
            if self._is_tty:
                self._ensure_thread()
                self._paint()

    def end_phase(self) -> None:
        with self._lock:
            if self._phase is None:
                return
            now = time.monotonic()
            if self._is_tty:
                self._paint(now, done=True)  # leave the resolved (✓) snapshot in scrollback
                sys.stderr.write("\n")
                self._last_width = 0
            else:
                sys.stderr.write(self._summary(now) + "\n")
            sys.stderr.flush()
            self._phase = None
            self._tasks = {}

    def start_task(self, name: str) -> None:
        with self._lock:
            self._tasks[name] = StatusTask(name=name, started_at=time.monotonic())
            if self._is_tty:
                self._paint()

    def finish_task(self, name: str, *, error: bool = False) -> None:
        with self._lock:
            task = self._tasks.get(name)
            if task is None:
                return
            task.done_at = time.monotonic()
            task.error = error
            if self._is_tty:
                self._paint()

    def replace_task(self, name: str) -> None:
        """For a single-task phase (cleanup) whose one task changes identity over
        time, e.g. as the cleanup LM falls through its model list."""
        with self._lock:
            self._tasks = {name: StatusTask(name=name, started_at=time.monotonic())}
            if self._is_tty:
                self._paint()

    def note(self, message: str) -> None:
        """Print a permanent line above the status line (or just a line, non-TTY)."""
        with self._lock:
            if self._is_tty:
                sys.stderr.write("\r" + " " * self._last_width + "\r" + message + "\n")
                self._last_width = 0
                self._paint()
            else:
                sys.stderr.write(message + "\n")
            sys.stderr.flush()

    def close(self) -> None:
        self._stop.set()

    def _ensure_thread(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def _worker(self) -> None:
        while not self._stop.wait(self._tick):
            with self._lock:
                if self._phase is not None and self._is_tty:
                    self._frame += 1  # advance the spinner (time-based, not per-event)
                    self._paint()

    def _paint(self, now: float | None = None, *, done: bool = False) -> None:
        # Caller holds the lock.
        if self._phase is None:
            return
        now = time.monotonic() if now is None else now
        spinner = SPINNER_DONE if done else SPINNER_FRAMES[self._frame % len(SPINNER_FRAMES)]
        width = shutil.get_terminal_size((80, 24)).columns
        line = render_status_line(
            self._phase,
            list(self._tasks.values()),
            phase_started=self._phase_started,
            now=now,
            width=width,
            spinner=spinner,
        )
        padding = " " * max(0, self._last_width - len(line))
        self._last_width = len(line)
        sys.stderr.write("\r" + line + padding)
        sys.stderr.flush()

    def _summary(self, now: float) -> str:
        phase_elapsed = format_elapsed(now - self._phase_started)
        tasks = list(self._tasks.values())
        if not tasks:
            return f"{SPINNER_DONE} {self._phase}  {phase_elapsed}"
        items = [
            f"{t.name} {format_elapsed(t.elapsed(now))}" + (" (failed)" if t.error else "")
            for t in tasks
        ]
        return f"{SPINNER_DONE} {self._phase}  " + ", ".join(items) + f"  ({phase_elapsed})"


# One process-wide display owns the stderr status line, so concurrent tasks never
# collide on it. Call sites use the context managers below rather than touching it.
_display = StatusDisplay()


def status_display() -> StatusDisplay:
    return _display


@contextlib.contextmanager
def status_phase(label: str) -> Generator[StatusDisplay]:
    """Run a phase: a labelled stretch of work that owns the status line. Tasks
    opened inside it (status_task) show as concurrent entries on that one line."""
    _display.begin_phase(label)
    try:
        yield _display
    finally:
        _display.end_phase()


@contextlib.contextmanager
def status_task(name: str) -> Generator[None]:
    """Mark a task running for the duration of the block; ✓ on success, ✗ on error.
    Safe to use from worker threads running concurrently within one phase."""
    _display.start_task(name)
    error = False
    try:
        yield
    except BaseException:
        error = True
        raise
    finally:
        _display.finish_task(name, error=error)


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
