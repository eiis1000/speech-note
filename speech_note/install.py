"""Consent-gated model installation.

One place that decides whether to download a missing model. Consent rules:
  * ``--auto-download`` (auto_yes=True): assume yes, download without prompting.
  * an interactive terminal: prompt once per model ("download ~SIZE into DIR?").
  * non-interactive without --auto-download: refuse, and the caller raises a clear,
    actionable error naming the model and its target directory.

Each model loader checks presence itself, then routes a miss through here, so the
download UX is identical across whisper.cpp, sherpa-onnx, and the HF backends.
Adding a model: detect "missing", then call ``require_consent`` before fetching.
"""

from __future__ import annotations

import sys
from typing import Callable, TypeVar

T = TypeVar("T")


class ModelInstallDeclined(RuntimeError):
    """Raised when a required model is absent and the user declined to install it."""


def _interactive() -> bool:
    try:
        return bool(sys.stdin and sys.stdin.isatty() and sys.stderr and sys.stderr.isatty())
    except Exception:
        return False


def consent_to_download(
    label: str,
    dest: str,
    *,
    size_hint: str | None = None,
    auto_yes: bool,
    interactive: bool | None = None,
) -> bool:
    """Return True if we may download ``label`` into ``dest``.

    auto_yes short-circuits to True. Otherwise an interactive terminal is prompted
    (default No); a non-interactive caller gets False.
    """
    if auto_yes:
        return True
    if interactive is None:
        interactive = _interactive()
    if not interactive:
        return False
    size = f" (~{size_hint})" if size_hint else ""
    print(
        f"\n{label} is not installed.\n  download{size} into: {dest}\nProceed? [y/N] ",
        end="",
        file=sys.stderr,
        flush=True,
    )
    try:
        answer = input().strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes")


def require_consent(
    label: str,
    dest: str,
    *,
    size_hint: str | None = None,
    auto_yes: bool,
    interactive: bool | None = None,
) -> None:
    """Prompt for consent; raise ModelInstallDeclined with an actionable message if denied."""
    if consent_to_download(
        label, dest, size_hint=size_hint, auto_yes=auto_yes, interactive=interactive
    ):
        return
    raise ModelInstallDeclined(
        f"{label} is not installed and the download was declined. Install it under "
        f"{dest}, or re-run with --auto-download to fetch it non-interactively."
    )


def load_or_install(
    loader: Callable[[bool], T],
    *,
    label: str,
    dest: str,
    size_hint: str | None = None,
    auto_yes: bool,
    interactive: bool | None = None,
) -> T:
    """Load a Hugging Face model offline; on a cache miss, ask consent then download.

    ``loader`` takes one argument, ``local_files_only``: it is called first with
    True (no network), and only if that fails — a cache miss — is the user asked to
    consent to a download, after which it is called again with False.
    """
    try:
        return loader(True)
    except ModelInstallDeclined:
        raise
    except Exception:
        require_consent(label, dest, size_hint=size_hint, auto_yes=auto_yes, interactive=interactive)
        return loader(False)
