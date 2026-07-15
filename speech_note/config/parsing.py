"""Parsing of user-supplied config: --asr source specs and the two user files.

The catalog holds the static tables; this module turns text a user wrote (CLI
tokens, the ASR config file, the env file) into those types / into os.environ.
"""

from __future__ import annotations

import os
from pathlib import Path

from .catalog import ASR_BACKENDS, AsrSource

__all__ = [
    "parse_asr_source",
    "parse_asr_sources",
    "USER_ASR_FILE",
    "load_user_asr_sources",
    "USER_ENV_FILE",
    "load_user_env",
]


def parse_asr_source(token: str) -> AsrSource:
    """Parse one ``backend[:model][@device]`` spec into a resolved AsrSource."""
    token = token.strip()
    if not token:
        raise ValueError("empty ASR source spec")
    rest, _, device = token.partition("@")
    backend, _, model = rest.partition(":")
    backend = backend.strip()
    if backend not in ASR_BACKENDS:
        raise ValueError(
            f"unknown ASR backend {backend!r}; choose from {', '.join(sorted(ASR_BACKENDS))}"
        )
    return AsrSource(backend, model.strip(), device.strip() or "auto").resolved()


def parse_asr_sources(tokens: list[str]) -> tuple[AsrSource, ...]:
    """Parse repeatable --asr values (each itself comma-separated) into a collection."""
    sources: list[AsrSource] = []
    for raw in tokens:
        for piece in raw.split(","):
            piece = piece.strip()
            if piece:
                sources.append(parse_asr_source(piece))
    return tuple(sources)


# A user config file listing the default ASR collection, one `backend[:model][@device]`
# spec per line (blank lines and # comments ignored). Mirrors USER_ENV_FILE: it lets a
# user change the default collection without editing source. CLI --asr overrides it.
USER_ASR_FILE = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "speech-note" / "asr"


def load_user_asr_sources(path: Path | None = None) -> tuple[AsrSource, ...]:
    """Read the ASR collection from USER_ASR_FILE; empty tuple if absent/empty."""
    path = path or USER_ASR_FILE
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return ()
    tokens = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return parse_asr_sources(tokens)


# A KEY=VALUE file (e.g. OPENROUTER_API_KEY=...) read at startup so secrets travel
# with speech-note regardless of the working directory or shell — kept out of the
# Nix store / git, unlike a direnv .envrc which only loads inside the repo tree.
USER_ENV_FILE = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "speech-note" / "env"


def load_user_env(path: Path | None = None) -> None:
    """Populate os.environ from USER_ENV_FILE without overriding the live shell.

    setdefault, not assignment: an explicitly exported variable always wins, so this
    is a fallback for shells that don't have the key, never an override.
    """
    path = path or USER_ENV_FILE
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            os.environ.setdefault(key, value)
