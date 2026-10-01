"""Private JSON worker for the pinned, isolated Phonon runtime."""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
from pathlib import Path
import sys


def run(operation: str, audio: str | None) -> dict:
    os.environ["FERMION_DEVICE"] = "cpu"
    cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    os.environ.setdefault("FERMION_CACHE_DIR", str(cache / "fermion"))
    if operation != "download":
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from fermion._catalog import SPEECH_PROFILES
    from fermion._speech.fetch import ensure, is_model_dir, profile_dir

    pin = SPEECH_PROFILES["five-value"]
    path = profile_dir(pin["repo"], pin["unpack_dir"])
    if operation == "download":
        path = ensure(pin["repo"], "five-value", pin)
    ready = is_model_dir(path) and (path / "model.fermion").is_file()
    if operation != "transcribe":
        return {"ready": ready, "path": str(path)}
    if not ready:
        raise RuntimeError(f"Phonon 2 model missing or incomplete: {path}")
    from fermion._speech.engine_phonon2_cpu import load

    model = load(path, profile="five-value", backend="phonon2-five-value")
    result = dataclasses.asdict(model.transcribe_detailed(audio))
    result["runtime"] = model.describe()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("check", "download", "transcribe"))
    parser.add_argument("audio", nargs="?")
    args = parser.parse_args()
    if args.operation == "transcribe" and not args.audio:
        parser.error("transcribe requires an audio path")
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = run(args.operation, args.audio)
        print(json.dumps(result, allow_nan=False))
    except Exception as exc:
        print(f"Phonon 2: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
