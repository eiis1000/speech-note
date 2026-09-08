#!/usr/bin/env python3
"""Turn one run's --export-sources directory into an annotation-eval case skeleton.

The eval (tools/annotation_eval.py) gets its authority from labeled cases built out
of real recordings that annoyed you. This makes creating one a two-step chore:

    speech-note ... --export-sources /tmp/export
    python tools/make_eval_case.py /tmp/export --name shower-monologue

then open evals/local/shower-monologue/labels.json and fill in the labels (the
skeleton explains each field). evals/local/ is gitignored — real transcript text
never enters history.

The case copies the source transcripts and the cleaned text; anchors and a trailing
notes list are stripped from clean.txt if present (exports predating 2.2.9 baked the
annotated text in).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

SKELETON = {
    "_comment": [
        "Fill these in, then run:  python tools/annotation_eval.py --cases <this dir>",
        "expected:  map of region-name -> list of anchor substrings of clean.txt.",
        "           A good auditor MUST flag each region; a note counts for a region",
        "           when its located span overlaps an anchor. If the doubtful phrase",
        "           repeats, include enough surrounding words to pin the instance.",
        "optional:  anchors where flagging is defensible either way - not scored.",
        "forbidden: anchors that must NOT be flagged (quiet passages only one source",
        "           heard, disfluency-only differences). Hits count as violations.",
    ],
    "expected": {},
    "optional": [],
    "forbidden": [],
}


def strip_annotations(text: str) -> str:
    """Plain prose from a possibly-annotated transcript: drop the notes list and the
    [n] anchors. Harmless on already-plain text."""
    parts = re.split(r"\n\nUnclear passages:\n", text, maxsplit=1)
    if len(parts) == 1:
        return text.rstrip() + "\n"
    body, notes = parts
    for number in re.findall(r"^\[(\d+)\]", notes, re.MULTILINE):
        body = body.replace(f"[{number}]", "")
    return body.rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_dir", type=Path, help="an --export-sources directory")
    parser.add_argument("--name", help="case name (default: the export directory's name)")
    parser.add_argument(
        "--dest",
        type=Path,
        default=REPO / "evals" / "local",
        help="cases root (default: evals/local — gitignored, for real recordings)",
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing case")
    args = parser.parse_args()

    sources = sorted(args.export_dir.glob("[0-9][0-9]-*.txt"))
    clean = args.export_dir / "clean.txt"
    if not sources:
        sys.exit(f"no NN-*.txt source transcripts in {args.export_dir}")
    if not clean.exists():
        sys.exit(f"no clean.txt in {args.export_dir} (the cleanup must have produced output)")

    case = args.dest / (args.name or args.export_dir.name)
    if case.exists() and not args.force:
        sys.exit(f"{case} already exists (use --force to overwrite)")
    case.mkdir(parents=True, exist_ok=True)

    for source in sources:
        shutil.copyfile(source, case / source.name)
    (case / "clean.txt").write_text(strip_annotations(clean.read_text()), encoding="utf-8")
    labels = case / "labels.json"
    if not labels.exists() or args.force:
        labels.write_text(json.dumps(SKELETON, indent=2) + "\n", encoding="utf-8")

    print(f"case written: {case}")
    print(f"now label it: $EDITOR {labels}")
    print(f"then run:     python tools/annotation_eval.py --cases {case}")


if __name__ == "__main__":
    main()
