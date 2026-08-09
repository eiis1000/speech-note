#!/usr/bin/env python3
"""Measure the SHIPPED annotation prompt against labeled cases on a live model panel.

Everything prompt-shaped is imported from speech_note.annotate — the eval exercises
exactly what ships, so the two cannot drift. What lives here instead is scoring:
labeled-region recall, forbidden-region violations, and the positional displacement
heuristic. Heuristics grade models; they must never edit product output, which is
why this is a tool and not part of the pipeline.

Usage:
    python tools/annotation_eval.py                      # every case, default panel
    python tools/annotation_eval.py --cases evals/cases/picnic --models google/gemma-4-31b-it

Cases come from evals/cases (committed, synthetic) plus evals/local (gitignored,
real recordings) when present. See evals/README.md for the case format. Raw replies
are saved under evals/out/<run>/ so a run can be re-scored without re-requesting.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from speech_note.annotate import (  # noqa: E402
    EXAMPLE_ASSISTANT,
    EXAMPLE_USER,
    SYSTEM_PROMPT,
    build_prompt,
    decode_response,
    notes_cap,
    response_format,
    verify_notes,
)
from speech_note.model import Transcript  # noqa: E402

URL = "https://openrouter.ai/api/v1/chat/completions"

# Free where the free route is usable; gemma-4-31b's free route 429s through every
# retry window, so it runs paid (identical weights, ~2k-token prompts, well under a
# cent per call).
DEFAULT_MODELS = [
    "google/gemma-4-31b-it",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "openai/gpt-oss-20b:free",
    "nvidia/nemotron-nano-9b-v2:free",
]

# gpt-oss rejects reasoning:{effort:"none"} ("Reasoning is mandatory"); give it low
# effort and room for the hidden tokens instead.
REASONING_MANDATORY = {"openai/gpt-oss-20b:free"}


def load_env_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    env_file = REPO / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("OPENROUTER_API_KEY not set and not found in .env")


# ------------------------------------------------------------------------- cases


class Case:
    def __init__(self, path: Path) -> None:
        self.name = path.name
        self.clean = (path / "clean.txt").read_text().strip()
        self.sources = [
            Transcript(
                label=f"asr{i}",
                model=p.stem.split("-", 1)[1],
                kind="asr-final",
                text=p.read_text().strip(),
            )
            for i, p in enumerate(sorted(path.glob("0*.txt")), start=1)
        ]
        labels = json.loads((path / "labels.json").read_text())
        self.expected: dict[str, list[str]] = labels.get("expected", {})
        self.optional: list[str] = labels.get("optional", [])
        self.forbidden: list[str] = labels.get("forbidden", [])


def discover_cases(paths: list[str] | None) -> list[Case]:
    if paths:
        return [Case(Path(p)) for p in paths]
    dirs: list[Path] = []
    for root in (REPO / "evals" / "cases", REPO / "evals" / "local"):
        if root.is_dir():
            dirs += sorted(d for d in root.iterdir() if (d / "labels.json").exists())
    if not dirs:
        raise SystemExit("no cases found under evals/")
    return [Case(d) for d in dirs]


# ----------------------------------------------------------------------- scoring


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


STOP = set(words("the a an and um uh i it is was that to of for you we like so but yeah in on"))


def locate(text: str, quote: str) -> tuple[int, int] | None:
    index = text.find(quote)
    if index != -1:
        return index, index + len(quote)
    parts = quote.split()
    if not parts:
        return None
    match = re.search(r"\s+".join(re.escape(w) for w in parts), text, re.IGNORECASE)
    return (match.start(), match.end()) if match else None


def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def best_source_position(alternative: str, sources: list[Transcript]) -> tuple[float, float]:
    """(relative position of best-matching window, overlap fraction) across sources.

    Known limitation: a source with partial coverage (e.g. a recognizer that gave up a
    third of the way in) distorts relative position — text at 0.9 of a short source may
    sit at 0.3 of the recording — so a DISPLACED flag against such a source can be a
    false positive. Read flags against the fullest source before believing them."""
    target = [w for w in words(alternative) if w not in STOP]
    if not target:
        return -1.0, 0.0
    want = set(target)
    best = (0.0, -1.0)
    for source in sources:
        tokens = words(source.text)
        if not tokens:
            continue
        width = max(len(target), 4)
        for start in range(0, max(1, len(tokens) - width + 1)):
            overlap = len(want & set(tokens[start : start + width])) / len(want)
            if overlap > best[0]:
                best = (overlap, start / max(1, len(tokens)))
    return best[1], best[0]


def note_alternative_texts(note) -> list[str]:
    """Alternative display texts, tolerant of strings and Citation objects."""
    out = []
    for alt in note.alternatives:
        text = alt if isinstance(alt, str) else alt.display()
        if text:
            out.append(str(text))
    return out


def score_case(case: Case, notes) -> dict:
    expected_spans = {
        region: [span for anchor in anchors if (span := locate(case.clean, anchor))]
        for region, anchors in case.expected.items()
    }
    optional_spans = [span for a in case.optional if (span := locate(case.clean, a))]
    forbidden_spans = [span for a in case.forbidden if (span := locate(case.clean, a))]

    # Placement must mirror the product: the shipped locator uses the note's
    # before/after context to pin the right instance of a repeated phrase, and
    # scoring it with a naive first-match find() would fake a miss.
    from speech_note.annotate import _locate as shipped_locate

    hit: set[str] = set()
    result = {
        "notes": len(notes),
        "unlocatable": 0,
        "spurious": 0,
        "forbidden": 0,
        "displaced": 0,
        "detail": [],
    }
    for note in notes:
        span = shipped_locate(case.clean, note) if hasattr(note, "before") else locate(
            case.clean, note.quote
        )
        if span is None:
            result["unlocatable"] += 1
            result["detail"].append(f'    UNLOCATABLE  "{note.quote[:70]}"')
            continue
        regions = [r for r, spans in expected_spans.items() if any(overlaps(span, s) for s in spans)]
        hit.update(regions)
        tag = ",".join(regions) if regions else "optional"
        if not regions and not any(overlaps(span, s) for s in optional_spans):
            if any(overlaps(span, s) for s in forbidden_spans):
                result["forbidden"] += 1
                tag = "FORBIDDEN"
            else:
                result["spurious"] += 1
                tag = "SPURIOUS"
        result["detail"].append(f'    [{tag}] "{note.quote[:60]}"')
        for alt in note_alternative_texts(note):
            alt_rel, overlap_frac = best_source_position(alt, case.sources)
            delta = abs(alt_rel - span[0] / max(1, len(case.clean)))
            bad = overlap_frac >= 0.5 and delta > 0.25
            if bad:
                result["displaced"] += 1
            suffix = f" [DISPLACED d={delta:.2f}]" if bad else ""
            result["detail"].append(f'          -> "{alt[:60]}"{suffix}')
    result["recall"] = f"{len(hit)}/{len(case.expected)}"
    return result


# ----------------------------------------------------------------------- requests


def ask(
    key: str, model: str, messages: list[dict], fmt: dict, out_file: Path
) -> tuple[list, str]:
    body: dict = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 12288 if model in REASONING_MANDATORY else 4096,
        # The same length-scaled schema the product sends, not a static default.
        "response_format": fmt,
        "reasoning": {"effort": "low" if model in REASONING_MANDATORY else "none"},
    }
    for attempt in range(6):
        response = requests.post(
            URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=body,
            timeout=300,
        )
        if response.status_code == 429:
            time.sleep(30 * (attempt + 1))
            continue
        if response.status_code == 400 and "response_format" in body:
            # Provider rejects json_schema: measure whether the prompt alone holds
            # the format — the situation the product is in on such providers.
            body.pop("response_format")
            continue
        if not response.ok:
            return [], f"HTTP {response.status_code} {' '.join(response.text.split())[:90]}"
        payload = response.json()
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(payload, indent=2))
        choice = (payload.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content") or ""
        notes, understood = decode_response(content)
        if not understood:
            finish = choice.get("finish_reason")
            return [], f"NOT the requested JSON (finish={finish}): {' '.join(content.split())[:90]}"
        return notes, ""
    return [], "429 after retries"


def verified(notes, case: Case) -> tuple[list, int]:
    """Apply the shipped citation check, exactly as the product does."""
    return verify_notes(notes, case.sources)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="*", help="case directories (default: all under evals/)")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--run-name", default="run")
    args = parser.parse_args()

    key = load_env_key()
    cases = discover_cases(args.cases)
    out_root = REPO / "evals" / "out" / args.run_name

    for case in cases:
        print(f"\n{'=' * 78}\nCASE {case.name}  ({len(case.sources)} sources)\n{'=' * 78}")
        messages_tail = build_prompt(case.clean, case.sources)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": EXAMPLE_USER},
            {"role": "assistant", "content": EXAMPLE_ASSISTANT},
            {"role": "user", "content": messages_tail},
        ]

        def one(model: str):
            slug = model.replace("/", "_").replace(":", "_")
            fmt = response_format(notes_cap(case.clean))
            notes, error = ask(key, model, messages, fmt, out_root / case.name / f"{slug}.json")
            return model, notes, error

        with ThreadPoolExecutor(max_workers=len(args.models)) as pool:
            for model, notes, error in pool.map(one, args.models):
                short = model.split("/")[-1].removesuffix(":free")
                if error:
                    print(f"  {short:30s} FAILED  {error}")
                    continue
                kept, rejected = verified(notes, case)
                scored = score_case(case, kept)
                print(
                    f"  {short:30s} recall={scored['recall']} notes={scored['notes']:2d} "
                    f"rejected={rejected} spurious={scored['spurious']} "
                    f"forbidden={scored['forbidden']} displaced={scored['displaced']} "
                    f"unlocatable={scored['unlocatable']}"
                )
                for line in scored["detail"]:
                    print(line)


if __name__ == "__main__":
    main()
