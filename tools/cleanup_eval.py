#!/usr/bin/env python3
"""Measure the SHIPPED cleanup prompt against labeled cases on a live model panel.

The counterpart of tools/annotation_eval.py for the first pass: requests are built by
speech_note.organizer's own cleanup_messages(), so what is measured is exactly what
ships. Scoring is mechanical:

  keeps    - must_keep phrases present (a correct UNION reconstruction cannot lose them)
  bad      - must_not phrases present (the cleanup promoted the wrong reading)
  ratio    - output words / mean source words, bounded by [min_ratio, max_ratio]
             (below = content dropped or summarized; above = padding/degeneration)
  residue  - disfluencies left in ("um", "uh", "you know", "i mean")

Cases are directories under evals/cases (committed, synthetic) or evals/local
(gitignored, real recordings) that contain a cleanup-labels.json next to the NN-*.txt
sources. Raw outputs are saved under evals/out/<run>/ for reading — the numbers rank,
the text decides.

Usage:
    python tools/cleanup_eval.py --repeats 3
    python tools/cleanup_eval.py --models deepseek/deepseek-v3.2 google/gemini-3-flash-preview
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from speech_note.model import Transcript  # noqa: E402
from speech_note.organizer import (  # noqa: E402
    _reference_words,
    cleanup_messages,
    cleanup_request_plan,
)

DEFAULT_URL = "https://openrouter.ai/api/v1/chat/completions"

# The incumbent lead first; the rest are the current cheap frontier worth auditioning.
DEFAULT_MODELS = [
    "deepseek/deepseek-v3.2",
    "google/gemini-3-flash-preview",
    "qwen/qwen3.6-35b-a3b",
    "moonshotai/kimi-k2.5",
    "z-ai/glm-5-flash",
    "openai/gpt-5-nano",
    "anthropic/claude-haiku-4.5",
    "google/gemma-4-26b-a4b-it",
]

DISFLUENCIES = re.compile(r"\b(um+|uh+|you know|i mean)\b", re.IGNORECASE)


def load_env_key(api_base: str) -> str:
    if "openrouter" not in api_base:
        return "unused-local"
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    env_file = REPO / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip("'\"")
    raise SystemExit("OPENROUTER_API_KEY not set and not found in .env")


class Case:
    def __init__(self, path: Path) -> None:
        self.name = path.name
        self.sources = [
            Transcript(
                label=f"asr{i}",
                model=p.stem.split("-", 1)[1],
                kind="asr-final",
                text=p.read_text().strip(),
            )
            for i, p in enumerate(sorted(path.glob("[0-9]*-*.txt")), start=1)
        ]
        labels = json.loads((path / "cleanup-labels.json").read_text())
        self.must_keep: list[str] = labels.get("must_keep", [])
        self.must_not: list[str] = labels.get("must_not", [])
        self.min_ratio: float = labels.get("min_ratio", 0.75)
        self.max_ratio: float = labels.get("max_ratio", 1.6)


def discover_cases(paths: list[str] | None) -> list[Case]:
    if paths:
        return [Case(Path(p)) for p in paths]
    dirs: list[Path] = []
    for root in (REPO / "evals" / "cases", REPO / "evals" / "local"):
        if root.is_dir():
            dirs += sorted(d for d in root.iterdir() if (d / "cleanup-labels.json").exists())
    if not dirs:
        raise SystemExit("no cleanup cases found under evals/ (need cleanup-labels.json)")
    return [Case(d) for d in dirs]


def score(case: Case, text: str, reference_words: int) -> dict:
    lowered = text.lower()
    words = len(text.split())
    ratio = words / max(1, reference_words)
    return {
        "words": words,
        "ratio": ratio,
        "keeps": sum(1 for phrase in case.must_keep if phrase.lower() in lowered),
        "bad": sum(1 for phrase in case.must_not if phrase.lower() in lowered),
        "residue": len(DISFLUENCIES.findall(text)),
        "in_band": case.min_ratio <= ratio <= case.max_ratio,
    }


def ask(
    key: str, model: str, messages: list[dict], url: str, out_file: Path,
    any_quant: bool = False,
) -> tuple[str, str, str]:
    body: dict = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 8192,
        "reasoning": {"effort": "none"},
    }
    if "openrouter" in url and not any_quant:
        # Routes silently swap providers and quantizations between runs; pin to
        # high-precision serving so repeats compare the same thing.
        body["provider"] = {"quantizations": ["bf16", "fp16", "fp8"]}
    unpinned = False

    def post() -> requests.Response:
        return requests.post(
            url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=body,
            timeout=600,
        )

    response = post()
    if response.status_code == 404 and "quantization" in response.text and "provider" in body:
        # Closed-weight models expose no quantization metadata, so the precision
        # filter excludes every endpoint. Retry unpinned and say so in the provider
        # column, so mixed-precision serving stays visible.
        body.pop("provider")
        unpinned = True
        response = post()
    if not response.ok:
        return "", f"HTTP {response.status_code} {' '.join(response.text.split())[:90]}", ""
    payload = response.json()
    served_by = str(payload.get("provider") or "?") + (" (unpinned)" if unpinned else "")
    choice = (payload.get("choices") or [{}])[0]
    content = ((choice.get("message") or {}).get("content") or "").strip()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(content + "\n")
    if choice.get("finish_reason") == "length":
        return "", "truncated reply (output token limit)", served_by
    if not content:
        return "", f"empty reply (finish={choice.get('finish_reason')})", served_by
    return content, "", served_by


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="*")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--run-name", default="cleanup")
    parser.add_argument("--api-base", default=DEFAULT_URL)
    parser.add_argument(
        "--any-quant",
        action="store_true",
        help="do not pin OpenRouter serving to bf16/fp16/fp8",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    key = load_env_key(args.api_base)
    out_root = REPO / "evals" / "out" / args.run_name

    for case in discover_cases(args.cases):
        reference = _reference_words(case.sources)
        plan = cleanup_request_plan(case.sources, context_tokens=65_536, max_output_tokens=16_384)
        messages = cleanup_messages(plan)
        print(f"\n{'=' * 78}\nCASE {case.name}  (reference {reference} words)\n{'=' * 78}")

        def one(job: tuple[str, int]):
            model, repeat = job
            slug = model.replace("/", "_").replace(":", "_")
            try:
                text, error, served_by = ask(
                    key, model, messages, args.api_base,
                    out_root / case.name / f"{slug}.{repeat}.txt",
                    any_quant=args.any_quant,
                )
            except (requests.RequestException, ValueError, AttributeError, TypeError) as exc:
                text, error, served_by = "", str(exc), ""
            return model, repeat, text, error, served_by

        jobs = [(model, repeat) for model in args.models for repeat in range(args.repeats)]
        by_model: dict[str, list] = {model: [] for model in args.models}
        providers: dict[str, list[str]] = {model: [] for model in args.models}
        with ThreadPoolExecutor(max_workers=min(12, len(jobs))) as pool:
            for model, repeat, text, error, served_by in pool.map(one, jobs):
                by_model[model].append((repeat, text, error))
                if served_by:
                    providers[model].append(served_by)

        for model in args.models:
            short = model.split("/")[-1].removesuffix(":free")
            runs, failures = [], 0
            for _repeat, text, error in sorted(by_model[model]):
                if error:
                    failures += 1
                    continue
                runs.append(score(case, text, reference))
            if not runs:
                print(f"  {short:28s} all runs FAILED "
                      f"({by_model[model][0][2] if by_model[model] else '?'})")
                continue

            def agg(field: str, fmt: str = ".1f") -> str:
                values = [r[field] for r in runs]
                mean = sum(values) / len(values)
                if min(values) == max(values):
                    return f"{mean:{fmt}}"
                return f"{mean:{fmt}} [{min(values):{fmt}}-{max(values):{fmt}}]"

            bands = sum(1 for r in runs if r["in_band"])
            fail_note = f" +{failures}fail" if failures else ""
            served = ",".join(sorted(set(providers[model]))) or "?"
            print(
                f"  {short:28s} n={len(runs)}{fail_note} keeps={agg('keeps')}/{len(case.must_keep)} "
                f"bad={agg('bad')} ratio={agg('ratio', '.2f')} in-band={bands}/{len(runs)} "
                f"residue={agg('residue')} via={served}"
            )


if __name__ == "__main__":
    main()
