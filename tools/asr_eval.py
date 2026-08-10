#!/usr/bin/env python3
"""Measure hosted transcription models on labeled hard clips.

Two endpoint kinds, matching the product's two ASR backends:
  stt    — dedicated transcription (OpenRouter /audio/transcriptions, multipart)
  audio  — audio-LLM via chat with input_audio (the product's "openrouter" backend,
           same system prompt)

Metrics per model:
  anchors   — labeled word sequences the recording genuinely contains (recall of
              hard passages; casefolded word-sequence containment, no fuzz)
  uniq8     — unique-8-gram ratio; below ~0.95 means degenerate repetition
  names     — capitalized tokens not in known_names and not produced by any other
              panel model: likely invented proper nouns (the gemini failure class)
  words/wpm — coverage and rate sanity

Cases live in evals/local/asr/<name>/ (gitignored — real audio never enters git):
clip.mp3 + asr-labels.json. Raw transcripts land in evals/out/<run>/ for reading.

Usage:
    python tools/asr_eval.py --repeats 2
    python tools/asr_eval.py --cases evals/local/asr/jul18 --models stt:openai/whisper-large-v3-turbo
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from speech_note.config import OPENROUTER_STT_API_BASE  # noqa: E402
from speech_note.transcribers import OPENROUTER_ASR_SYSTEM_PROMPT  # noqa: E402

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

# kind:model. The current -P lineup first, then challengers.
DEFAULT_MODELS = [
    "stt:openai/whisper-large-v3-turbo",
    "stt:nvidia/parakeet-tdt-0.6b-v3",
    "audio:google/gemini-3-flash-preview",
    "stt:openai/whisper-large-v3",
    "stt:mistralai/voxtral-mini-transcribe",
    "stt:microsoft/mai-transcribe-1.5",
    "stt:qwen/qwen3-asr-flash-2026-02-10",
    "stt:deepgram/nova-3",
    "audio:openai/gpt-audio-mini",
    "audio:google/gemini-3.6-flash",
]


def load_env_key() -> str:
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
        self.clip = path / "clip.mp3"
        labels = json.loads((path / "asr-labels.json").read_text())
        self.anchors: list[str] = labels.get("anchors", [])
        self.known_names: set[str] = {n.casefold() for n in labels.get("known_names", [])}
        import subprocess

        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(self.clip)],
            capture_output=True, text=True,
        )
        self.duration = float(probe.stdout.strip() or 0)


def discover_cases(paths: list[str] | None) -> list[Case]:
    if paths:
        return [Case(Path(p)) for p in paths]
    root = REPO / "evals" / "local" / "asr"
    dirs = sorted(d for d in root.iterdir() if (d / "asr-labels.json").exists()) if root.is_dir() else []
    if not dirs:
        raise SystemExit("no ASR cases under evals/local/asr/")
    return [Case(d) for d in dirs]


def word_key(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.casefold()))


def transcribe_stt(key: str, model: str, clip: Path) -> tuple[str, str]:
    with clip.open("rb") as handle:
        response = requests.post(
            OPENROUTER_STT_API_BASE,
            headers={"Authorization": f"Bearer {key}"},
            files={"file": ("audio.mp3", handle, "audio/mpeg")},
            data={"model": model, "response_format": "json"},
            timeout=600,
        )
    if not response.ok:
        return "", f"HTTP {response.status_code} {' '.join(response.text.split())[:80]}"
    text = response.json().get("text") or ""
    return text, "" if text else "empty transcript"


def transcribe_audio_llm(key: str, model: str, clip: Path) -> tuple[str, str]:
    encoded = base64.b64encode(clip.read_bytes()).decode()
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": OPENROUTER_ASR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": encoded, "format": "mp3"},
                    }
                ],
            },
        ],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 16384,
    }
    response = requests.post(
        CHAT_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=body,
        timeout=900,
    )
    if not response.ok:
        return "", f"HTTP {response.status_code} {' '.join(response.text.split())[:80]}"
    content = ((response.json().get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    return content, "" if content else "empty reply"


def uniq_ngram_ratio(text: str, n: int = 8) -> float:
    words = word_key(text).split()
    if len(words) < n:
        return 1.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return len(set(grams)) / len(grams)


def capitalized_tokens(text: str) -> set[str]:
    # Mid-sentence capitalized words, crude proper-noun proxy. Sentence starts are
    # excluded by requiring the preceding character to not be terminal punctuation.
    names = set()
    for match in re.finditer(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{2,})\b", text):
        names.add(match.group(1).casefold())
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="*")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--run-name", default="asr")
    args = parser.parse_args()

    key = load_env_key()
    out_root = REPO / "evals" / "out" / args.run_name

    for case in discover_cases(args.cases):
        print(f"\n{'=' * 78}\nCASE {case.name}  ({case.duration:.0f}s)\n{'=' * 78}")

        def one(job: tuple[str, int]):
            spec, repeat = job
            kind, model = spec.split(":", 1)
            fn = transcribe_stt if kind == "stt" else transcribe_audio_llm
            text, error = fn(key, model, case.clip)
            slug = model.replace("/", "_")
            out = out_root / case.name / f"{slug}.{repeat}.txt"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text + "\n")
            return spec, repeat, text, error

        jobs = [(spec, r) for spec in args.models for r in range(args.repeats)]
        results: dict[str, list[str]] = {spec: [] for spec in args.models}
        errors: dict[str, list[str]] = {spec: [] for spec in args.models}
        with ThreadPoolExecutor(max_workers=min(10, len(jobs))) as pool:
            for spec, _repeat, text, error in pool.map(one, jobs):
                (errors[spec] if error else results[spec]).append(error or text)

        # Names produced by >=2 panel models are probably real; a name unique to one
        # model (and not in known_names) is the invention signal. A token whose
        # LOWERCASE form appears in other models' output is an ordinary word that one
        # model happened to capitalize, not a name (measured: "trigger", "largely").
        name_counts: dict[str, int] = {}
        lowercase_all: dict[str, set[str]] = {}
        for spec, texts in results.items():
            model_names = set()
            lowercase_all[spec] = set()
            for text in texts:
                model_names |= capitalized_tokens(text)
                lowercase_all[spec] |= set(word_key(text).split())
            for name in model_names:
                name_counts[name] = name_counts.get(name, 0) + 1

        for spec in args.models:
            short = spec.split(":", 1)[1].split("/")[-1]
            texts = results[spec]
            if not texts:
                print(f"  {short:32s} FAILED  {errors[spec][0] if errors[spec] else '?'}")
                continue
            joined_keys = [f" {word_key(t)} " for t in texts]
            anchor_hits = sum(
                1 for anchor in case.anchors
                if all(f" {word_key(anchor)} " in k for k in joined_keys)
            )
            words = sum(len(t.split()) for t in texts) // len(texts)
            wpm = words / max(case.duration / 60, 0.01)
            uniq = min(uniq_ngram_ratio(t) for t in texts)
            others_words = set().union(
                *(words for other, words in lowercase_all.items() if other != spec)
            ) if len(args.models) > 1 else set()
            invented = sorted(
                name for name in (capitalized_tokens(" ".join(texts)) - case.known_names)
                if name_counts.get(name, 0) <= 1 and name not in others_words
            )
            fail_note = f" +{len(errors[spec])}fail" if errors[spec] else ""
            print(
                f"  {short:32s} n={len(texts)}{fail_note} anchors={anchor_hits}/{len(case.anchors)} "
                f"words={words} wpm={wpm:.0f} uniq8={uniq:.3f} "
                f"invented-names={','.join(invented) if invented else '-'}"
            )


if __name__ == "__main__":
    main()
