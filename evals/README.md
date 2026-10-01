# Prompt evals

Labeled cases for measuring the three model-facing passes — cleanup, the uncertainty
audit, and hosted ASR — against real model panels. Each harness imports the *shipped*
prompt and request machinery from `speech_note`, so what is measured is what runs; the
scoring heuristics live here and never touch product output. The shipped prompts and
model lists are all the way they are because a labeled bake-off said so. This
directory is that discipline made permanent: no prompt, schema, or model-list change
ships without a number.

Layout:

- `cases/` — synthetic labeled cases, committed. Invented text only; nothing from a
  real recording may ever land here (repo rule: no transcript text in git history).
- `local/` — real-recording cases, **gitignored**. Same directory format. The evals
  run at full strength on a machine that has them and at reduced strength elsewhere.
- `out/` — raw model replies from eval runs, gitignored (they quote the cases).

A case is one directory. Which label file it contains decides which harness picks it
up, and a directory may carry more than one:

## `labels.json` — the uncertainty audit (`tools/annotation_eval.py`)

Needs `0N-<name>.txt` (one per ASR source, in collection order) and `clean.txt` (the
cleaned transcript the auditor sees) alongside:

- `expected`: map of region name → list of anchor substrings of `clean.txt`. A good
  auditor must flag every region; a note counts for a region when its located span
  overlaps any of that region's anchors. Anchors can carry surrounding words to pin a
  specific instance of a repeated phrase.
- `optional`: anchors where flagging is defensible either way — neither rewarded nor
  penalised.
- `forbidden`: anchors that must NOT be flagged (a quiet passage only one source
  heard, a disfluency-only difference). Hits are counted as violations.

Reported per model: region recall, spurious / forbidden / displaced / unlocatable
notes, and citations rejected by `verify_notes`.

## `cleanup-labels.json` — the cleanup pass (`tools/cleanup_eval.py`)

Needs the same `0N-<name>.txt` sources; no `clean.txt`, since the cleanup is what is
being produced.

- `must_keep`: phrases a correct UNION reconstruction cannot lose. Case-insensitive
  substring match.
- `must_not`: phrases whose presence means the wrong reading was promoted — the
  confabulation trap (a lone fluent invention beating two corroborating sources).
- `min_ratio` / `max_ratio`: the acceptable band for output words ÷ mean source words
  (default 0.75–1.6). Below the band means content was dropped or summarized; above
  it means padding or degeneration.

Reported per model: must-keep coverage, must-not hits, length ratio and in-band count,
and leftover disfluency residue.

## `asr-labels.json` — hosted recognizers (`tools/asr_eval.py`)

Lives under `local/asr/<name>/` next to a `clip.mp3`, so these cases are always
gitignored — real audio never enters git.

- `anchors`: word sequences the recording genuinely contains, as confirmed by the
  person who made it. Recall over these is the hard-passage score (casefolded
  word-sequence containment, no fuzz).
- `known_names`: proper nouns that really are in the recording, so they are not
  counted against a model as invented.

Reported per model: anchor recall, unique-8-gram ratio (below ~0.95 means degenerate
repetition), words/wpm coverage, and capitalized tokens no other panel model produced
— the invented-proper-noun signal.

## Running

    python tools/annotation_eval.py                    # all cases, default panel
    python tools/cleanup_eval.py --repeats 3
    python tools/asr_eval.py --cases evals/local/asr/jul18
    python tools/annotation_eval.py --cases evals/cases/picnic --models google/gemma-4-31b-it

Every harness takes `--repeats` and reports mean and range across the runs — free
routes vary wildly run to run, and a single sample has fooled us before.
Truncated replies and request failures count as failed runs. The annotation eval
requires schema support, matching the product; it does not retry without the schema.

The two text harnesses (`annotation_eval`, `cleanup_eval`) additionally pin OpenRouter
serving to bf16/fp16/fp8 so repeats compare the same weights (`--any-quant` opts out)
and report the provider that actually answered; `--api-base` points either of them at
a local `llama-server`, which is how local models get numbers instead of reputations.

`tools/make_eval_case.py` turns any `--export-sources` directory into an annotation
case skeleton under `local/` — sources, `clean.txt` with any anchors stripped, and a
`labels.json` whose fields are explained inline. Add `cleanup-labels.json` by hand if
you want the same recording to grade cleanup too.

Requires `OPENROUTER_API_KEY` in the environment (the tools read `.env` themselves if
present).

## Local Phonon-2 evaluation — 2026-10-01

**Decision: worth including as an optional CPU backend; no default change yet.**
Measured on the Ryzen AI 9 HX 370 using Fermion 0.2.4 with its C int8 encoder and
C TDT decoder, against the shipped Sherpa 1.12.38 / Parakeet v2 int8 backend.
Both received identical audio normalized by speech-note. Runs were sequential;
no audio was uploaded. Downloads and normalization are excluded from the times.

| Measurement | Phonon-2, 12 threads | Current Sherpa, 16 threads |
| --- | ---: | ---: |
| 65.2-minute discussion, warm transcription | 42.14–42.19 s (two processes) | 133.19 s |
| Peak process RSS through long recording | 2.57–2.58 GiB | 2.00 GiB |
| Quiet 132-second dictation, labeled phrases recovered | 5/6 | 0/6 |
| Same dictation after denoising, phrases recovered | 5/6 | 4/6 |

Sherpa at a matched 12 threads took **114.70 s** on the long recording, so
Phonon's advantage remains **2.7×** after controlling thread count. The two
Phonon long transcripts were byte-identical. Its 131 contiguous decoded windows
cover the full 3,913.984 seconds; independent beginning/middle/end clips align
with the corresponding full-transcript passages. This checks for gross cutoff,
not arbitrary omissions or word accuracy.

The quiet-clip gap is partly **VAD dropping speech**, not purely a model quality
gap. Bypassing Sherpa's VAD and feeding it Phonon's exact five windows raises its
score from 0/6 to 4/6 on the original clip; Phonon still gets 5/6. This is a
separate failure mode from the earlier cleanup-provider cutoff. No production
VAD threshold or model default was changed by this evaluation.

Costs: initial Phonon loading plus imports took 23.76 s; a new process after CPU
plane-cache creation took 9.75 s. Warm short-clip decoding was 1.4–1.6 s. The
164 MB download becomes about 460 MiB of runtime files after building that cache,
plus the retained archive; it is not a 164 MB RAM model. The public offline CLI
also passed. Integration needs a scoped newer Transformers stack: the current
5.5.4 package lacks `ParakeetForTDT`; this test used 5.17.0 with its matching
tokenizers/safetensors dependencies in an isolated scratch environment.

Scope is two independent recordings, with original/denoised variants of the
labeled one. There is no human reference transcript for the long technical
discussion, so **no full WER or technical-term accuracy claim** is justified.
Keep Whisper as the distinct model family if Phonon is added; shared Parakeet
ancestry is not evidence of independent errors.

Aggregate data and exact versions/hashes: [phonon2-local-20261001.json](phonon2-local-20261001.json).
Private raw results, segments, logs and benchmark scripts are under the ignored
`evals/out/phonon2-local-20261001/`; the prepared audio and isolated runtime remain
under `tmp/phonon-bench/`. The benchmark itself did not change application code
or installed commands.

The subsequent integration adds optional `--asr phonon` in version 2.2.49,
with isolated pinned dependencies and unchanged defaults. The Nix-store CLI
passed offline short and 65-minute MP3 runs from outside the checkout; the long
output was byte-identical to the benchmark's 9,185-word transcript. The packaged
worker decoded all 131 windows in 41.63 s with passive OpenMP waiting, which is
now the adapter's default unless explicitly overridden. Missing-cache full-auto
runs refuse to download or write accepted text without consent. Regression
coverage includes failed workers, malformed results, incomplete window coverage,
truncation, and MP3-container versus normalized-PCM duration differences.
