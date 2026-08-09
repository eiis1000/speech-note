# Prompt evals

Labeled cases for measuring the annotation pass (and, later, cleanup) against real
model panels. The shipped prompts changed once because a bake-off proved a rewrite
better on a labeled case; this directory is that discipline made permanent — no
prompt or schema change ships without a number.

Layout:

- `cases/` — synthetic labeled cases, committed. Invented text only; nothing from a
  real recording may ever land here (repo rule: no transcript text in git history).
- `local/` — real-recording cases, **gitignored**. Same directory format. The eval
  runs at full strength on a machine that has them and at reduced strength elsewhere.
- `out/` — raw model replies from eval runs, gitignored.

Case format (one directory per case):

- `0N-<name>.txt` — one file per ASR source, in collection order.
- `clean.txt` — the cleaned transcript the auditor sees.
- `labels.json`:
  - `expected`: map of region name → list of anchor substrings of `clean.txt`; a
    good auditor must flag each region (a note counts if its located span overlaps
    any anchor). Anchors can carry context to pin a specific instance of a
    repeated phrase.
  - `optional`: anchors where flagging is defensible either way — neither rewarded
    nor penalised.
  - `forbidden`: anchors that must NOT be flagged (quiet passages only one source
    heard, disfluency-only differences). Hits are counted as violations.

Run:

    python tools/annotation_eval.py            # all cases, default model panel
    python tools/annotation_eval.py --cases evals/cases/picnic --models google/gemma-4-31b-it

Requires `OPENROUTER_API_KEY` in the environment (the tool reads `.env` itself if
present). Scoring heuristics (positional displacement etc.) live here, not in the
shipped code, on purpose: heuristics grade models; they must never edit output.
