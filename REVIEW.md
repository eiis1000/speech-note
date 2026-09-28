# Second correctness pass — 2026-09-28

The earlier review missed failures between layers. It relied too heavily on
existing happy-path tests, checked functions more carefully than the contracts
between them, and ran packaged commands inside a checkout that could shadow the
packaged source. Its completeness claims exceeded that evidence.

This pass reread all 24 application Python files, five Python tools, the shell
test runner, all four pre-existing test files, both Nix definitions, the lockfile,
runtime/example configuration, README files, and synthetic evaluation fixtures.
Historical AUDIT.md was not part of the complete source reread. New tests trace
rejected responses through fallback, persistence, and process exit status.

Corrections, in logical commits:

- `cadfe29`: shared strict response validation across chat, hosted STT, and evals;
  malformed content and embedded provider errors cannot become successful text.
- `e5f4390`: malformed/trailing audit JSON advances fallback; invalid entries
  cannot masquerade as a clean audit; word fragments cannot anchor inside other
  words; exact output-budget equality is allowed; skipped calls clear old history.
- `c285c2b`: separate concurrent diagnostics, atomic text replacement, retained
  evidence after output/export failures, original ZIP provenance, protected input
  paths, fresh export directories, and removal of stale latest recordings.
- `446b557`: replay backpressure, completion after read errors, queued-audio
  recovery, explicit capture-loss failure, unattended download behavior, and
  sequential ownership of an automatically launched local batch server.
- `66cf432`: recovery when the runtime artifact directory fails, atomic WAV
  publication, exclusive recovery recording names, unsupported device-rate
  rejection, and closure of old server log handles before restart.
- Final verification commit: real HTTP/subprocess regression coverage, corrected
  E2E schema/output expectations, explicit ASR choices independent of user config,
  and long-form coverage checks relative to the supplied recording.

Verification: 298 tests run, 289 passed, nine gated model-dependent tests skipped.
The supplied real-speech fixture WAVs are absent. Seven new provider/audit/budget
tests produced nine failing assertions against the pre-fix `7bb0288` source.
The Nix package was built and its CLI tested from outside the checkout against a
local HTTP server: provider-error fallback, exhausted fallback, and output-write
failure all produced the expected files, diagnostics, and exit codes. A synthetic
fast replay through the packaged ffmpeg/VAD path checks the partial final frame.

These checks establish the tested failure contracts. They do not certify that an
LM preserves every semantic detail, prove arbitrary model/hardware behavior, or
guarantee persistence when every recovery destination is unavailable. The package
build does not activate a NixOS generation or replace the system-installed command.

---

# Correctness pass — 2026-09-08

Read all 31 original Python files (application, tests, and eval tools), both Nix
files, the shell test runner, and the runtime/example configuration documentation.
Generated outputs, dependency caches, and secret files were excluded from the
source-code review. Changes preserve the existing pipeline and model prompts.

## Corrections

- **Local cleanup selection:** `build_organizer` treated the bundled GGUF as an
  explicit override, bypassing the preferred model and default RAM refusal. It
  now delegates the automatic choice correctly and avoids downloading the small
  model when the preferred one is usable. Custom API endpoints no longer launch
  an unrelated server on port 8011.
- **CLI contracts:** explicit providers override presets; invalid numeric values
  and empty model/source lists fail clearly; unused hosted ASR backends do not
  require credentials for text-only or `--no-asr` runs.
- **ASR adapters:** faster-whisper receives `--auto-download`; CTC passes its CPU
  device explicitly instead of allowing Transformers to auto-select a GPU;
  Sherpa consumes the final complete window and pads the remaining samples;
  truncated audio-LLM replies are reported as errors.
- **Audio:** short clips produce finite normalization telemetry, the final partial
  block obeys the peak ceiling, and level/peak grids follow the actual sample
  rate. Audio conversion explicitly selects mono 16-bit PCM and excludes video.
- **Capture:** the live worker waits for the final queued segment; replay honors
  `--no-asr`; interruptions and input failures preserve captured audio. Failed
  full-auto captures retain a recovery WAV outside their temporary directory.
- **Output integrity:** concurrent auto-named outputs cannot overwrite one another;
  empty heuristic cleanup fails full-auto with diagnostics; an unannotated run
  removes stale `clean.plain.latest` while keeping archived prose. The plain
  artifact is now gitignored.
- **Response integrity:** malformed and empty chat replies advance the fallback
  chain. Citation checks preserve Unicode words and reject display text that
  reorders or multiplies words from the cited source.
- **Evals:** truncated replies fail scoring, per-request failures do not abort
  the panel, annotation requests retain schema enforcement, source loaders include
  source 10 onward, and case creation preserves numeric references in plain text.
- **Test isolation:** the cleanup-prompt export test now writes into its temporary
  directory; the request-shaping test no longer fetches a live model catalog.

## Verification

- Baseline: 243 tests run, 234 passed and 9 opt-in full-stack tests skipped.
- Final suite: 269 tests run, 260 passed and 9 skipped; 26 new regression tests.
- The first 21 regressions also ran against an isolated copy of the original
  commit, reproducing 32 failing assertions and 15 errors across subtests.
- Real CLI smoke runs passed for help, text cleanup, invalid numeric input,
  empty cleanup, ZIP cleanup, parallel same-stem inputs, and replay capture.
  Fixtures were synthetic; replay used real ffmpeg and webrtcvad, preserving all
  1,920 decoded samples. Logs are under `tmp/review-smoke/`.
- `git diff --check` passed.

Tests used the existing Nix Python 3.13.13 environment. A fresh shell realization
could not complete because uncached dependency fetches returned HTTP 502; no
dependency pins were changed. Explicitly enabling the full-stack suite confirmed
that its recordings and Parakeet model caches are absent. Real GPU inference,
microphone hardware, and hosted model quality were not exercised in this pass.
