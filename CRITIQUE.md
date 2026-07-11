# speech_note codebase critique — 2026-07-05

> Status: every item under "Fixed in this pass" is implemented and committed
> (fecdd26..b5028fb, v2.1.9–2.1.17); 143 unit tests pass and the dry-text,
> file (real whisper+sherpa ASR), and replay-capture pipelines were smoke-run
> end to end — the capture rework additionally under a real pty (spinner
> repaints, note interleaving, interactive copy prompt). One extra bug was
> found *by* the smoke run and fixed (item 9).

Full-package read (all 17 modules, ~5,400 lines) after the flat-ASR-collection,
status-display, and cleanup-union rework. Overall shape is good: the layering
(config → cli/resolve → pipeline → asr/organizer → session/artifacts) is real,
provenance travels with data, outcomes are values, and presentation is mostly
separated from compute. The issues below are the places where an abstraction is
missing, half-applied, or left over from the old architecture.

## Fixed in this pass

### 1. `model.py` vs `models.py` is a naming footgun
Domain types live in `model.py`; consent-gated installation lives in `models.py`.
One letter distinguishes "the Transcript dataclass" from "download prompting" —
every future reader will import the wrong one at least once.
**Fix:** rename `models.py` → `install.py` (it is about installing models, and
`require_consent` / `load_or_install` read naturally there).

### 2. There is a Transcriber protocol, but it only exists as folklore
`asr.py:53` says `Transcriber = Any`; `_transcribe()` (asr.py:203) uses
`inspect.signature` to sniff whether a backend accepts `duration_seconds`;
`prepare_sources` and `ensure_loaded` use `getattr(..., callable)` duck-typing;
and `transcribers.py:674` defines an `AsrTranscriber` union that nothing imports
(dead code). Meanwhile all six transcribers already implement the same three
methods, modulo signature drift.
**Fix:** one `Transcriber` base class in `transcribers.py` with a uniform
`transcribe_file(path, language, *, duration_seconds=None)` and no-op
`ensure_downloaded()` / `ensure_loaded()` defaults. Kill the signature sniffing,
the getattr checks, the `Any` alias, and the unused union.

### 3. `PreparedJob` is a positional 4-tuple doing a dataclass's job
`asr.py:266` — call sites read `job[1].device_kind`, `prepared[0][3]`, and
re-unpack `(label, source, transcriber, prep_error)` four times. In
`pipeline.run_final_asr` this forces a redundant `failed_labels` set even though
each job already carries its own error.
**Fix:** frozen dataclass `PreparedSource(label, source, transcriber, error)`.

### 4. `build_organizer` returns a tuple the Organizer already contains
Every entry point does `organizer, supervisor = build_organizer(...)` and then
`finally: if supervisor is not None: supervisor.close()` — four copies of the
same lifecycle boilerplate, for an object that is already `organizer.supervisor`.
**Fix:** return only the `Organizer`, give it `close()`, and let the pipelines
use one `try/finally organizer.close()` (via `contextlib.closing`-style usage).

### 5. The cleanup request is planned three times and deduped three times
`cleanup_request_plan()` (filter + dedupe + full prompt build over possibly
hundreds of KB) runs in `run_cleanup_stage` (gate), again inside
`Organizer._cleanup_with_llm`, and a third time in `write_sources_export`.
`Organizer.cleanup` *also* filters+dedupes before calling `_cleanup_with_llm`,
and `_cleanup_with_llm` recomputes `_reference_words`/minimum-words that the
prompt already derived — the length warning and the prompt's stated expectation
are kept in sync only by parallel code.
**Fix:** `CleanupRequestPlan` carries `reference_words`/`minimum_words`;
`Organizer.cleanup(sources, plan=...)` accepts the plan the pipeline already
computed; the warning check reads the plan instead of re-deriving. (The export
path still recomputes — it runs once, at the end, on a cold path.)

### 6. A missing `whisper-cli` aborts the whole run instead of one source
`WhisperCppTranscriber.__init__` calls `resolve_whisper_cpp_binary`, so
`build_transcribers()` raises before any source runs — a traceback, and the
sherpa/openrouter peers never get to transcribe. That contradicts the design
invariant that each ASR source fails independently.
**Fix:** resolve the binary lazily (first command build), so it surfaces as a
per-source `✗` while the other sources still run.

### 7. Status-line task names can collide
`run_asr_collection` keys tasks by `short_source_name(source)`; two sources of
the same backend (`--asr whisper-cpp:medium,whisper-cpp:large-v3`) share one
name, so the second `start_task` silently replaces the first's finished entry.
**Fix:** uniquify duplicate display names (`whisper`, `whisper 2`) when building
the jobs.

### 8. Leftovers from the primary/secondary era
- `pipeline.run_dry_text_pipeline` still labels its transcript `"primary"`
  (labels are now `asr1..N` / `live` / `extra:<file>` / `user`).
- `model.Transcript` docstring still documents `"primary"`, `"secondary"`.
- `SECONDARY_OVERLAP_SECONDS` is now just the CTC stride → `CTC_STRIDE_SECONDS`.
- `cleanup_sources` guards against re-adding external/user transcripts with an
  `id()` set, but the first list can only contain `asr-final`/`asr-live` kinds —
  the check is dead.
- `capture.run_capture_pipeline` wraps `build_transcribers` in a
  "Loading ASR models" phase, but construction has been lazy/cheap since the
  consent rework — the phase flashes and lies; loading actually happens in
  `run_final_asr`'s prepare/preload.
- `terminal.status_task` is exported and documented but has no callers.
- `resolve_config` duplicates the comma-split of `--organizer-model` in both
  provider branches.
- `review_panels` took a `config` parameter it never used.

### 9. A no-speech skip rendered as ✗ (failed) on the status line
Found during the verification smoke run: sherpa's VAD correctly hears nothing
in a tone-only clip; `AsrOutcome` models that as a skip ("a skip, not a
failure"), but `run_asr_collection` derived the status mark from `not
outcome.ok`, collapsing skip and error into the same ✗.
**Fix:** mark the task failed only when `outcome.error` is set.

### 10. `capture.py` ran a second, private status-line system (round 2)
`CaptureRunner` had its own `\r`-repaint stack — `_print_status`/`_print_event`,
a terminal lock, a cached width, and a 1 Hz timer thread — parallel to the
`StatusDisplay` that owns the stderr status line everywhere else. Two owners of
one terminal line is exactly the bug class the status rearchitecture removed;
the private timer also spewed `\r` lines into piped/`--full-auto` stderr, since
it never checked for a TTY.
**Fix:** capture is one status phase ("Listening"/"Replaying"), the
right-aligned phase total is the recording timer, and live transcript chunks,
audio warnings, and stop messages print above the line via `note()`. Verified
under a pty and piped.

### 11. Two near-identical unique-path helpers in two modules
`pipeline.unique_directory_path` duplicated `naming.unique_output_path`'s
counting loop one module away from it.
**Fix:** moved beside its sibling in `naming.py`.

### 12. `raw_text` carried a dead fallback branch
It looked up the live transcript by label, then fell through to a loop whose
`"asr-live"` case could never fire (every asr-live transcript has label
"live"). Now a straight kind-priority scan: asr-final, asr-live, then
user/external.

## Noted, deliberately not changed

- **`Config` is a 40-field flat dataclass.** Grouping into sub-configs
  (capture/asr/organizer/output) would read better but touches every test and
  call site for zero behavior change; the flat shape is at least honest and
  immutable. Revisit only if the field count keeps growing.
- **`ChatClient.chat` control flow** (four `continue` paths through
  `_advance_or_raise`) is dense but freshly debugged against real OpenRouter
  failure modes (200-with-error-body, empty choices, transient fallthrough);
  restructuring it now is risk without a bug.
- **`raw_text` / `cleanup_sources` / `review_panels` each re-derive** "which
  transcripts matter" with slightly different policies. The differences are
  deliberate (raw fallback vs LM input vs review display), so a shared helper
  would need policy flags — more abstraction, not less.
- **`config.py` mixes static tables with parsing/env-loading/presentation.**
  Splitting it (catalog vs parsing vs display names) is defensible but pure
  churn at this size; the module docstring already draws the line accurately.
- **Cleanup-LM progress:** the deferred "progress indicators rather than just
  times" request is mostly satisfied by the `StatusDisplay` rework (model-named
  task + elapsed + fallback notes). True token-level progress needs a streaming
  request (`stream: true`) and is a feature, not a cleanup — still deferred.
