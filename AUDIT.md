# Audit: hallucination chain, code review, ASR model evaluation

Date: 2026-08-02. Version at audit: 2.1.21.

Test corpus: one 132 s low-SNR spoken-journal recording supplied by the user, plus its
app-generated transcript, run through the real `-fP` pipeline. Deliberately no transcript
text, audio, or input filenames are reproduced in this file.

Nothing in this document has been implemented. Items are ordered by severity.

---

## Part 1 — The hallucination chain

The reported symptom: on a hard-to-hear recording, a whole passage of the cleaned output
was fluent, specific, confident, and entirely fabricated.

The chain was reproduced end to end and has **two independent links**.

### H1. The default `--online-paid` ASR source confabulates fluently on unintelligible audio

`google/gemini-3-flash-preview` (`OPENROUTER_ASR_MODEL`, `config/catalog.py:63`) produced the
fabricated passage **verbatim**. It was then copied into the cleaned output unchanged. The
other three sources (whisper.cpp medium-q8_0, sherpa Parakeet, the app transcript) each had a
*garbled but different* reading of the same seconds; none contained the fabricated content.

Measured over 16 trials (4 Gemini models x 2 gain settings x 2 repeats), plus 32 further
trials for the length statistics below:

- Every Gemini model invents content in that region on **every** call.
- The invention is **different every time** — different names, different relationships,
  different events. It is a sample from a distribution, not a stable mis-decode.
- Changing the input gain changes the invention completely.

This contradicts the load-bearing claim in `catalog.py:58-61`:

> "given the whole file, it annotates non-speech instead of confabulating it (the failure
> mode of per-chunk audio-LLM transcription)"

That was verified (2026-06-28) on a **clear** 44-minute recording. It does not generalize to
low-SNR audio, and it is currently the stated justification for leading the `--online-paid`
ASR collection with this source. The comment should be narrowed to the case it was actually
tested on.

Note the failure is *worse* than a bad local transcript: whisper and Parakeet emit visibly
garbled text on unintelligible audio, which downstream stages and the reader can recognise.
A fluent audio-LLM emits grammatical prose that is indistinguishable from real content.

### H2. The cleanup prompt is structurally required to adopt the fabrication

This is the more important defect, because it applies to **every** source and every model.

`organizer.py` `SYSTEM_PROMPT` and `build_user_prompt` instruct the cleanup LM to reconstruct
the **union** of what the sources heard, and explicitly forbid the inference that would have
caught this:

- "A passage appearing in only one transcript almost always means the other recognizers
  failed to hear it — NOT that it is fake."
- "Use agreement between sources to decide HOW a word was said ... never WHETHER a passage
  exists."
- "A passage carried by a single source SHOULD be kept unless it is clearly recognition
  garbage."

That rule is correct for a **sensitivity** difference — the other sources have *nothing*
there, so they missed faint speech. It is wrong for an **intelligibility** difference — the
other sources have *different garbled text* there, meaning every recognizer heard the same
unintelligible audio and each guessed differently. The prompt has no way to distinguish the
two cases, so it treats the second as the first.

Observed consequence, twice: the cleanup LM kept the single fluent invention and **discarded
the three corroborated readings**. It did not produce a union at all — it produced a
substitution, because fluency was the only signal available to rank the candidates.

A second observed consequence on an earlier run of the same file: the same few seconds of
speech were rendered **twice**, as two consecutive unrelated statements, because two
divergent readings were both "kept".

**Validated fix.** A revised prompt that keeps the sensitivity rule for gaps, adds the
intelligibility rule for disagreement, and permits an `[unclear]` marker:

| prompt | trials | fabrication reproduced | `[unclear]` marks | words |
|---|---|---|---|---|
| current | 2 | **2 / 2** (verbatim) | 0 | 213, 235 |
| revised | 2 | **0 / 2** | 2, 3 | 205, 201 |

The revised prompt also **retained** the corroborated reading that the current prompt had
dropped, at essentially unchanged length — so this is not a content-loss tradeoff. Same
cleanup model (`deepseek/deepseek-v3.2`), same four real source transcripts.

### H3. Uncertainty is never surfaced to the reader

The output is uniformly fluent prose whether the underlying audio was pristine or
unintelligible, and the prompt actively forbids annotation ("MUST NOT write notes about the
transcript"). For a journal the user re-reads weeks later — when they can no longer remember
what they actually said — this is the worst available failure mode: there is no way to tell
which sentences are trustworthy.

The `[unclear]` marker from H2 addresses this at zero cost. Consider it a feature, not a
diagnostic.

### H4. The prompt's length floor pushes toward padding

`build_user_prompt` computes `reference_words` from the **raw** source word counts, which
include disfluencies and (for app-generated transcripts) speaker markers, then tells the model
the result "should land near {target} words or more" while *also* demanding aggressive
disfluency removal. On a filler-heavy recording those two instructions pull in opposite
directions, and "MUST NOT pad with junk" only forbids *junk* — fluent invention satisfies the
letter of the contract.

Honesty about strength of evidence: on this file the floor was **not** the binding constraint
(output 213 words vs a 184-word target), so this is a design smell identified by reading, not
a demonstrated cause. It becomes load-bearing once H2's fix starts legitimately *shortening*
output — a transcript that collapses unintelligible stretches to `[unclear]` should be allowed
to fall below the floor without being flagged. The floor and the H2 fix must land together.

---

## Part 2 — Two measured bugs

### B1. `normalize_pcm_wav` *attenuates* quiet recordings — `audio.py:198`

```python
gain = min(target_rms / stats.rms_abs, peak_ceiling / stats.peak_abs, NORMALIZE_MAX_GAIN)
```

The peak term is computed from the **absolute maximum sample**, so a single transient — a
bump, a click, a chair — vetoes the entire gain stage.

Measured on the test recording:

| term | value |
|---|---|
| RMS-driven gain (wants +14.8 dB) | 5.483 |
| peak-driven gain (one -0.7 dBFS transient) | **0.863** |
| `NORMALIZE_MAX_GAIN` | 12.0 |
| **applied** | **0.863 — the file is made quieter** |
| p99.9 peak instead of absolute max | 3.999 |

Input was RMS -34.8 dBFS with median per-second speech at -38.9 dBFS. The pipeline handed
whisper audio that was *quieter still*, when +12 dB was available.

This defeats gain normalization on exactly the quiet recordings that need it most, and quiet
input is a primary hallucination driver. Fix: derive the peak from a high percentile
(p99.9) and clip the few samples above the ceiling, or use a real limiter.

### B2. whisper.cpp carries decoder context; faster-whisper does not — `transcribers.py`

`FasterWhisperTranscriber.transcribe_file` sets `condition_on_previous_text=False` (line 141).
`WhisperCppTranscriber._command` (line 210) never passes `-mc 0`, so whisper.cpp uses its
default `-mc -1` and carries prior decoded text forward as context — the classic mechanism by
which one confabulation self-reinforces into a sustained narrative.

The two backends disagree on the single most important anti-hallucination setting, and the
**default** backend is the unsafe one.

Measured on the test recording (medium-q8_0, GPU, same audio):

| config | words | change |
|---|---|---|
| current flags | 210 | baseline; contains two invented clauses |
| `-mc 0` | 175 | both invented clauses gone |
| `-mc 0 -sns -nf` | 179 | — |
| `-mc 0 -sns` + p99.9 gain (B1 fix) | 162 | — |

Also available in this whisper.cpp build and entirely unused: `-sns/--suppress-nst`,
`--vad` (built-in silero), `-nf/--no-fallback`, `-et/--entropy-thold`, `-lpt/--logprob-thold`,
`-nth/--no-speech-thold`. `--vad` needs a VAD model file that is not currently fetched, so it
would require an install path; the others are free.

### B3. `ChatClient` hardcodes `temperature: 0.0` with no `top_p` — `organizer.py:337-342`

`mistralai/voxtral-small-24b-2507` rejects the request outright:

```
HTTP 400 "top_p must be 1 when using greedy sampling."
```

Because the OpenRouter **ASR** backend reuses `ChatClient`, no Mistral-hosted audio model can
be used as an ASR source at all. Sending `top_p: 1` alongside `temperature: 0` fixes it and is
a no-op for every other provider.

---

## Part 3 — ASR model evaluation

Same 132 s recording, same system prompt and mp3 encoding the pipeline actually uses,
8 trials per model.

### The direct question: `gemini-3-flash-preview` vs `gemini-3.5-flash-lite`

**`gemini-3.5-flash-lite` is disqualified: it silently truncates.**

| model | price in / out / audio (per M) | words over 8 trials | truncated |
|---|---|---|---|
| `gemini-3-flash-preview` *(current default)* | $0.50 / $3.00 / $1.00 | 263 x6, 267, 281 | 0/8 |
| `gemini-3.5-flash-lite` | $0.30 / $2.50 / $0.30 | **87, 122, 131**, 233, 252, 257, 264, 266 | **3/8** |
| `gemini-3.1-flash-lite` | $0.25 / $1.50 / $0.50 | 267 x7, 269 | 0/8 |
| `gemini-3.6-flash` | $1.50 / $7.50 / $1.50 | 252-277 | 0/8 |

`3.5-flash-lite` is cheaper than the current default on all three axes (3.3x cheaper on audio,
the dominant ASR cost — $0.0019 vs $0.0045 per run here). It is still the wrong choice: in
3 of 8 trials it stopped at 33-50% of the recording with `finish_reason: "stop"`, i.e. it
reports success while dropping half the content. For this pipeline that is the worst
failure mode there is, and no price makes it acceptable.

Note that the current default is *also* stale: it is a **preview** build of generation 3,
while 3.1, 3.5, and 3.6 have all shipped.

### The recommendation: `gemini-3.1-flash-lite`

- 0/8 truncation, and **fully deterministic** (267 words in 7 of 8 trials).
- Cheapest of the four on input, half the current default's output price.
- Its readings of the unintelligible region come out *visibly garbled* rather than as fluent
  fiction — which for this pipeline is a feature: it lets H2's disagreement rule detect the
  region instead of being fooled by it.
- It is already present in `config/display.py MODEL_SHORT_NAMES` as `"gemini-lite"`, but no
  default list references it — so that entry is currently dead code, or a leftover intent.

`gemini-3.6-flash` is the best-behaved overall but costs 3-5x the current default with no
measured reduction in confabulation rate.

### Other speech-to-text models on OpenRouter (25 audio-input models, all tested paths)

| model | result |
|---|---|
| `openai/gpt-audio-mini` | **degenerate loop**: 10,944 words, one clause repeated to the token cap, `finish_reason: "length"`, $0.04/run |
| `mistralai/voxtral-small-24b-2507` | HTTP 400 — see B3; untestable until that is fixed |
| `nvidia/nemotron-3-nano-omni-...:free` | replies that it cannot access the audio; the `input_audio` path does not work |
| `thinkingmachines/inkling-small` | full coverage, cheap ($0.0017), but paraphrases and invents heavily |
| `google/gemini-2.5-flash-lite` | not evaluated; superseded by the 3.x lite line |

No dedicated (non-LLM) transcription model on OpenRouter accepts a 132 s single request, which
is consistent with the existing note in `catalog.py:229-234`.

### A free hallucination detector, from the variance data

`3-flash-preview` and `3.1-flash-lite` are deterministic in *length* yet produced **completely
different content** in the unintelligible region across gain settings, and `3.5-flash-lite`
and `3.6-flash` vary run to run on identical input.

That variance is itself the signal. Two calls to the same audio-LLM on the same audio, diffed:
regions that agree are real speech; regions that disagree are unintelligible. This is a
stronger and cheaper uncertainty estimate than cross-*model* disagreement (which confounds
intelligibility with model quality), and at $0.002-0.004 per extra call it is nearly free.
Worth considering as the mechanism behind H2/H3 rather than relying on the cleanup LM's
judgement alone.

---

## Part 4 — Smaller code observations

Ordered roughly by value. None is urgent.

1. **`--organizer-timeout` is inert below 45 s.** Its default is `12.0`
   (`cli.py:281`), but `request_timeout_seconds` floors at `ORGANIZER_MIN_REQUEST_TIMEOUT = 45`.
   The documented default is never the effective value. Either raise the default to 45 or
   document the floor.

2. **`--online-paid` prints a warning that contradicts its own help text.**
   `pipeline.py:171-176` prints "free remote endpoints may log or train on inputs" whenever the
   provider is not local — including `-P`, whose help text says the paid models are "not
   logged". Gate the wording on which model list is in play.

3. **Stale "secondary" wording.** `transcribers.py:334` labels the CTC download
   "secondary ASR model", left over from the removed primary/secondary model.

4. **Dead `MODEL_SHORT_NAMES` entry** for `google/gemini-3.1-flash-lite` — see Part 3; this is
   worth resolving *by using the model*, not by deleting the line.

5. **`SherpaTranscriber` attribute drift.** `__init__` sets `self.vad = None`, which is never
   used; `ensure_loaded` creates `self._vad_config`, which is never declared in `__init__`.

6. **`effective_model` special case.** `asr.py:224-228` isinstance-checks
   `WhisperCppTranscriber` to get the model that actually ran. A base-class `effective_model`
   property (defaulting to the configured name) would remove the branch — the same cleanup
   already applied when `Transcriber` gained its base class.

7. **`cleanup_request_plan` is built up to three times per run** — once in
   `run_cleanup_stage`, again via `_dedupe_sources` inside `Organizer.cleanup`, and again in
   `write_sources_export` to dump the prompt. Each rebuild formats the full prompt over every
   transcript. Harmless but wasteful, and it leaves room for the exported prompt to drift from
   the one actually sent.

8. **`discover_directory_inputs` (batch mode)** correctly cannot re-ingest its own `.txt` /
   `.json` outputs, but `ARCHIVE_AUDIO_EXTENSIONS` includes `.mp4` and `.webm`, so an unrelated
   video sitting in the directory becomes an input. Probably intended; worth a deliberate
   decision.

### Things deliberately *not* criticised

The architecture is in good shape and most of what is here is right:

- The flat ASR collection with device-grouped concurrency is a genuinely good design, and the
  `net`/`gpu`/`cpu` grouping giving the remote source free overlap is elegant.
- `StatusDisplay` as sole owner of the stderr line, with the non-TTY milestone fallback.
- Consent-gated installation routed through one module.
- `Session` / `ArtifactStore` single-commit-point separation, and "computation never prints;
  reporting never computes".
- The `config/` split into catalog / parsing / display, and `Config.organizer` grouping.
- Length checks as **warnings, never gates** — the right call, and the reason the fabricated
  output was at least written rather than silently discarded.

The hallucination findings above are not architectural failures. They are a wrong empirical
assumption about one model (H1), and a prompt policy that was correct for the two-local-source
case it was written for and became unsafe when a fluent audio-LLM joined the collection (H2).

---

## Suggested order of work

1. B1 (normalizer peak veto) and B2 (`-mc 0`) — small, local, independently verifiable, and
   they reduce the amount of garbage entering the cleanup stage in the first place.
2. H2 + H3 + H4 together — the prompt rewrite, the `[unclear]` marker, and relaxing the length
   floor so an honest short transcript is not flagged. Validated above; must land as one change.
3. Part 3 — repoint `OPENROUTER_ASR_MODEL` to `gemini-3.1-flash-lite`, and narrow the
   over-broad comment at `catalog.py:58-61`.
4. B3 (`top_p`), then Part 4 items as convenient.

---

# Appendix — follow-up round

Three follow-ups: a dedicated-STT bake-off, a normalization redesign, and a design for the
Link 2 problem that keeps the best reading instead of erasing it.

## A1. OpenRouter has a dedicated transcription API — the `catalog.py` premise is stale

`catalog.py:229-234` says the dedicated transcription models "reject a long single request and
must be chunked", which is the entire justification for using an audio-LLM as the network ASR
source. There is now a `POST /api/v1/audio/transcriptions` endpoint (base64 `input_audio` or
multipart) with **13 dedicated STT models**, and it accepted the whole 132 s file in one
request from every one of them.

**The headline result: no dedicated STT model confabulated.** All 12 working models emitted
visibly garbled text in the unintelligible span. None produced a fluent invented narrative —
that behaviour is specific to audio-**LLMs**. And 11 of 12 independently produced the reading
the user confirmed as correct, which the audio-LLM pipeline had replaced with fiction.

Cost and stability for the 132 s file (`n=3` where measured):

| model | cost | words | stability | notes |
|---|---|---|---|---|
| `openai/whisper-large-v3-turbo` | **$0.0015** | 234 | 234/234/234 | deterministic; cheapest |
| `nvidia/parakeet-tdt-0.6b-v3` | $0.0033 | 264 | 264/264/264 | deterministic; v3 of the local v2 |
| `openai/gpt-4o-mini-transcribe` | $0.0033 | 256 | 251-262 | |
| `qwen/qwen3-asr-flash` | $0.0046 | 250 | 250/250/251 | |
| `openai/gpt-4o-transcribe` | $0.0058 | 203 | 212-224 | cleanest reading of the hard span |
| `mistralai/voxtral-mini-transcribe` | $0.0065 | 248 | — | |
| `deepgram/nova-3` | $0.0095 | 185 | — | **dropped** the hard passage entirely |
| `openai/whisper-1` | $0.0133 | 269 | — | |
| `fish-audio/transcribe-1` | $0.0133 | 239 | — | |
| `microsoft/mai-transcribe-1.5` | $0.0133 | 270 | — | |
| `openai/whisper-large-v3` | $0.0033 | 244 then 135 | 135/135/135 | unstable across batches |
| `x-ai/grok-stt-1.0` | $0.0037 | **49** | — | truncates badly |
| `google/chirp-3` | — | — | — | HTTP 400 |

For reference the current default, `google/gemini-3-flash-preview`, costs $0.0045 and invents.

**Recommendation:** replace the `openrouter` audio-LLM ASR backend with the transcription
endpoint, defaulting to `openai/whisper-large-v3-turbo` (3x cheaper than the current default,
deterministic, no confabulation). Because these are so cheap, running two or three as
independent peers costs ~$0.008 total and turns cross-source agreement into a *trustworthy*
signal — which is what Link 2 needs.

### The endpoint also returns the signals Link 2 wants

`response_format: verbose_json` yields per-segment `avg_logprob`, `no_speech_prob`,
`compression_ratio`, `temperature`, plus `timestamp_granularities: ["word"]` for word-level
start/end times.

`avg_logprob` **does** localize the unintelligible span (median -0.552; the worst windows,
-0.552 and -0.607, cover exactly the bad stretch, while the clean opening and closing are
-0.296 and -0.273). But it is emitted per 30 s decode window, not per segment — identical
values repeat across segments from the same window. So it is a coarse "this third of the
recording is unreliable" signal, not a clause-level one. Clause-level localisation needs
cross-source agreement over the word timestamps.

(Incidentally the same response shows a textbook trailing-silence hallucination: a "Thank you."
segment at 131.4-132.1 s.)

## A2. Normalization: single global gain is the wrong shape

Requirement: recordings change level *within* one file — mic in a pocket, headphone swap,
distance changes, varying background noise. No single gain can serve that, and the current code
additionally has the peak-veto bug (B1).

Measured over speech frames only (frames >10 dB above the local noise floor). "spread" is
p90-p10 of per-frame level — lower means more consistent.

Real recording / the same recording with a synthetic 6-span envelope from -26 to +6 dB:

| approach | real: level / spread | swing: level / spread | clipped |
|---|---|---|---|
| raw | -39.3 / 17.6 dB | -50.0 / 29.7 dB | 0 |
| **current code** | -40.6 / 17.6 dB | -48.6 / 29.6 dB | 0 |
| p99.9 peak (naive B1 fix) | -27.3 / 17.6 dB | -31.9 / 29.8 dB | **1283-1311** |
| `loudnorm` (EBU R128) | -23.0 / 15.9 dB | -28.5 / 26.3 dB | 0 |
| `speechnorm` | -25.2 / 20.4 dB | -32.2 / 33.9 dB | 0 |
| `dynaudnorm` | -22.4 / 17.2 dB | -30.8 / 27.2 dB | 0 |
| `compand` | -16.3 / 16.2 dB | -25.1 / 38.2 dB | 1790-4540 |
| `highpass,afftdn,dynaudnorm` | -22.4 / 27.3 dB | -38.2 / 38.8 dB | 9 |
| sliding AGC, global floor | -23.0 / 15.3 dB | -28.8 / 24.6 dB | 0 |
| **two-pass AGC (recommended)** | **-16.0 / 13.3 dB** | **-19.1 / 16.5 dB** | **0** |

Findings:

1. **The current code makes the real recording quieter than raw** (-40.6 vs -39.3 dBFS). B1
   confirmed twice.
2. **The naive p99.9 fix clips ~1300 samples.** It needs a look-ahead limiter, not just a
   different peak estimate. Do not ship the percentile change alone.
3. **Denoising hurt** — `afftdn` roughly doubled the level spread. Don't.
4. **`speechnorm` and `compand` made consistency worse**, despite raising the level.
5. **`loudnorm` is the best one-liner** and a legitimate cheap option.
6. **The winner is a two-pass AGC**: pass 1 levels spans using a *sliding local* noise floor,
   which lets pass 2's silero VAD see even levels and refine. Sliding-window speech-gated
   RMS → target, gain curve smoothed (duck fast / lift slowly) so it never jumps mid-word,
   then a look-ahead limiter instead of letting one transient veto the gain.
7. **A global noise floor is the trap.** Speech in a quiet span falls below the *global* floor
   and gets classified as silence, so it receives no gain — which is why the single-pass
   version only reached -28.8 dBFS on the swing case. Counterintuitively the local-percentile
   energy gate beat silero VAD on its own, because silero still sees the whole file at one gain.

Downstream effect, same ASR both times:

| audio | local whisper (`-mc 0`) | remote whisper-turbo |
|---|---|---|
| swing, current | 150 words | 238 words |
| swing, loudnorm | 153 | 235 |
| swing, 1-pass AGC | 157 | 249 |
| **swing, 2-pass AGC** | **164 (+9%)** | **270 (+13%)** |
| real, current | 175 | 234 |
| **real, 2-pass AGC** | 160 | **248 (+6%)** |

Monotonic recovery on the level-swinging case in both engines, which is the case that matters.
The user-confirmed ground-truth phrase survived in **every** condition, so nothing regressed.

Two honest caveats: word count is a weak proxy for quality (more words can mean more
hallucination — the local-whisper drop on the *real* file is ambiguous and may be less
invention rather than lost content), and **leveling cannot improve SNR**. If the mic was in a
pocket the speech is quiet *and* muffled *and* noise-dominated; normalization gets each span
into the range the model expects, it does not add information.

## A3. Link 2, revisited: keep the best reading, annotate the doubt

The earlier prompt fix was **wrong** in the way that matters: it replaced content the user
confirmed as correct with `[unclear]`. Erasing a correct reading is a worse failure than
keeping an uncertain one. Requirements, per the user:

- Pick the most reasonable reading and keep it inline.
- Annotate that it is uncertain, ideally with the alternatives.
- Do **not** give the cleanup LM interpretive leeway — it summarizes given any.

The core design error is asking one model to do three jobs at once (preserve, clean, and judge
confidence). The judging is what creates the leeway that causes summarizing.

**Separate detection from rendering.** Detection should be mechanical:

- Word-level timestamps are now available from the transcription endpoint; whisper.cpp can emit
  them (`--no-timestamps` is currently forced on); sherpa already has segment times.
- Bin the timeline, compute cross-source token agreement per bin, and mark low-agreement bins.
  With dedicated STT models this is trustworthy, because every source is an honest
  garble-emitter — no source contributes fluent fiction that outranks the others.
- `avg_logprob` per 30 s window is a cheap corroborating signal.

Then rendering has no judgement left to exercise. Two candidate shapes:

**Option A — annotation pass (preferred).** Leave the current cleanup prompt *completely
untouched*, since it is proven not to summarize. Add a second, cheap call that receives the
cleaned text plus the sources and returns **JSON spans only** — never prose:
`[{quote, confidence, alternatives:[...]}]`. Code inserts the markers. This structurally cannot
damage the transcript, because the call that produces prose and the call that judges it are
different calls, and the judge's output is data.

**Option B — mechanical spans in the prompt.** Compute the low-agreement spans first, then tell
the cleanup LM exactly which stretch is doubtful and what each source heard there, with a
bounded instruction: render your single best reading, then append
`[unclear: also heard as "..."]`. The LM never decides *what* is uncertain, only how to word
one specific span.

Both keep the best reading inline. Option A is safer and independently testable; Option B is
one call instead of two. They compose — A's detector can feed B.

Marker format is a UX choice: inline `⟨?|alternative⟩`, a trailing footnote list, or the user's
explicit `[unclear audio. candidates: 1... 2...]`. A trailing footnote block keeps the body
readable, which matters most for a journal.

Note that with dedicated STT sources the problem shrinks a great deal on its own: 11 of 12
models agreed on the passage that the audio-LLM pipeline fabricated over. Fixing the source
(A1) is the highest-leverage change, and it makes the consensus signal that Link 2 needs
actually reliable.

Also: **the H4 length floor must relax** before any of this ships. Annotations add length rather
than removing it, so the pressure is milder than feared, but a transcript that honestly reports
uncertainty must not be flagged "suspiciously short".

---

# Appendix 2 — validation on a 65-minute recording, and what was implemented

Test corpus for this round: a 65.2-minute recording (stereo 48 kHz AAC), plus the earlier
132 s one. Again, no transcript text or filenames appear here.

## B1 confirmed on a second real file

Peak 0.0 dBFS (a transient at full scale), RMS -24.7 dBFS. The old global gain came out at
**0.794** — it attenuated this recording too. That is now two real files out of two.

## Normalization results

The implemented two-pass AGC, run through the real `normalize_pcm_wav`:

| | 65-min file | 132 s file | synthetic -26..+6 dB envelope |
|---|---|---|---|
| speech level before | -23.9 dBFS | -39.3 | -50.0 |
| speech level after | -21.3 | -16.0 | -19.1 |
| **spread before** | **10.3 dB** | 17.6 | 29.7 |
| **spread after** | **7.6 dB** | 13.3 | 16.5 |
| clipped samples | 0 | 0 | 0 |

5.4 s of CPU for 65 minutes of audio, inside the existing "Normalizing audio" phase that
already overlaps the model preload.

Downstream word counts on the 65-minute file (hosted whisper-large-v3-turbo, deterministic):
raw 8182, old normalizer 8130, new AGC 8082 — a 1.2% spread, i.e. **neutral on this file**.
That is the honest result: this recording's level barely varies (10 dB), so there was little
for leveling to fix. The measured benefit is on level-*varying* audio, where the earlier round
showed +9% (local) and +13% (hosted) content recovery. Reported as neutral-here rather than
claimed as a win.

Two implementation bugs found while building it, both now covered by tests:

1. `np.convolve(mode="same")` returns the *kernel* length when the kernel is longer than the
   signal, which desynced the gain curve from the frames on any clip shorter than the window.
2. The gain cap applied *per pass*, so two passes compounded 34 dB into 64 dB — amplifying the
   noise floor of a near-silent input by the same amount. The cap is now cumulative.

## The audio-LLM fails catastrophically on a long recording

`google/gemini-3-flash-preview` on the 65-minute file:

- **18,975 words** — 291 per minute, against a plausible 125 wpm from every dedicated model.
- Of those, **9,596 consecutive repetitions of a single word** — half the output is a
  degenerate loop, and the unique-8-gram ratio is 0.49.
- `finish_reason: "length"`: it burned to the 32k output cap and was cut off.
- 348 seconds, $0.196 — against 16 seconds and $0.043 for hosted whisper-large-v3-turbo.

Combined with the fabrication on unintelligible audio from the previous round, the audio-LLM
route is not viable. It remains reachable via `--asr openrouter`; it is no longer any default.

## Hosted STT: two practical constraints discovered

1. **Base64 JSON does not scale.** A 65-minute mp3 is 17.3 MB, 23.0 MB as base64, and the
   gateway answers **502** after 37 s. 15.7 MB of base64 still passed, so the limit sits
   between. **Multipart form-data works** and carries no encoding overhead — 200 OK in 9 s.
   The implementation uses multipart. (16 kbps opus is another lever: 9.8 MB base64, and it
   scored *more* words than the 17 MB mp3.)
2. **`verbose_json` is whisper-only.** `nvidia/parakeet-tdt-0.6b-v3`,
   `openai/gpt-4o-mini-transcribe` and `qwen/qwen3-asr-flash` all reject it with HTTP 400
   ("does not support response_format"). Plain `json` is the portable request, so that is what
   is sent — which also means segment timings and `avg_logprob` are only available from the
   whisper family, not from every source.

## What was implemented

1. **Two-pass speech AGC** replacing the global peak-vetoed gain (`audio.py`, constants and
   rationale in `config/catalog.py`). Sliding-window leveling over speech frames judged
   against a *local* noise floor, asymmetric gain smoothing, look-ahead limiter, cumulative
   gain cap. `NormalizationResult` now reports speech level and spread before/after rather
   than one gain number.
2. **`openrouter-stt` backend** (`transcribers.py`): dedicated hosted ASR over
   `/audio/transcriptions`, multipart upload, duration-scaled timeout. A 200 response with no
   transcript raises rather than returning `""` (which the collection would otherwise report
   as "no speech detected").
3. **`--online-paid` now runs Whisper and Parakeet as hosted models** instead of locally:
   `whisper-large-v3-turbo` + `parakeet-tdt-0.6b-v3`, both larger than the local checkpoints.
   The audio-LLM is no longer in that default.
4. **Network sources each get their own scheduling group** (`asr.py`). They hold no local
   device, so grouping them all under `"net"` made two hosted calls serialize for no reason.
5. **Option A: the uncertainty annotation pass** (`organizer.py`). After a successful cleanup,
   a second call receives the cleaned text plus the sources and returns **JSON spans only** —
   `{"uncertain": [{"quote", "alternatives"}]}`. Code locates each quote and inserts
   `[unclear audio; also heard as "..." / "..."]` after it. Properties:
   - The cleanup prompt is **completely untouched**, so the summarizing risk is unchanged.
   - The pass emits no prose, so it cannot rewrite, shorten, or reorder anything. The worst
     outcome is zero annotations.
   - A quote that cannot be located verbatim (whitespace differences aside) is **dropped, not
     guessed at** — annotating the wrong span is worse than not annotating.
   - Any failure — bad JSON, HTTP error, prompt too large — is recorded and ignored; the
     transcript is already correct without markers.
   - Skipped when there is only one source (no disagreement to find, so no reason to pay).
   - On by default; `--no-annotate-uncertainty` disables.
6. Diagnostics schema 3 → 4 (annotation fields, new normalization fields). Version 2.2.0.

The earlier concern that the **H4 length floor** must relax before annotation ships does not
apply to Option A: it only *appends* markers, so output never gets shorter and the shortness
check is unaffected. The floor's padding pressure on the cleanup prompt remains an open item.

Tests: 169 passing (was 151), including regression tests for the transient-veto bug, the
level-varying case, the multipart upload, and 13 annotation tests. `pyright`: 0 errors.

## Blocked

The OpenRouter balance fell below the **$0.50 minimum for audio requests** partway through
(HTTP 402), after the gemini long-file run cost $0.196. Consequences:

- `whisper-large-v3-turbo` is validated on both files (deterministic, cheapest, no truncation).
- **`parakeet-tdt-0.6b-v3` is validated only on the 132 s file** (deterministic there, 264
  words, 0/3 truncation). It is the second hosted default on that evidence plus vendor
  diversity, but it has not run against a long recording.
- The annotation pass has not been exercised against a *remote* cleanup model, only locally.

Top up the balance and the remaining validation is one `-P` run.

## Correction to the annotation result above

The annotation pass was then exercised end-to-end against the *bundled local* model, and it
does not work there. `gemma-4-E2B-it-UD-Q4_K_XL` produces the correct JSON shape and then
loops inside the first entry's `alternatives` array until it hits the output cap, so not even
one entry completes. Consequences, all now handled:

- A "salvage" parser recovers the entries that *did* complete from a truncated reply. It
  cannot help this particular failure (nothing completed) but it is correct and kept, since a
  capable model that overruns mid-list would otherwise have its whole reply discarded.
- The pass now distinguishes a well-formed empty result ("nothing to flag") from an
  unparseable one, and records the latter. Without that, a model that never emits valid JSON
  looked exactly like a clean recording — which is how this was found in the first place.
- The default is therefore **provider-dependent**: annotation is ON for remote cleanup
  providers and OFF for the local one, so the extra call is not spent where it cannot succeed.
  `--annotate-uncertainty` / `--no-annotate-uncertainty` overrides either way.

Note the local run's own output contains a badly garbled clause that *should* have been
flagged — so the feature is warranted; the 2B model simply cannot perform the audit. It
remains unvalidated against a remote cleanup model because of the balance blocker below.

Tests: 175 passing. `pyright`: 0 errors.

---

# Appendix 3 — corrections after review

Four things were wrong in the previous round. Each is corrected below with the reasoning
that replaced it.

## A3.1 The speech gate was wrong, and is gone

The old normalizer classified frames as speech or not and derived the gain only from the
"speech" ones. That is backwards for a case that happens constantly: **speech is often
quieter than the noise around it** — a speaker walking into a shower, a phone in a pocket
— and it is still perfectly recoverable. A gate in that situation selects the *noise* as
its reference and levels to that, leaving the real speech further below target than
before. Nothing about the audio is knowable well enough to justify classifying it.

The replacement has no gate, no classification, and never removes anything:

1. **Level estimate**: a high percentile (p75) of frame loudness in a long *centred*
   window (90 s). A percentile needs no gate — it is already robust to pauses (which drag
   a mean down, so the gain overshoots) and to clicks (which set a maximum).
2. **A slow ride**: two successive centred moving averages (45 s each, i.e. a triangular
   kernel). A level change must persist on that timescale before the gain follows it.
3. **Zero-phase**: both stages are centred, computed over the whole file. A causal filter
   lags the audio it describes, so it turns the gain up *after* the quiet part has gone.
4. **Peaks are compressed, never used to reduce the overall gain**: a look-ahead soft-knee
   compressor (threshold -14 dBFS, 4:1, 20 ms look-ahead) with a safety limiter behind it.

Measured against the requirements:

| requirement | result |
|---|---|
| quiet recording is boosted, never attenuated | 132 s file x7.68; 65 min file x2.05 (old code: x0.86 and x0.79) |
| a few minutes of quiet becomes normal | 3-minute quiet stretch: **+19.8 dB** of ride |
| a brief dip is ridden through | 5-second dip: **+2.2 dB**, i.e. ignored |
| the gain does not lag | ride is already +3.6 dB up **10 s before** the quiet stretch starts |
| one transient cannot veto the gain | file with a full-scale click still gains x7.67 |
| nothing is cut when speech is under the noise | waveform correlation input/output **0.998** |

Spread (p90-p10 of frame loudness) on the 65-minute file: **12.2 -> 6.4 dB**. 3.4 s of CPU
for 65 minutes.

## A3.2 Multipart does not split the recording

To answer the question directly: `multipart/form-data` is only the HTTP *transfer*
encoding. The recording is sent whole, as one body part, in one request, and the server
transcribes the entire file. It is the same whole-file semantics as base64-in-JSON, minus
the 33% encoding inflation that pushed a 65-minute file past the gateway limit. Nothing is
chunked, split, or stitched anywhere in this pipeline.

## A3.3 The audio-LLM is back in the paid defaults

Removing it from `--online-paid` overreached: the instruction was to keep the option, not
to demote it. `ONLINE_PAID_ASR_SOURCES` is now the two dedicated hosted recognizers **plus**
the Gemini audio-LLM, listed last so it is never the raw/fallback transcript.

This is also a better design than it first looked. The audio-LLM is the most sensitive
source on faint speech, and its inventions disagree with both dedicated recognizers — which
is precisely the signal the annotation pass keys on. A fabrication from it now gets flagged
rather than silently believed.

Worth recording honestly: on the 65-minute file this round it did **not** degenerate
(10,652 words, 163 wpm, no loop), where a previous run produced 18,975 words containing
9,596 repetitions of one word. So that failure is intermittent, not deterministic.

## A3.4 Truncated annotation replies: fixed at the source, and nothing is dropped

`_salvage_entries` is deleted. Reconstructing meaning from a half-written reply was
treating a symptom.

**The root cause was asking for JSON instead of requiring it.** The annotation request now
carries a `response_format` JSON **schema**. For a local model this is not advisory:
llama.cpp compiles the schema to a GBNF grammar and constrains sampling with it, so an
unterminated string is not a reachable state. `maxLength`/`maxItems` in the schema bound the
worst-case reply, so "ran out of budget" stops being possible rather than being something to
recover from. `finish_reason == "length"` is now reported as an error instead of salvaged.

Result: the bundled gemma-E2B quant, which previously looped inside the first entry until
the token cap and produced nothing usable, now returns well-formed answers — 5 notes on the
132 s file, no error.

**Unlocatable quotes are no longer dropped.** A note whose quote cannot be found verbatim
(or which overlaps an already-marked span) is listed in a trailing section under
"Uncertain passages (the recording does not clearly support these):". Marking a guessed span
would be worse than a separate list, but discarding the auditor's finding is worse than
either.

One further quality problem surfaced once gemma could answer at all: it returned notes whose
"alternative" was the quote verbatim (`"I did."` offered as an alternative reading of
`"I did."`), and alternatives lifted from an unrelated part of the recording. The first is now
filtered in code — alternatives that do not differ from the quote, or from each other, are
dropped, and a note left with none is dropped with it. The second is a judgement failure no
filter can catch, which is why the default remains provider-dependent: **on for remote
models, off for the bundled local one**. The reason is now capability, not format.

Annotation also moved to its own module, and `organizer.py` was split by concern:

| module | concern |
|---|---|
| `chat.py` | HTTP transport: auth, model fallback, `response_format`, `finish_reason` |
| `llama_server.py` | the optional local llama-server child process and its GGUF |
| `organizer.py` | cleanup only: prompt, plan, length sanity check |
| `annotate.py` | the uncertainty pass that runs after cleanup |

## A3.5 OpenAI's own transcription models

Tested directly against `api.openai.com/v1/audio/transcriptions`.

| model | 132 s file | duration limit | rate consistency |
|---|---|---|---|
| `gpt-4o-transcribe` | 237 words, 6.5 s | **hard cap 1400 s (23.3 min)**, explicit error | **degrades with length**: 127 wpm at 5 min, 117 at 10 min, **87 at 20 min** |
| `gpt-transcribe` | 237 words, 4.9 s | works to **50 min**; 60 min rejected as "corrupted or unsupported" | **stable**: 138/131/137/135/138/130 wpm at 5/10/20/30/40/50 min |

`gpt-transcribe` ($0.0045/min, so ~$0.29 for 65 minutes) is the better of the two by a wide
margin: consistent rate, no degeneration, 6x the duration headroom. But it still cannot take
a 65-minute file, and it is ~7x the price of hosted `whisper-large-v3-turbo` ($0.043 for the
same recording, no duration limit encountered). `gpt-4o-transcribe`'s silent thinning with
length is disqualifying on its own — 87 wpm where every other model reads 125-145.

Neither is adopted as a default. Both are reachable through the OpenRouter STT backend by
model name.

## A3.6 The new OpenRouter key cannot reach audio models

The key added to `.env` as `OPENROUTER_API_KEY` returns HTTP 404 for every audio model:
`"No endpoints available matching your guardrail restrictions and data policy"`. Its account
has a privacy/data-policy setting that excludes all audio providers. `OPENROUTER_API_KEY_EIIS`
in the same file (identical to the key in `~/.config/speech-note/env`) works.

Fix at <https://openrouter.ai/settings/privacy> for that account, or leave the working key
where `load_user_env` finds it. Nothing in the code needs to change; the failure path behaved
correctly — all three sources failed, each error was reported, and the run exited 1 without
crashing.

## End-to-end validation

`-P` on the 65-minute recording, everything on:

- normalization: ride +1.6..+11.0 dB, spread 12.2 -> 6.4 dB, 0 clipped samples
- three hosted ASR sources, all concurrent: whisper-turbo 18.8 s, parakeet-v3 15.2 s,
  gemini 104.9 s, ASR phase total 1:44 — i.e. the max, so the per-source scheduling groups
  work
- 8017 / 9426 / 10652 words respectively
- cleanup: deepseek-v3.2, 294 s, 7882 words
- **8 uncertainty annotations, 0 in the appendix, no error**

Tests: 180 passing. `pyright`: 0 errors.

Still open from Part 4: the inert `--organizer-timeout` default, and the `-P` warning that
still says "free remote endpoints may log or train on inputs" when the paid models are not.
