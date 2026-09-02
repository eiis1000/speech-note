# speech-text

A vibe-coded local-first system for high-quality microphone dictation and audio-file transcription.

The pipeline for every run:

1. capture or load audio
2. probe its duration once, then level it with a slow time-varying gain ride
   (a 3-minute quiet stretch gets boosted; a 5-second dip barely moves; peaks
   are handled by a look-ahead compressor, never by cutting the overall gain)
3. transcribe with a collection of ASR sources, running sources on different
   devices concurrently (the default collection is whisper.cpp on the GPU plus
   sherpa-onnx Parakeet on the CPU, which don't contend; `--online-paid` swaps
   in three hosted recognizers)
4. clean the transcript with an LM that sees every source named, with its
   model and any reliability note
5. audit the cleaned transcript with a second LM pass that marks the passages
   the sources do not jointly support — `[n]` anchors in the text, one
   "Unclear passages" list at the end with the competing readings, each
   verified verbatim against the source transcripts before it is shown
6. commit all artifacts once: `raw.latest`, `clean.latest`, `clean.plain.latest`
   (the pre-annotation prose, when the audit ran), `recording.latest.wav` (live
   capture only), `diagnostics.latest.json`, and a timestamped copy of each
   under `logs/`

**A note for other users:** I highly recommend changing the settings to suit your
hardware, and in particular if you have an OpenRouter API key and are not concerned
with privacy **I recommend using the OpenRouter organizer path** (`-P`) rather than
the mildly-weak gemma-4-E2B default. For a stronger *local* default, drop the QAT
gemma-4-26B-A4B GGUF into `~/.cache/huggingface/gguf/` — it is picked up
automatically when it fits in RAM (see [Cleanup LM](#cleanup-lm)).

Code layout: the `speech_note/` package is the program (`cli` → `pipeline` /
`capture` → `transcribers` / `asr` → `organizer` / `annotate`, with `session`
holding the run record). Supporting modules: `config/` (all defaults, the
backend tables, and the ASR-spec/env parsing), `chat` (the OpenAI-compatible
HTTP transport and model fallback chain), `llama_server` (the optional local
llama-server child process, its RAM guard, and GGUF install), `model` (the
`Transcript` domain type), `install` (consent-gated model download), `hardware`
(GPU detection), `audio` / `devices` / `terminal` / `naming` / `textproc`.
`tools/` has the whisper.cpp Vulkan probe, the no-nix suite runner
(`run_tests_uv.sh`), and the eval harnesses (see [Evals](#evals)); `tests/` the
test suites; `evals/` the labeled eval cases.

Extending it (where new code goes):

- **an ASR backend** — add a `BackendSpec` to the `ASR_BACKENDS` registry in
  `config/catalog.py`, then either a `*Transcriber` class in `transcribers.py`
  (in-process, wired in `asr.build_transcriber`) or, for an external CLI, a
  command in `asr.build_subprocess_command` (with `in_process=False` in the
  registry).
- **a cleanup provider** — extend `organizer.py` (`build_organizer`) and
  `chat.py` (`ChatClient`).
- **a model that needs downloading** — detect "missing", then route through
  `install.require_consent` / `install.load_or_install` so it shares the install UX.

## Installation

This is a Linux + [Nix](https://nixos.org/download) project (flakes enabled). The
flake pins the entire toolchain — the Python environment, `whisper.cpp` (Vulkan),
`llama.cpp` (Vulkan), CrispASR, sherpa-onnx, and PyTorch (ROCm) — so there is no
separate `pip install`:

```sh
git clone <repo-url> speech-text
cd speech-text
nix develop          # builds/fetches the toolchain, drops you in a dev shell
speech-note --help
```

(With [direnv](https://direnv.net), `cp .envrc.example .envrc && direnv allow`
enters the shell automatically.)

The GPU paths are built for an **AMD** iGPU here (Vulkan for Whisper/llama, ROCm
for PyTorch). They run on CPU anywhere, and NVIDIA users should read
[Hardware notes and other GPUs](#hardware-notes-and-other-gpus) before first run.

Models are fetched on first use, not bundled: the first run that needs a missing
model prompts before downloading it (or pass `--auto-download` to skip the
prompts) — including the default cleanup GGUF. See "Models and network" below for
the full policy, and [Cleanup LM](#cleanup-lm) for using a different one.

Inside the shell, `speech-note` runs the Nix-store copy of the code — edits to
the working tree are picked up by `python -m speech_note` (or by re-entering
the shell). Tests: `python -m unittest discover -s tests` (fast logic suite), or
`python tests/test_speech_note.py` for that file alone. When the Nix cache is
unreachable, `bash tools/run_tests_uv.sh` runs the same suite from PyPI wheels —
it needs only `numpy` and `requests`, because `sounddevice`/PortAudio and
`webrtcvad` are imported lazily and the few tests that need them self-skip.
The full-stack no-mock suite — real Whisper, real Parakeet ASR, real local
cleanup LM — runs on demand and self-skips any stage whose model *or* test
fixture is missing. It needs two recordings you supply: `tests/fixtures/short.wav`
(~15 s) and `tests/fixtures/long.wav` (~50 s; looped past the 400 s mark to
exercise long-form transcription). With those in place:
`SPEECH_NOTE_E2E=1 python -m unittest discover -s tests -p 'test_e2e.py'`.

Discovery commands:

```sh
speech-note --help
speech-note --list-input-devices
```

Models and network: nothing is bundled — models download on first use. When a
required model (Whisper ggml, a Parakeet ASR model, etc.) is missing, an
interactive run **asks first** — "*&lt;model&gt; is not installed. download ~&lt;size&gt;
into &lt;dir&gt;? [y/N]*" — and only fetches it if you agree. `--auto-download` answers
yes to all (for scripts/`--full-auto`); a non-interactive run without it fails
with a message naming the model and where to put it. This includes the default
cleanup GGUF (fetched from `config.DEFAULT_GGUF_REPO`); if its download fails or
is declined, place a GGUF at `config.DEFAULT_GGUF_MODEL` yourself, point
`--organizer-gguf` elsewhere, or use `--online-free` / `--online-paid`.
After the first fetch everything runs offline.

## Terminal dictation

```sh
speech-note
```

- prompts for the cleanup model and a microphone (`Enter` accepts defaults)
- prints a live preview transcript while recording (always a small resident
  faster-whisper model, regardless of the final backend)
- warms the final Whisper model and the cleanup LM in the background while you
  speak
- `Enter` (or Ctrl-C) stops; Ctrl-C also works during the final ASR/cleanup
  stages
- shows the outputs side by side and asks what to copy
  (`1`/`2`/`3` = shown outputs, `n` or `Enter` = nothing). Non-interactive runs
  (piped, or `--full-auto`) skip this and copy nothing.

The live preview is *not* fed to the cleanup LM as an equal source; it is used
only as a fallback when the final Whisper pass fails.

## Audio-file pipeline

```sh
speech-note --input /path/to/audio.m4a
```

Unattended runs that should leave only a cleaned transcript in the current
directory:

```sh
speech-note --input /path/to/audio.m4a --full-auto
```

`--full-auto` writes an auto-named `<input>-clean.txt` only if cleanup actually
succeeded (failed, empty, or truncated cleanup writes nothing and the process
exits nonzero), keeps the usual artifacts in a temporary directory, and on any
error drops a matching `<input>-clean-diagnostics.json` next to the output.

Pass several inputs after one `--input` (or repeat the flag), or pass a directory
to process each top-level audio/zip as an independent full-auto job. Batch outputs
land beside their inputs; subdirectories and previous `.txt`/`.json` outputs are
not ingested:

```sh
speech-note -fPi first.zip second.m4a
speech-note -fPi /path/to/inbox
speech-note -fFi /path/to/inbox --parallel
```

`-P` includes `--parallel` in its preset. Every other mode is sequential by
default—even if its current backends happen to be remote—and must opt in with
`--parallel`. `--no-parallel` overrides `-P`. At most four items run at once:
each one is a whole pipeline, and levelling a single 65-minute recording holds
about 1.5 GB, so an unbounded fan-out over a directory of long files exhausts
RAM rather than going faster. `--parallel-workers N` raises the ceiling when the
inputs are short or the machine is large. Failures remain per-item: the rest of
the batch completes, and the command exits nonzero if any item failed.

Add external transcripts (Google Recorder etc.) as extra sources for cleanup —
`.srt`/`.vtt` files are parsed down to their text, `.txt` and `.json` are
passed through:

```sh
speech-note --input rec.m4a --extra-transcript rec.google.txt
```

A zip export containing one recording plus transcript files:

```sh
speech-note --input /path/to/export.zip
```

Export every transcript that was fed to the cleanup LM (each ASR pass and each
`--extra-transcript`), one file per source plus the cleaned `clean.txt`, so you
can compare what each source heard or reuse a single source later as an
`--extra-transcript`:

```sh
speech-note --input rec.m4a --export-sources ./rec-sources/
# -> rec-sources/01-<source>.txt, 02-<source>.txt, ..., clean.txt
```

Cleanup-only modes:

```sh
# existing transcripts, no audio (all transcripts are equal peers):
speech-note -x rec.whisper.txt -x rec.parakeet.txt

# have the audio but reuse existing transcripts instead of re-running ASR:
speech-note -i rec.m4a --no-asr -x rec.whisper.txt

# raw text, no files ("||" marks chunk boundaries):
speech-note --input-text 'um this is bad || and should be cleaned'
```

Replay an audio file through the live-capture path (timing/debugging):

```sh
speech-note --replay-input-file rec.m4a --replay-speed 1.0
```

## ASR

A run transcribes the audio with an ordered **collection** of ASR sources and
hands every transcript to the cleanup LM as a peer. Two independent engines that
make different errors give the cleanup LM the cross-checks it needs to tell a real
word from an ASR artifact. There is no "primary"/"secondary" — order is only a
soft preference for which transcript is surfaced as the raw fallback when cleanup
is off or fails.

The default collection is GPU Whisper plus CPU Parakeet:

| source | model | runs on |
|---|---|---|
| `whisper-cpp` | `medium-q8_0` (ggml) | GPU (Vulkan) |
| `sherpa` | `csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8` | CPU |

They use different devices, so they run concurrently and the pair finishes in
roughly the time of the slower one. Sources that share a device run sequentially
within their device group.

### Choosing the collection

`--asr` adds a source. It is repeatable, and each value may be a comma-separated
list. A source is `backend[:model][@device]`; the model and device are optional
and fall back to the backend's defaults:

```sh
speech-note --input rec.m4a --asr whisper-cpp:small.en@gpu --asr sherpa
speech-note --input rec.m4a --asr 'whisper-cpp:medium-q8_0,ctc@cpu'
speech-note --input rec.m4a --asr sherpa --organizer-mode off   # one source, raw
```

With no `--asr`, the collection comes from the user config file
`~/.config/speech-note/asr` if present (one `backend[:model][@device]` per line,
`#` comments allowed), otherwise the built-in default above.

Available backends:

| backend | default model | runs on |
|---|---|---|
| `whisper-cpp` | `medium-q8_0` (ggml) | GPU (Vulkan); `@cpu` or a GPU index |
| `faster-whisper` | `Systran/faster-whisper-medium.en` | CPU |
| `sherpa` | `csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8` | CPU |
| `ctc` | `nvidia/parakeet-ctc-0.6b` | CPU (`@cuda:0` to force GPU) |
| `onnx` | `nemo-parakeet-tdt-0.6b-v2` int8 | CPU |
| `crispasr` | `parakeet-tdt-0.6b-v2-q4_k.gguf` | GPU (Vulkan) |
| `pocketsphinx` | bundled en-us | CPU |
| `openrouter-stt` | `openai/whisper-large-v3-turbo` | hosted (OpenRouter `/audio/transcriptions`; whole recording in one multipart request; needs `OPENROUTER_API_KEY`) |
| `openrouter` | `google/gemini-3-flash-preview` | hosted (audio-LLM via chat `input_audio`; see the confabulation caveat below) |

`--online-paid` swaps the default collection to three hosted `openrouter-stt`
recognizers — whisper-large-v3-turbo, parakeet-tdt-0.6b-v3, and
mai-transcribe-1.5 — larger checkpoints than fit locally, from three different
vendors so their errors don't correlate, and a 65-minute recording transcribes
in ~16 s for a few cents. Measured on a hard low-SNR recording, every working
dedicated recognizer emitted visibly garbled text where the audio was
unintelligible and *not one invented a narrative* — fluent confabulation is an
audio-LLM behaviour. The `openrouter` audio-LLM backend stays selectable
(`--asr openrouter`) for its faint-speech sensitivity, but it is no default:
on long files it degenerates into repetition loops, and on hard audio it has
replaced garble with fiction. The uncertainty-annotation pass is what makes
using it survivable.

Each source's model is fetched on first use (consent-gated; `--auto-download` to
skip the prompt). Other knobs: `--whisper-cpp-model /path/model.bin` points
whisper-cpp sources at an explicit ggml file (diagnostics record the file that
actually ran); `--asr-cpu-threads` sets the CPU thread budget;
`--no-strip-asr-timestamps` keeps Parakeet `[hh:mm:ss]` markers;
`python tools/whisper_cpp_probe.py` checks whether whisper-cli sees Vulkan.

### Backend notes

- **`sherpa`** decodes the Parakeet-TDT-0.6B-v2 transducer in C++ at ~48–70x
  realtime on the CPU, with punctuation and casing. It has no length limit: silero
  VAD segments the audio (each segment capped at `SHERPA_MAX_SEGMENT_SECONDS`, 20s)
  so the FastConformer encoder never sees the long sequence that OOMs a single
  full-attention pass. Decoding is sequential, not batched (on the CPU, batching
  only adds padding waste — it is a GPU occupancy lever). The model bundle and the
  silero VAD file are ~650 MB total.
- **`onnx`** runs the same Parakeet-TDT model through onnx-asr's Python decode
  loop — a portable fallback, ~3x slower than `sherpa` on the CPU. **`crispasr`**
  runs Parakeet-TDT on the GPU (Vulkan).
- **`ctc`** has no length limit: audio past Parakeet CTC's ~400s position-embedding
  cliff is transcribed by the HF pipeline's native long-form striding (240s windows
  with a 5s stride), windowing the audio and stitching at the logit level. The
  transformers Parakeet-CTC port omits `config.inputs_to_logits_ratio`, which the
  pipeline needs to align strides (without it only the first window survives —
  ~755 words for a 600s clip vs ~1840); the backend sets it from the encoder
  subsampling factor x hop length (8 x 160 = 1280).
- A source that finds no speech (e.g. its VAD saw none) is reported as a note,
  separately from errors, in both the terminal output and diagnostics.

### `large-v3-turbo-q5_k` — available, but needs a strong cleanup LM

`--asr whisper-cpp:large-v3-turbo-q5_k` is offered as a Whisper model option, but
**with a caveat**: on long recordings it tends to *hallucinate repeats* —
degenerate loops where a word or phrase is emitted many times in a row ("and and
and …", "you can believe that it's an action." ×8). It is also no faster than
`medium-q8_0` on an iGPU (turbo prunes only the decoder; the encoder — the
bottleneck here — is unchanged), so on this hardware there is little reason to
prefer it; it is wired up for machines where it might pay off.

The cleanup LM is meant to remove these loops: another source in the collection
won't corroborate the repetition, the prompt tells the model to drop uncorroborated
repeats, and this model's transcript carries an explicit warning about its looping.
**The bundled gemma-4-E2B is too small to do this reliably** — it strips the worst
loops but lets others through. So pair `large-v3-turbo-q5_k` with a stronger cleanup
model: `--organizer-gguf /path/to/bigger.gguf` for a larger local model (see
[Cleanup LM](#cleanup-lm)), or `--online-paid` / `--online-free` for OpenRouter.

## Cleanup LM

`--organizer-mode llama` (default) sends all sources to an OpenAI-compatible
endpoint; `heuristic` is a cheap regex cleanup; `off` passes the first ASR
transcript through.

**Connectivity presets** (work with or without `--full-auto`; explicit
`--organizer-provider`/`--organizer-model` still override):

| flag | cleanup | notes |
|---|---|---|
| `-O` / `--offline` | local GGUF | nothing leaves the machine |
| `-F` / `--online-free` | OpenRouter, free models | long provider-diverse fallback chain (free tiers flap); free endpoints may log/train on inputs |
| `-P` / `--online-paid` | OpenRouter, paid models | deepseek-v3.2 → gemini-3-flash, then falls back to the free chain; paid endpoints aren't logged; needs `OPENROUTER_API_KEY`. Also swaps ASR to the hosted recognizer trio and enables parallel batch processing by default (see [ASR](#asr)) |

ASR stays local (whisper + sherpa) under `--offline` and `--online-free`;
`--online-paid` runs it hosted. On OpenRouter runs the uncertainty audit uses
its own semantic-judge chain, so the cleanup model does not grade its own work:
GPT-5.4 mini → Claude Fable 5 → Claude Opus 4.6 when the run is already paying,
and gemma-4-31b → nemotron-3-super when the cleanup chain is entirely free — a
free-preset run should not be billed for its audit, nor have its transcript sent
to endpoints its own privacy notice never named.
When a request carries a response schema, the client also sets
`provider.require_parameters` so OpenRouter never routes it to a provider that
would silently ignore the schema.

A run says what leaves the machine before it leaves — but only when that is
news. Asking for the remote path suppresses its own notice: `-P`, `-F`,
`--organizer-provider openrouter`, `--asr` naming a hosted backend, and the
interactive provider prompt are all informed decisions, and warning you about
the thing you just typed is the noise that teaches people to skip warnings.
What still announces itself is the path you did not ask for on this command
line — most usefully a hosted backend left in `~/.config/speech-note/asr`
months ago, where a run really can upload audio you were not thinking about.
Notices are printed once per process, not once per batch item.

`--no-privacy-notices` silences both regardless; `--privacy-notices` forces
both on. To turn them off for good, put `SPEECH_NOTE_NO_PRIVACY_NOTICES=1` in
the environment or in `~/.config/speech-note/env`.

`OPENROUTER_API_KEY` is read from the environment, or from
`~/.config/speech-note/env` (a `KEY=value` file, chmod 600) when the env var is
unset — so the key travels with the tool regardless of which directory you run
it from. A shell export always wins.

**Local (default):** launches the bundled Vulkan `llama-server` on
`127.0.0.1:8011` with the best installed GGUF that fits in RAM. Two models are
known to the config (`speech_note/config/catalog.py`): the bundled
**gemma-4-E2B** (`DEFAULT_GGUF_MODEL`, 3.2 GB, auto-installed on first use,
consent-gated) and the preferred **gemma-4-26B-A4B QAT** (`PREFERRED_GGUF_MODEL`,
14.2 GB, place it in `~/.cache/huggingface/gguf/` yourself). When the preferred
file is present and the RAM guard says it fits at the configured context, it is
selected automatically — measured on the labeled evals it annotates at the
hosted panel's level and keeps every labeled phrase in cleanup, where E2B finds
no annotations and loses content on every case. A **RAM guard** protects both
paths: the iGPU allocates from system RAM, so an oversized model doesn't fail
cleanly — it swap-thrashes. The default path refuses (loudly) to launch a model
that doesn't fit; an explicit `--organizer-gguf /path/to/model.gguf` is always
respected, with a warning. For full control over the server invocation, use
`--organizer-server-command` instead.

The server runs with full layer offload and KV-cache offload to the GPU
(~1.5x faster generation — measured 37 vs 24 tok/s on a 3.9k-token prompt). The
old Gemma-4-GGUF/Vulkan slot-init hang that forced KV onto the CPU is fixed in
the current llama.cpp; `--no-organizer-kv-offload` restores the workaround for
older stacks. One 890M-specific note for the 26B: `--organizer-gpu-layers auto`
(the default) spills its expert tensors to the CPU at the 64k default context
(~8.5 tok/s); at `--organizer-context-tokens 32768` or less, `--organizer-gpu-layers
999` fits fully on the iGPU and runs ~20 tok/s. The model the server *says* it
served is what gets recorded.

**OpenRouter (better accuracy at faster speed):**

```sh
export OPENROUTER_API_KEY=...   # checked at startup, not mid-run
speech-note --input long.m4a --organizer-provider openrouter
```

The preferred model list in `config.py` is an ordering, not a claim: it is
filtered against the live `/models` catalog before use, so stale entries are
dropped instead of becoming retry noise. **Privacy:** this sends your full
transcript off-machine, and free-tier endpoints may log or train on inputs;
the run prints a notice when that's about to happen.

Robustness details: the request timeout scales with the estimated work; a
response with `finish_reason == "length"` is reported as a truncated (failed)
cleanup rather than passed off as complete; output that is suspiciously short
relative to the *mean* source length — the mean, so one filler-heavy source
cannot demand a bloated transcript — is kept but flagged with a warning in the
output and diagnostics. An HTTP error advances to the next model in the chain;
only a rejected credential fails the run outright, since that is the one failure
another model cannot fix.

For arbitrary prompts against the same local server, keep it running yourself
and use `curl http://127.0.0.1:8011/v1/chat/completions`.

## Uncertainty annotation

On a hard recording the cleaned transcript is uniformly fluent prose whether
the audio was pristine or unintelligible, so the reader has no way to tell
which sentences to trust. A second LM pass audits the finished transcript
against every source and marks the passages the sources do not jointly
support: a bare `[n]` anchor in the text, and one numbered **"Unclear
passages:"** list at the end giving the competing readings ("sources also
heard: …"). Where the sources disagree, the audio was hard and cleanup picked
one reading — the note shows that it did, and what the alternatives were.

Design properties, in order of importance:

- **The pass cannot rewrite anything.** It returns JSON spans (constrained by
  an actual JSON schema — grammar-enforced on the local server); code inserts
  the anchors. The worst it can do is annotate nothing.
- **Alternatives are citations.** Each one names its source and quotes it
  verbatim; `verify_notes` looks every citation up as a contiguous word
  sequence in the actual sources and drops what it cannot find — a mechanical
  containment check, no linguistic judgement in code. Fabricated "alternatives"
  never reach the reader, and a display text that isn't a faithful reduction of
  its verified quote is replaced by the quote itself.
- **Placement is context-pinned.** Notes carry before/after context so the
  right instance of a repeated phrase gets the anchor; unanchorable notes join
  the same list, marked as such, instead of being dropped.
- **Judgement lives in the prompt, not the code.** The prompt teaches with a
  worked example (a real user/assistant message pair) and names the base rate;
  the majority rule says agreement between two sources is evidence about a
  third, not proof.

Defaults: ON for remote cleanup providers, ON for local runs that will serve
the preferred QAT 26B (it measured audit-clean on the labeled evals), OFF
otherwise — the bundled E2B's judgements are unreliable, and an explicit
`--organizer-gguf` / `--organizer-server-command` / `--organizer-api-base`
means an unknown model, which doesn't get judgement duties by default.
`-u` / `--annotate-uncertainty` forces it on, `-U` / `--no-annotate-uncertainty`
off. When the pass runs, `clean.plain.latest` keeps the pre-annotation prose
for TTS/pasting.

## Evals

The prompts and model choices above are all backed by labeled evals that run
the *shipped* machinery (imported from the product modules, so eval and product
cannot drift):

- `tools/annotation_eval.py` — the audit pass: recall over expected divergence
  regions, spurious/displaced/forbidden notes, citation rejections.
- `tools/cleanup_eval.py` — the cleanup pass: must-keep coverage, wrong-reading
  promotion, length band, disfluency residue.
- `tools/asr_eval.py` — hosted recognizers on real clips: labeled-anchor
  recall, degeneration (unique-8-gram ratio), invented proper nouns.
- `tools/make_eval_case.py` — turns any `--export-sources` directory into a
  new labeled case skeleton in two minutes.

Committed cases under `evals/cases/` are synthetic traps for observed failure
classes (confabulation promotion, displaced citations, second-instance
anchoring); real-recording cases live under `evals/local/`, which is gitignored
so nothing from a real recording can enter history. All tools take `--repeats`
(free routes vary wildly run to run) and pin OpenRouter serving to
bf16/fp16/fp8 quantizations, reporting the provider that actually answered.
`--api-base` points the same evals at a local llama-server, which is how local
models get numbers instead of reputations. See `evals/README.md`.

## Hardware notes and other GPUs

This was built and tuned on an **AMD Strix Point** laptop (Radeon 890M iGPU,
ROCm) — an integrated GPU that shares system RAM and has no CUDA. Several defaults
are shaped by that, and a discrete NVIDIA card changes the calculus. The defaults
all *work* on any machine (they fall back to CPU/Vulkan); this is about getting
the most out of better hardware.

**What the AMD iGPU forced, and why:**

- **The Parakeet source runs on the CPU (sherpa-onnx), not the GPU.** Parakeet-TDT
  is an autoregressive transducer; its decode loop is dominated by per-step launch
  overhead unless you have NVIDIA's CUDA-graph conditional-node decoding, which
  doesn't exist on ROCm. On this iGPU the GPU path bottoms out around 18–21×
  realtime. sherpa-onnx runs the *same* model's decode loop on the CPU in C++
  *and* it leaves the GPU free for the Whisper source, so the two run concurrently
  (the iGPU is shared, so a second GPU source would just serialize behind Whisper).
  On a discrete card with its own VRAM, neither constraint holds.
- **Whisper and the cleanup LM use Vulkan**, the portable GPU backend that works on
  AMD, via `whisper.cpp` and `llama.cpp`.
- **Small, quantized models** (Whisper `medium-q8_0`, Parakeet int8, a 2B-class
  cleanup GGUF) — sized for an iGPU sharing ~tens of GB of system RAM.
- **The XDNA2 NPU is unused** — no usable execution provider in this stack.

**If you have a discrete GPU or an NVIDIA GPU:**

- **PyTorch:** for NVIDIA, swap `torchWithRocm` for the CUDA build in `flake.nix`'s
  `pythonEnv` (nixpkgs `python3Packages.torchWithCuda`). Only PyTorch-based paths
  need this; the defaults don't use PyTorch at all.
- **Parakeet:** sherpa-onnx (CPU) is excellent and keeps the GPU free, so it's a
  fine default to keep. If you specifically want GPU TDT, sherpa-onnx also has CUDA
  execution providers, and on NVIDIA the autoregressive decode is fast (CUDA graphs)
  rather than overhead-bound. With dedicated VRAM you can run every source on the
  GPU (`--asr whisper-cpp@gpu --asr ctc@cuda:0`) since they no longer have to share
  one iGPU.
- **Whisper:** `whisper.cpp` has a CUDA build (or use a CUDA faster-whisper source
  via `--asr faster-whisper@cuda`); both are faster than Vulkan on NVIDIA. If you
  have a discrete GPU rather than an iGPU, you can also afford `large-v3` instead of
  `medium` and the larger organizer required to handle the hallucinated repetitions
  thereof.
- **Cleanup LM:** point `llama.cpp` at CUDA and run a bigger GGUF — with more
  VRAM a 7–12B cleanup model is realistic. Just pass `--organizer-gguf
  /path/to/model.gguf` (no config edit needed). A stronger cleanup model also
  matters if you use `--asr whisper-cpp:large-v3-turbo-q5_k`, whose hallucinated
  repeats the bundled gemma-E2B can't fully remove. Or sidestep local entirely with
  `--online-paid` / `--online-free` (OpenRouter).
- **Quantization:** more memory means you can move up from int8/q8 to fp16 or
  larger checkpoints for quality.

None of this is wired to autodetect — the flake pins one coherent AMD-oriented
toolchain on purpose. Treat the above as the set of knobs to turn for a different
GPU.

## Diagnostics

```sh
jq '.timings, .asr, .cleanup, .errors, .skips' diagnostics.latest.json
```

The diagnostics file (schema 6) records the resolved config, the input
duration, every transcript with its provenance (label, producing model, kind),
per-pass timing *and realtime factor*, the leveling gain actually applied,
cleanup telemetry (served model, finish reason, token estimates), the
annotation pass's applied notes as structured data (quote / alternatives /
anchored) with its rejected-citation count and timing, and an event timeline
stamped in seconds since run start. Errors and skips are separate lists;
`audio_levels.measured` says whether level data exists rather than reporting
zeros for file runs.

Exit code is `0` only if the run produced output (and, under `--full-auto`,
cleanup succeeded).

## License

GNU General Public License v3.0 (`GPL-3.0-or-later`) — see [LICENSE](LICENSE).
Distributing this or a derivative means shipping the source under the same
license.
