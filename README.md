# speech-text

A vibe-coded local-first system for high-quality microphone dictation and audio-file transcription.

The pipeline for every run:

1. capture or load audio
2. probe its duration once, then gain-normalize it
3. transcribe with a collection of local ASR sources, running sources on
   different devices concurrently (the default collection is whisper.cpp on the
   GPU plus sherpa-onnx Parakeet on the CPU, which don't contend)
4. clean the transcript with an LM that sees every source named, with its
   model and any reliability note
5. commit all artifacts once: `raw.latest`, `clean.latest`,
   `recording.latest.wav` (live capture only), `diagnostics.latest.json`, and a
   timestamped copy of each under `logs/`

**A note for other users:** I highly recommend changing the settings to suit your
hardware, and in particular if you have an OpenRouter API key and are not concerned
with privacy **I recommend using the OpenRouter organizer path** rather than the 
mildly-weak gemma-4-E2B default.

Code layout: the `speech_note/` package is the program (`cli` → `pipeline` /
`capture` → `transcribers` / `asr` / `organizer`, with `session` holding
the run record). Supporting modules: `config` (all defaults and the backend
tables), `models` (consent-gated model download), `hardware` (GPU detection),
`audio` / `devices` / `terminal` / `naming` / `textproc`. `tools/` has the
whisper.cpp Vulkan probe and `tests/` the test suites.

Extending it (where new code goes):

- **an ASR backend** — add a `BackendSpec` to the `ASR_BACKENDS` registry in
  `config.py`, then either a `*Transcriber` class in `transcribers.py` (in-process,
  wired in `asr.build_transcriber`) or, for an external CLI, a command in
  `asr.build_subprocess_command` (with `in_process=False` in the registry).
- **a cleanup provider** — extend `organizer.py` (`build_organizer` / `ChatClient`).
- **a model that needs downloading** — detect "missing", then route through
  `models.require_consent` / `models.load_or_install` so it shares the install UX.

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
the shell). Tests: `python -m unittest discover -s tests` (fast logic suite).
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
speech-note --input-audio /path/to/audio.m4a
```

Unattended runs that should leave only a cleaned transcript in the current
directory:

```sh
speech-note --input-audio /path/to/audio.m4a --full-auto
```

`--full-auto` writes an auto-named `<input>-clean.txt` only if cleanup actually
succeeded (failed, empty, or truncated cleanup writes nothing and the process
exits nonzero), keeps the usual artifacts in a temporary directory, and on any
error drops a matching `<input>-clean-diagnostics.json` next to the output.

Add external transcripts (Google Recorder etc.) as extra sources for cleanup —
`.srt`/`.vtt` files are parsed down to their text, `.txt` and `.json` are
passed through:

```sh
speech-note --input-audio rec.m4a --extra-transcript rec.google.txt
```

A zip export containing one recording plus transcript files:

```sh
speech-note --input-archive /path/to/export.zip
```

Cleanup-only modes:

```sh
# existing transcripts, no audio:
speech-note --primary-transcript rec.whisper.txt --secondary-transcript rec.parakeet.txt

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
speech-note --input-audio rec.m4a --asr whisper-cpp:small.en@gpu --asr sherpa
speech-note --input-audio rec.m4a --asr 'whisper-cpp:medium-q8_0,ctc@cpu'
speech-note --input-audio rec.m4a --asr sherpa --organizer-mode off   # one source, raw
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
| `--offline` | local GGUF | nothing leaves the machine |
| `--online-free` | OpenRouter, free models | long provider-diverse fallback chain (free tiers flap); free endpoints may log/train on inputs |
| `--online-paid` | OpenRouter, paid models | deepseek-v3.2 → gemini-3-flash, then falls back to the free chain; paid endpoints aren't logged; needs `OPENROUTER_API_KEY` |

ASR is local (whisper + sherpa) in every mode — no OpenRouter ASR has been set up for
this: the transcription models (`gpt-4o-mini-transcribe`, `whisper-large-v3`)
reject chat audio input, and the audio-LLMs that accept it truncate long audio.

`OPENROUTER_API_KEY` is read from the environment, or from
`~/.config/speech-note/env` (a `KEY=value` file, chmod 600) when the env var is
unset — so the key travels with the tool regardless of which directory you run
it from. A shell export always wins.

**Local (default):** launches the bundled Vulkan `llama-server` on
`127.0.0.1:8011` with the GGUF configured in `speech_note/config.py`
(`DEFAULT_GGUF_MODEL`, gemma-4-E2B). It is fetched from `DEFAULT_GGUF_REPO` on
first use like the other models (consent-gated; `--auto-download` to skip the
prompt) — or place a GGUF at that path yourself. To run a **stronger local
cleanup model** without touching the config or hand-writing a launch command, point
`--organizer-gguf /path/to/model.gguf` at any chat GGUF; speech-note serves it
with the same flags. (A bigger model is worth it if your hardware allows — and is
effectively required to clean up the looping that `--asr
whisper-cpp:large-v3-turbo-q5_k` produces, which gemma-E2B can't.) For full control
over the server invocation, use `--organizer-server-command` instead.
Run uses full layer offload and KV-cache offload to the GPU
(~1.5x faster generation — measured 37 vs 24 tok/s on a 3.9k-token prompt). The
old Gemma-4-GGUF/Vulkan slot-init hang that forced KV onto the CPU is fixed in
the current llama.cpp; `--no-organizer-kv-offload` restores the workaround for
older stacks. The model the server *says* it served is what gets recorded.

**OpenRouter (better accuracy at faster speed):**

```sh
export OPENROUTER_API_KEY=...   # checked at startup, not mid-run
speech-note --input-audio long.m4a --organizer-provider openrouter
```

The preferred model list in `config.py` is an ordering, not a claim: it is
filtered against the live `/models` catalog before use, so stale entries are
dropped instead of becoming retry noise. **Privacy:** this sends your full
transcript off-machine, and free-tier endpoints may log or train on inputs;
the run prints a notice when that's about to happen.

Robustness details: the request timeout scales with the estimated work; a
response with `finish_reason == "length"` is reported as a truncated (failed)
cleanup rather than passed off as complete; output that is suspiciously short
relative to the shortest source is kept but flagged with a warning in the
output and diagnostics.

For arbitrary prompts against the same local server, keep it running yourself
and use `curl http://127.0.0.1:8011/v1/chat/completions`.

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

The diagnostics file records the resolved config, the input duration, every
transcript with its provenance (label, producing model, kind), per-pass timing
*and realtime factor*, the normalization gain actually applied, cleanup
telemetry (served model, finish reason, token estimates), and an event timeline
stamped in seconds since run start. Errors and skips are separate lists;
`audio_levels.measured` says whether level data exists rather than reporting
zeros for file runs.

Exit code is `0` only if the run produced output (and, under `--full-auto`,
cleanup succeeded).

## License

GNU General Public License v3.0 (`GPL-3.0-or-later`) — see [LICENSE](LICENSE).
Distributing this or a derivative means shipping the source under the same
license.
