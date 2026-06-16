# speech-text

Local speech-to-text notes: microphone dictation and audio-file transcription.

The pipeline for every run:

1. capture or load audio
2. probe its duration once, then gain-normalize it
3. transcribe with Whisper (primary) and optionally a secondary local ASR model —
   in parallel when the two use different devices (the default pairing is
   whisper.cpp on the GPU plus sherpa-onnx Parakeet on the CPU, which don't contend)
4. clean the transcript with an LM that sees every source named, with its
   model and a reliability hint
5. commit all artifacts once: `raw.latest`, `clean.latest`,
   `recording.latest.wav` (live capture only), `diagnostics.latest.json`, and a
   timestamped copy of each under `logs/`

Code layout: the `speech_note/` package is the program (`cli` → `pipeline` /
`capture` → `transcribers` / `secondary` / `organizer`, with `session` holding
the run record). Supporting modules: `config` (all defaults and the backend
tables), `models` (consent-gated model download), `hardware` (GPU detection),
`audio` / `devices` / `terminal` / `naming` / `textproc`. `tools/` has the
whisper.cpp Vulkan probe and `tests/` the test suites.

Extending it (where new code goes):

- **a secondary ASR backend** — add it to `SECONDARY_BACKENDS` +
  `SECONDARY_DEFAULT_MODELS` in `config.py`, then either a `*Transcriber` class in
  `transcribers.py` (in-process, wired in `secondary.build_local_secondary_transcriber`)
  or a command in `secondary.build_subprocess_command` (+ list it in
  `SUBPROCESS_SECONDARY_BACKENDS`).
- **a primary ASR backend** — a class in `transcribers.py`, selected in
  `pipeline.build_primary_transcriber`.
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
prompts). The one model with no auto-install source is the cleanup GGUF — see
[Cleanup LM](#cleanup-lm). See "Models and network" below for the full policy.

Inside the shell, `speech-note` runs the Nix-store copy of the code — edits to
the working tree are picked up by `python -m speech_note` (or by re-entering
the shell). Tests: `python -m unittest discover -s tests` (fast logic suite).
The full-stack no-mock suite — real Whisper, real secondary ASR, real local
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
required model (Whisper ggml, the secondary ASR model, etc.) is missing, an
interactive run **asks first** — "*&lt;model&gt; is not installed. download ~&lt;size&gt;
into &lt;dir&gt;? [y/N]*" — and only fetches it if you agree. `--auto-download` answers
yes to all (for scripts/`--full-auto`); a non-interactive run without it fails
with a message naming the model and where to put it. The cleanup GGUF is the one
exception: it has no pinned download source, so place it at the path in
`config.DEFAULT_GGUF_MODEL` yourself (or use `--online-free` / `--online-paid`).
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

## Primary ASR (Whisper)

Default: `whisper.cpp` with Vulkan, model `medium-q8_0` from
`~/.cache/whisper.cpp`. The first run prompts to download it; to pre-install (or
fetch a different model) manually:

```sh
mkdir -p ~/.cache/whisper.cpp && cd ~/.cache/whisper.cpp
whisper-cpp-download-ggml-model medium-q8_0
```

- `--asr-model tiny.en` etc. selects another ggml model (faster-whisper model
  names are aliased automatically)
- `--whisper-cpp-model /path/model.bin` uses an explicit file; diagnostics
  record the file that actually ran, not the alias
- `--asr-backend faster-whisper` is the CPU fallback
- `--whisper-cpp-device cpu` forces CPU (default is GPU index 0; pass another
  index to pick a GPU); `python tools/whisper_cpp_probe.py` checks whether
  whisper-cli sees the Vulkan backend

Whisper-only run:

```sh
speech-note --input-audio rec.m4a --no-secondary-asr --organizer-mode off
```

## Secondary ASR

`--secondary-asr-backend` selects the engine; the model defaults to the standard
one for that backend, so changing the backend never silently reuses a mismatched
model name (override with `--secondary-asr-model`):

| backend | default model | runs on |
|---|---|---|
| `sherpa` (default) | `csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8` | CPU (overlaps the GPU primary) |
| `onnx` | `nemo-parakeet-tdt-0.6b-v2` int8 | CPU |
| `crispasr` | `parakeet-tdt-0.6b-v2-q4_k.gguf` | GPU (Vulkan) |
| `ctc` | `nvidia/parakeet-ctc-0.6b` | CPU by default (overlaps the GPU primary; `--secondary-asr-device` to force GPU) |
| `pocketsphinx` | bundled en-us | CPU |

Secondary-only transcription of a file:

```sh
speech-note --secondary-only --input-audio rec.m4a
speech-note --secondary-only --input-audio rec.m4a --secondary-asr-backend crispasr
```

Notes:

- the default `sherpa` backend runs the same Parakeet-TDT-0.6B-v2 int8 model as
  `onnx`, but decodes the transducer in C++ instead of onnx-asr's Python
  per-segment loop. On the CPU that is ~3x faster (48–70x realtime here vs ~18x for
  `onnx`), and it keeps the pass off the GPU so it overlaps the Vulkan Whisper
  primary. Output carries punctuation and casing. There is **no length limit**:
  silero VAD segments the audio (each segment capped at `SHERPA_MAX_SEGMENT_SECONDS`,
  20s) so the FastConformer encoder never sees the long sequence that OOMs a single
  full-attention pass; decoding is sequential, not batched (on the CPU, batching
  only adds padding waste — it is a GPU occupancy lever). The model bundle and the
  silero VAD file come from the Hugging Face cache (offline by default;
  `--auto-download` fetches them once, ~650 MB)
- the `onnx` backend is the same Parakeet-TDT model run through onnx-asr's Python
  decode loop — kept as a portable fallback, but ~3x slower than `sherpa` on the
  CPU. `crispasr` runs Parakeet-TDT on the GPU (Vulkan), useful only for
  `--secondary-only` since it contends with the Whisper primary
- the CTC backend has no length limit: audio past Parakeet CTC's ~400s
  position-embedding cliff is transcribed by the HF pipeline's native long-form
  striding (240s windows with a 5s stride), which windows the audio and stitches
  at the logit level. The transformers Parakeet-CTC port omits
  `config.inputs_to_logits_ratio`, which the pipeline needs to align strides
  (without it only the first window survives — ~755 words for a 600s clip vs
  ~1840); the backend sets it from the encoder subsampling factor x hop length
  (8 x 160 = 1280)
- a backend that finds no speech (e.g. its VAD saw none) is reported as a note,
  separately from errors, in both the terminal output and diagnostics

## Cleanup LM

`--organizer-mode llama` (default) sends all sources to an OpenAI-compatible
endpoint; `heuristic` is a cheap regex cleanup; `off` passes the primary
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
(`DEFAULT_GGUF_MODEL`) — place a chat GGUF at that path (this is the one model
that isn't auto-installed), or point elsewhere with `--organizer-server-command`.
Run uses full layer offload and KV-cache offload to the GPU
(~1.5x faster generation — measured 37 vs 24 tok/s on a 3.9k-token prompt). The
old Gemma-4-GGUF/Vulkan slot-init hang that forced KV onto the CPU is fixed in
the current llama.cpp; `--no-organizer-kv-offload` restores the workaround for
older stacks. The model the server *says* it served is what gets recorded.

**OpenRouter (long recordings):**

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

- **Secondary ASR runs on the CPU (sherpa-onnx), not the GPU.** Parakeet-TDT is an
  autoregressive transducer; its decode loop is dominated by per-step launch
  overhead unless you have NVIDIA's CUDA-graph conditional-node decoding, which
  doesn't exist on ROCm. On this iGPU the GPU path bottoms out around 18–21×
  realtime. sherpa-onnx runs the *same* model's decode loop in C++ on the CPU at
  48–70× — faster *and* it leaves the GPU free for the Whisper primary, so the two
  passes overlap (the iGPU is shared, so a GPU secondary would just serialize
  behind Whisper). On a discrete card with its own VRAM, neither constraint holds.
- **Whisper and the cleanup LM use Vulkan**, the portable GPU backend that works on
  AMD, via `whisper.cpp` and `llama.cpp`.
- **Small, quantized models** (Whisper `medium-q8_0`, Parakeet int8, a 2B-class
  cleanup GGUF) — sized for an iGPU sharing ~tens of GB of system RAM.
- **The XDNA2 NPU is unused** — no usable execution provider in this stack.

**If you have a discrete GPU or an NVIDIA GPU:**

- **PyTorch:** for NVIDIA, swap `torchWithRocm` for the CUDA build in `flake.nix`'s
  `pythonEnv` (nixpkgs `python3Packages.torchWithCuda`). Only PyTorch-based paths
  need this; the defaults don't use PyTorch at all.
- **Secondary ASR:** sherpa-onnx (CPU) is still excellent and keeps the GPU free,
  so it's a fine default to keep. If you specifically want GPU TDT, sherpa-onnx
  also has CUDA execution providers, and on NVIDIA the autoregressive decode is
  fast (CUDA graphs) rather than overhead-bound. With dedicated VRAM you can also
  drop the "secondary must be CPU to overlap the primary" rule and run both on the
  GPU.
- **Whisper:** `whisper.cpp` has a CUDA build (or use a CUDA `faster-whisper` via
  `--asr-backend faster-whisper --asr-device cuda`); both are faster than Vulkan on
  NVIDIA. If you have a discrete GPU rather than an iGPU, you can also afford 
  `large-v3` instead of `medium`.
- **Cleanup LM:** point `llama.cpp` at CUDA, and/or run a bigger GGUF — with more
  VRAM a 7–12B cleanup model is realistic. Or sidestep local entirely with
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
