"""ASR transcriber implementations.

Each transcriber exposes transcribe_file(path, language) -> str and raises on
failure. Heavy runtimes are imported lazily so building one backend never pays
for another's dependencies.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

from .config import (
    SAMPLE_WIDTH,
    SHERPA_MAX_SEGMENT_SECONDS,
    SHERPA_VAD_FILE,
    SHERPA_VAD_REPO,
    WHISPER_CPP_MODEL_ALIASES,
    WHISPER_CPP_MODEL_DIR,
)
from .terminal import debug_log
from .textproc import normalize_paragraphs, normalize_spacing


def whisper_cpp_model_name(model_name: str) -> str:
    if model_name in WHISPER_CPP_MODEL_ALIASES:
        return WHISPER_CPP_MODEL_ALIASES[model_name]
    leaf = model_name.rsplit("/", 1)[-1]
    return leaf.removeprefix("faster-whisper-").removeprefix("whisper-")


def default_whisper_cpp_model_path(model_name: str) -> Path:
    return WHISPER_CPP_MODEL_DIR / f"ggml-{whisper_cpp_model_name(model_name)}.bin"


def resolve_whisper_cpp_binary(binary: Path | None = None) -> Path:
    if binary is not None:
        return binary
    found = shutil.which("whisper-cli")
    if found is None:
        raise RuntimeError("whisper-cli not found; enter the Nix shell or set --whisper-cpp-binary")
    return Path(found)


class FasterWhisperTranscriber:
    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        compute_type: str,
        cpu_threads: int,
        download_root: Path | None,
    ) -> None:
        from faster_whisper import WhisperModel

        self.model_name = model_name
        kwargs: dict[str, Any] = {
            "device": device,
            "compute_type": compute_type,
            "cpu_threads": cpu_threads,
        }
        if download_root is not None:
            kwargs["download_root"] = str(download_root)
        self.model = WhisperModel(model_name, **kwargs)

    def transcribe_file(self, path: Path, language: str) -> str:
        segments, _info = self.model.transcribe(
            str(path),
            language=language,
            beam_size=1,
            best_of=1,
            condition_on_previous_text=False,
            vad_filter=False,
            without_timestamps=True,
        )
        return normalize_spacing(" ".join(segment.text.strip() for segment in segments))


class WhisperCppTranscriber:
    def __init__(
        self,
        *,
        model_name: str,
        model_path: Path | None,
        binary: Path | None,
        cpu_threads: int,
        device: int,
        use_gpu: bool,
        auto_download: bool = False,
    ) -> None:
        self.model_name = model_name
        self.model_path = model_path if model_path is not None else default_whisper_cpp_model_path(model_name)
        self.binary = resolve_whisper_cpp_binary(binary)
        self.cpu_threads = cpu_threads
        self.device = device
        self.use_gpu = use_gpu
        self.auto_download = auto_download
        self.last_stderr = ""

    def _ensure_model_installed(self) -> None:
        """Download the ggml model on consent if it is missing from the cache."""
        if self.model_path.exists():
            return
        model = whisper_cpp_model_name(self.model_name)
        if self.model_path != default_whisper_cpp_model_path(self.model_name):
            # An explicit --whisper-cpp-model path we can't source automatically.
            raise RuntimeError(f"whisper.cpp model not found: {self.model_path}")
        from .models import require_consent

        require_consent(
            f"Whisper model '{model}' (whisper.cpp ggml)",
            str(WHISPER_CPP_MODEL_DIR),
            auto_yes=self.auto_download,
        )
        downloader = shutil.which("whisper-cpp-download-ggml-model")
        if downloader is None:
            raise RuntimeError(
                "whisper-cpp-download-ggml-model is not on PATH; enter the Nix dev shell"
            )
        WHISPER_CPP_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        print(f"downloading Whisper model {model}...", file=sys.stderr, flush=True)
        result = subprocess.run(
            [downloader, model, str(WHISPER_CPP_MODEL_DIR)],
            check=False, capture_output=True, text=True,
        )
        if result.returncode != 0 or not self.model_path.exists():
            detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
            raise RuntimeError(f"failed to download whisper.cpp model {model}: {detail}")

    @property
    def effective_model(self) -> str:
        """The model that actually runs: the resolved file, not the alias."""
        return self.model_path.name

    def _command(self, path: Path, language: str) -> list[str]:
        self._ensure_model_installed()
        command = [
            str(self.binary),
            "--model", str(self.model_path),
            "--file", str(path),
            "--language", language,
            "--threads", str(self.cpu_threads),
            "--beam-size", "1",
            "--best-of", "1",
            "--no-timestamps",
            "--no-prints",
            "--device", str(self.device),
        ]
        if not self.use_gpu:
            command.append("--no-gpu")
        return command

    def transcribe_file(self, path: Path, language: str) -> str:
        result = subprocess.run(
            self._command(path, language),
            check=False,
            capture_output=True,
            text=True,
        )
        self.last_stderr = result.stderr
        if result.returncode != 0:
            detail = normalize_spacing(result.stderr) or f"exit code {result.returncode}"
            raise RuntimeError(f"whisper.cpp failed: {detail}")
        return normalize_paragraphs(result.stdout)

    def backend_summary(self) -> str:
        interesting = [
            line
            for line in self.last_stderr.splitlines()
            if "ggml_vulkan:" in line or "load_backend:" in line or "whisper_init_state:" in line
        ]
        return " | ".join(interesting[:6])

    def warm_model_cache(self) -> None:
        """Pre-read the model file so the first subprocess hits the page cache."""
        try:
            with self.model_path.open("rb") as handle:
                while handle.read(16 * 1024 * 1024):
                    pass
        except OSError:
            pass


class CTCTranscriber:
    """Parakeet CTC (and similar) via transformers, in-process.

    Long audio is handled by the HF ASR pipeline's native long-form transcription
    (chunk_length_s / stride_length_s): it windows the waveform, runs each window,
    and stitches at the logit level using the stride, so there is no maximum input
    length and no text-level merging.

    One catch: the transformers ParakeetForCTC port does not set
    config.inputs_to_logits_ratio, which the pipeline needs to align chunk strides
    in logit space (it defaults the value to 1, which makes chunk_length_s keep
    only the first window — 755 words for a 600s clip vs ~1840 once fixed). We set
    it from the encoder's subsampling factor x the feature hop length.
    """

    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        download_root: Path | None,
        chunk_length_seconds: float | None,
        stride_seconds: float | None,
        auto_download: bool,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.download_root = str(download_root) if download_root is not None else None
        self.chunk_length_seconds = chunk_length_seconds
        self.stride_seconds = stride_seconds or 0.0
        self.auto_download = auto_download
        self.asr_pipeline = None
        self.load_seconds: float | None = None

    def _pipeline_device(self) -> int:
        text = self.device.strip().lower()
        if not text or text == "cpu":
            return -1
        if text == "auto":
            try:
                import torch

                return 0 if torch.cuda.is_available() else -1
            except Exception:
                return -1
        if text.startswith("cuda"):
            parts = text.split(":", 1)
            if len(parts) == 2 and parts[1].isdigit():
                return int(parts[1])
            return 0
        if text.isdigit():
            return int(text)
        return -1

    def ensure_loaded(self) -> None:
        if self.asr_pipeline is not None:
            return
        started = time.monotonic()
        from transformers import AutoModelForCTC, AutoProcessor, pipeline

        from .models import load_or_install

        def _load(local_only: bool) -> tuple[Any, Any]:
            kwargs: dict[str, Any] = {"local_files_only": local_only}
            if self.download_root is not None:
                kwargs["cache_dir"] = self.download_root
            proc = AutoProcessor.from_pretrained(self.model_name, **kwargs)
            mdl = AutoModelForCTC.from_pretrained(self.model_name, **kwargs)
            return proc, mdl

        processor, model = load_or_install(
            _load,
            label=f"secondary ASR model '{self.model_name}' (Hugging Face)",
            dest=self.download_root or "the Hugging Face cache",
            auto_yes=self.auto_download,
        )
        self._fix_logit_ratio(model, processor)
        pipeline_kwargs: dict[str, Any] = {}
        pipeline_device = self._pipeline_device()
        if pipeline_device >= 0:
            pipeline_kwargs["device"] = pipeline_device
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            **pipeline_kwargs,
        )
        self.load_seconds = round(time.monotonic() - started, 3)
        debug_log(f"ctc pipeline ready model={self.model_name} device={pipeline_device}")

    @staticmethod
    def _fix_logit_ratio(model: Any, processor: Any) -> None:
        """Set config.inputs_to_logits_ratio when the model port omits it.

        The ASR pipeline aligns chunk strides in logit space using this value and
        defaults it to 1 when absent, which breaks chunk_length_s long-form
        transcription (only the first window survives). For a mel-input encoder it
        is the encoder subsampling factor x the feature hop length (Parakeet:
        8 x 160 = 1280 samples per output frame).
        """
        config = model.config
        if getattr(config, "inputs_to_logits_ratio", 1) not in (None, 0, 1):
            return
        encoder = getattr(config, "encoder_config", None)
        subsampling = getattr(encoder, "subsampling_factor", None)
        hop_length = getattr(processor.feature_extractor, "hop_length", None)
        if subsampling and hop_length:
            config.inputs_to_logits_ratio = int(subsampling) * int(hop_length)
            debug_log(f"ctc set inputs_to_logits_ratio={config.inputs_to_logits_ratio}")

    def transcribe_file(self, path: Path, language: str, *, duration_seconds: float | None = None) -> str:
        del language, duration_seconds  # English-only; native chunking handles length.
        self.ensure_loaded()
        assert self.asr_pipeline is not None
        kwargs: dict[str, Any] = {}
        if self.chunk_length_seconds is not None:
            # Native long-form transcription: the pipeline windows the audio,
            # runs each window, and stitches at the logit level using the stride.
            kwargs["chunk_length_s"] = self.chunk_length_seconds
            if self.stride_seconds:
                kwargs["stride_length_s"] = self.stride_seconds
        result = self.asr_pipeline(str(path), **kwargs)
        text = result.get("text", "") if isinstance(result, dict) else result
        return normalize_spacing(str(text))


class SherpaTranscriber:
    """Parakeet TDT via sherpa-onnx, in-process on the CPU.

    Same Parakeet-TDT-0.6B-v2 int8 model as the onnx backend, but sherpa-onnx runs
    the transducer decode loop in C++ instead of onnx-asr's Python per-segment loop,
    which is ~3x faster on the CPU (48-70x realtime vs 18x, measured) and keeps the
    pass off the GPU so it overlaps the Vulkan Whisper primary. Output carries
    punctuation and casing.

    Long audio is handled by silero VAD: the file is segmented into speech regions
    capped at SHERPA_MAX_SEGMENT_SECONDS and each segment is decoded independently,
    so the FastConformer encoder never sees the long sequence that OOMs a single
    full-attention pass. There is no input length limit. Decoding is sequential, not
    batched: on the CPU, batching only adds padding waste (it is a GPU occupancy
    lever) and measured slower.
    """

    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        download_root: Path | None,
        num_threads: int,
        auto_download: bool,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.download_root = str(download_root) if download_root is not None else None
        self.num_threads = max(1, int(num_threads))
        self.auto_download = auto_download
        self.recognizer: Any = None
        self.vad: Any = None
        self.load_seconds: float | None = None

    def _provider(self) -> str:
        text = self.device.strip().lower()
        # The Nix onnxruntime exposes only the CPU provider here, and a CPU pass
        # overlaps the GPU primary; only honor an explicit CUDA request.
        return "cuda" if text.startswith("cuda") else "cpu"

    def _resolve_model_files(self) -> tuple[str, str, str, str, str]:
        from huggingface_hub import hf_hub_download, snapshot_download

        from .models import load_or_install

        def _load(local_only: bool) -> tuple[str, str]:
            kwargs: dict[str, Any] = {"local_files_only": local_only}
            if self.download_root is not None:
                kwargs["cache_dir"] = self.download_root
            model_dir = snapshot_download(
                self.model_name, allow_patterns=["*.onnx", "tokens.txt"], **kwargs
            )
            vad = hf_hub_download(SHERPA_VAD_REPO, SHERPA_VAD_FILE, **kwargs)
            return model_dir, vad

        model_dir, vad_path = load_or_install(
            _load,
            label=f"sherpa-onnx Parakeet model '{self.model_name}' (+ silero VAD)",
            dest=self.download_root or "the Hugging Face cache",
            size_hint="650 MB",
            auto_yes=self.auto_download,
        )
        base = Path(model_dir)

        def pick(*names: str) -> str:
            for name in names:
                path = base / name
                if path.exists():
                    return str(path)
            raise RuntimeError(f"sherpa-onnx bundle {self.model_name} missing {names[0]}")

        encoder = pick("encoder.int8.onnx", "encoder.onnx")
        decoder = pick("decoder.int8.onnx", "decoder.onnx")
        joiner = pick("joiner.int8.onnx", "joiner.onnx")
        tokens = pick("tokens.txt")
        return encoder, decoder, joiner, tokens, vad_path

    def ensure_loaded(self) -> None:
        if self.recognizer is not None:
            return
        started = time.monotonic()
        import sherpa_onnx

        encoder, decoder, joiner, tokens, vad_path = self._resolve_model_files()
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            tokens=tokens,
            num_threads=self.num_threads,
            model_type="nemo_transducer",
            decoding_method="greedy_search",
            provider=self._provider(),
        )
        vad_config = sherpa_onnx.VadModelConfig()
        vad_config.silero_vad.model = vad_path
        vad_config.silero_vad.threshold = 0.5
        vad_config.silero_vad.min_silence_duration = 0.1
        vad_config.silero_vad.max_speech_duration = float(SHERPA_MAX_SEGMENT_SECONDS)
        vad_config.sample_rate = 16_000
        self._vad_config = vad_config
        self.load_seconds = round(time.monotonic() - started, 3)
        debug_log(f"sherpa recognizer ready model={self.model_name} provider={self._provider()}")

    def transcribe_file(self, path: Path, language: str, *, duration_seconds: float | None = None) -> str:
        del language, duration_seconds  # English model; VAD segmentation handles length.
        self.ensure_loaded()
        assert self.recognizer is not None
        import librosa
        import numpy as np
        import sherpa_onnx

        audio, _ = librosa.load(str(path), sr=16_000, mono=True)
        # Peak-normalize before VAD: the silero threshold is amplitude-relative, so
        # quiet input (the pipeline's gain stage runs upstream, but --secondary-only
        # and looped test clips may not) would otherwise drop most speech.
        peak = float(np.abs(audio).max()) if audio.size else 0.0
        if 0.0 < peak < 0.95:
            audio = (audio / peak * 0.95).astype(np.float32)
        vad = sherpa_onnx.VoiceActivityDetector(
            self._vad_config,
            buffer_size_in_seconds=max(60.0, len(audio) / 16_000 + 5.0),
        )
        window = 512
        texts: list[str] = []

        def drain() -> None:
            while not vad.empty():
                segment = vad.front.samples
                stream = self.recognizer.create_stream()
                stream.accept_waveform(16_000, segment)
                self.recognizer.decode_stream(stream)
                if stream.result.text:
                    texts.append(stream.result.text)
                vad.pop()

        for start in range(0, len(audio) - window, window):
            vad.accept_waveform(audio[start : start + window])
            drain()
        vad.flush()
        drain()
        return normalize_spacing(" ".join(texts))


class PocketSphinxTranscriber:
    def __init__(self, *, model_name: str, sample_rate: int) -> None:
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.decoder: Any = None

    def ensure_loaded(self) -> None:
        if self.decoder is not None:
            return
        from pocketsphinx import Config, Decoder

        acoustic_model, language_model, dictionary = self._resolve_model_paths()
        config = Config()
        config.set_string("-hmm", str(acoustic_model))
        config.set_string("-lm", str(language_model))
        config.set_string("-dict", str(dictionary))
        config.set_string("-logfn", os.devnull)
        self.decoder = Decoder(config)

    def _resolve_model_paths(self) -> tuple[Path, Path, Path]:
        from pocketsphinx import get_model_path

        if self.model_name not in {"default", "pocketsphinx-en-us"}:
            raise RuntimeError("the pocketsphinx backend supports only the bundled English model")
        base = Path(get_model_path())
        for candidate in (base, base / "model"):
            acoustic_model = candidate / "en-us" / "en-us"
            language_model = candidate / "en-us" / "en-us.lm.bin"
            dictionary = candidate / "en-us" / "cmudict-en-us.dict"
            if acoustic_model.exists() and language_model.exists() and dictionary.exists():
                return acoustic_model, language_model, dictionary
        raise RuntimeError(f"PocketSphinx default model files not found under {base}")

    def transcribe_file(self, path: Path, language: str, *, duration_seconds: float | None = None) -> str:
        del language, duration_seconds
        self.ensure_loaded()
        assert self.decoder is not None
        from .audio import convert_to_pcm_wav

        with tempfile.TemporaryDirectory(prefix="speech-note-ps-") as tmp_dir:
            wav_path = Path(tmp_dir) / "audio.wav"
            convert_to_pcm_wav(path, wav_path, sample_rate=self.sample_rate)
            with wave.open(str(wav_path), "rb") as handle:
                if handle.getnchannels() != 1 or handle.getsampwidth() != SAMPLE_WIDTH:
                    raise RuntimeError("PocketSphinx input must be mono 16-bit PCM")
                self.decoder.start_utt()
                while chunk := handle.readframes(2048):
                    self.decoder.process_raw(chunk, False, False)
                self.decoder.end_utt()
                hypothesis = self.decoder.hyp()
        return normalize_spacing(hypothesis.hypstr if hypothesis is not None else "")


PrimaryTranscriber = FasterWhisperTranscriber | WhisperCppTranscriber
LocalSecondaryTranscriber = SherpaTranscriber | CTCTranscriber | PocketSphinxTranscriber


def thread_env(cpu_threads: int) -> dict[str, str]:
    """Environment for compute subprocesses, with explicit thread counts.

    Values are set explicitly (not setdefault) so a leaked OMP_NUM_THREADS from
    the parent shell cannot silently undercut the configured parallelism.
    """
    env = os.environ.copy()
    threads = str(cpu_threads)
    env["OMP_NUM_THREADS"] = threads
    env["OPENBLAS_NUM_THREADS"] = threads
    env["MKL_NUM_THREADS"] = threads
    env["NUMEXPR_NUM_THREADS"] = threads
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


@contextlib.contextmanager
def limited_torch_threads(cpu_threads: int):
    """Best-effort torch thread cap for in-process transcribers."""
    try:
        import torch

        torch.set_num_threads(cpu_threads)
        with contextlib.suppress(RuntimeError):
            torch.set_num_interop_threads(min(4, cpu_threads))
    except Exception:
        pass
    yield
