"""ASR transcriber implementations.

Every backend subclasses Transcriber: transcribe_file raises on failure, and the
two prepare hooks default to no-ops so callers never need duck-typing. Heavy
runtimes are imported lazily so building one backend never pays for another's
dependencies.
"""

from __future__ import annotations

import base64
import contextlib
import json
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


class Transcriber:
    """One ASR backend.

    ensure_downloaded makes the model present (consent-gated) without loading it;
    ensure_loaded brings the heavy runtime up. Both default to no-ops so backends
    with nothing to fetch or keep resident don't have to stub them.
    """

    def ensure_downloaded(self) -> None:
        return None

    def ensure_loaded(self) -> None:
        return None

    def transcribe_file(
        self, path: Path, language: str, *, duration_seconds: float | None = None
    ) -> str:
        raise NotImplementedError


class FasterWhisperTranscriber(Transcriber):
    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        compute_type: str,
        cpu_threads: int,
        download_root: Path | None,
        auto_download: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.download_root = download_root
        self.auto_download = auto_download
        self.model: Any = None

    def ensure_loaded(self) -> None:
        """Load the model, downloading it (consent-gated) on a cache miss.

        Lazy and gated, like the other backends: construction never touches the
        network, so a source that is built but never run — or one the user
        declines to download — costs nothing and prompts nothing.
        """
        if self.model is not None:
            return
        from faster_whisper import WhisperModel

        from .install import load_or_install

        kwargs: dict[str, Any] = {
            "device": self.device,
            "compute_type": self.compute_type,
            "cpu_threads": self.cpu_threads,
        }
        if self.download_root is not None:
            kwargs["download_root"] = str(self.download_root)

        def _load(local_only: bool) -> Any:
            return WhisperModel(self.model_name, local_files_only=local_only, **kwargs)

        self.model = load_or_install(
            _load,
            label=f"faster-whisper model '{self.model_name}' (Hugging Face)",
            dest=str(self.download_root) if self.download_root is not None else "the Hugging Face cache",
            auto_yes=self.auto_download,
        )

    def ensure_downloaded(self) -> None:
        # faster-whisper couples download and load; loading is the only handle we
        # have on the fetch, so the up-front prep step loads here too.
        self.ensure_loaded()

    def transcribe_file(
        self, path: Path, language: str, *, duration_seconds: float | None = None
    ) -> str:
        del duration_seconds  # faster-whisper handles any length natively
        self.ensure_loaded()
        assert self.model is not None
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


class WhisperCppTranscriber(Transcriber):
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
        # Resolved lazily (first command build): a missing whisper-cli then fails
        # this one source instead of aborting construction of the whole collection.
        self._binary_arg = binary
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
        from .install import require_consent

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

    def ensure_downloaded(self) -> None:
        """Download the ggml model (consent-gated) without running a transcription."""
        self._ensure_model_installed()

    @property
    def effective_model(self) -> str:
        """The model that actually runs: the resolved file, not the alias."""
        return self.model_path.name

    def _command(self, path: Path, language: str) -> list[str]:
        self._ensure_model_installed()
        command = [
            str(resolve_whisper_cpp_binary(self._binary_arg)),
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

    def transcribe_file(
        self, path: Path, language: str, *, duration_seconds: float | None = None
    ) -> str:
        del duration_seconds  # whisper.cpp handles any length natively
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


class CTCTranscriber(Transcriber):
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

        from .install import load_or_install

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

    def ensure_downloaded(self) -> None:
        # transformers couples download and load via from_pretrained, so the
        # up-front prep step loads here too.
        self.ensure_loaded()

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


class SherpaTranscriber(Transcriber):
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
        self._model_files: tuple[str, str, str, str, str] | None = None

    def _provider(self) -> str:
        text = self.device.strip().lower()
        # The Nix onnxruntime exposes only the CPU provider here, and a CPU pass
        # overlaps the GPU primary; only honor an explicit CUDA request.
        return "cuda" if text.startswith("cuda") else "cpu"

    def _resolve_model_files(self) -> tuple[str, str, str, str, str]:
        if self._model_files is not None:
            return self._model_files
        from huggingface_hub import hf_hub_download, snapshot_download

        from .install import load_or_install

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
        self._model_files = (encoder, decoder, joiner, tokens, vad_path)
        return self._model_files

    def ensure_downloaded(self) -> None:
        """Resolve/download the model bundle (consent-gated) without the heavy
        recognizer construction — that stays lazy so it can overlap normalization."""
        self._resolve_model_files()

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


class PocketSphinxTranscriber(Transcriber):
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

    def ensure_downloaded(self) -> None:
        # The English model ships with pocketsphinx; nothing to download, but the
        # prep step still resolves it (and surfaces a clear error if it is absent).
        self.ensure_loaded()

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


OPENROUTER_ASR_SYSTEM_PROMPT = (
    "You are a verbatim speech-to-text transcriber. Transcribe the spoken words in the "
    "audio exactly as said. Output only the transcript text — no preamble, no headings, "
    "no timestamps, no commentary. Transcribe only words that are actually spoken: if a "
    "stretch is silence or non-speech background noise, skip it rather than inventing "
    "words to fill it."
)


def _openrouter_asr_user_text(language: str) -> str:
    text = "Transcribe this audio recording verbatim."
    language = (language or "").strip()
    if language and language.lower() not in {"auto", "und", ""}:
        text += f" The speech is primarily in language code '{language}'."
    return text


class OpenRouterTranscriber(Transcriber):
    """Whole-file ASR via an OpenRouter audio-LLM (the chat ``input_audio`` path).

    The dedicated OpenRouter transcription models reject a long single request, but an
    audio-LLM such as Gemini 3 Flash transcribes an entire long recording in one chat
    call — and, given the whole file at once, annotates non-speech instead of
    confabulating it (the failure mode of per-chunk audio-LLM transcription). The whole
    file is transcoded to mono mp3 (a long 16 kHz wav is too large to base64 into a JSON
    body), sent as ``input_audio``, and the reply is the transcript. Nothing is
    downloaded; the only requirement is the API key. Failures are non-fatal — the source
    simply produces no transcript and its peers (local whisper+sherpa) still run.
    """

    def __init__(
        self,
        *,
        model_name: str,
        api_base: str,
        auth_env: str,
        timeout: float,
        mp3_sample_rate: int,
        max_output_tokens: int,
    ) -> None:
        self.model_name = model_name
        self.api_base = api_base
        self.auth_env = auth_env
        self.timeout = timeout
        self.mp3_sample_rate = mp3_sample_rate
        self.max_output_tokens = max_output_tokens

    def ensure_downloaded(self) -> None:
        """No model to fetch; fail early and clearly if the API key is missing."""
        if not os.environ.get(self.auth_env, "").strip():
            raise RuntimeError(
                f"{self.auth_env} is not set; it is required for the 'openrouter' ASR backend"
            )

    def _effective_timeout(self, duration_seconds: float | None) -> float:
        """Floor for short clips; grow the ceiling with audio length (the request
        itself returns in well under a minute, this only bounds how long we wait)."""
        if not duration_seconds:
            return self.timeout
        return min(900.0, max(self.timeout, 90.0 + duration_seconds * 0.25))

    def transcribe_file(
        self, path: Path, language: str, *, duration_seconds: float | None = None
    ) -> str:
        # Whole file in one request; length is handled by the API, not chunking.
        from .audio import encode_to_mp3
        from .chat import ChatClient

        with tempfile.TemporaryDirectory(prefix="speech-note-or-asr-") as tmp_dir:
            mp3_path = Path(tmp_dir) / "audio.mp3"
            encode_to_mp3(path, mp3_path, sample_rate=self.mp3_sample_rate)
            data = base64.b64encode(mp3_path.read_bytes()).decode("ascii")
        timeout = self._effective_timeout(duration_seconds)
        client = ChatClient(
            api_base=self.api_base,
            models=[self.model_name],
            timeout=timeout,
            auth_env=self.auth_env,
        )
        messages: list[Any] = [
            {"role": "system", "content": OPENROUTER_ASR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _openrouter_asr_user_text(language)},
                    {"type": "input_audio", "input_audio": {"data": data, "format": "mp3"}},
                ],
            },
        ]
        response = client.chat(messages, max_tokens=self.max_output_tokens, timeout=timeout)
        return normalize_spacing(response.content)


class OpenRouterSttTranscriber(Transcriber):
    """Whole-file ASR via OpenRouter's dedicated /audio/transcriptions endpoint.

    Unlike the audio-LLM path (OpenRouterTranscriber) this talks to a real speech
    recognizer, which matters for more than accuracy: on unintelligible audio a
    dedicated model emits garbled text, whereas an audio-LLM emits fluent invented
    prose that nothing downstream can distinguish from real speech. See the measurement
    notes on OPENROUTER_STT_MODEL.

    The recording is transcoded to mono mp3 and uploaded as multipart form-data — not
    base64 in a JSON body, which a long recording overflows (a 65-minute file is ~17 MB
    of mp3, ~23 MB base64, and the gateway 502s on that). Nothing is downloaded; the only
    requirement is the API key. Failures are non-fatal: the source produces no transcript
    and its peers still run.
    """

    def __init__(
        self,
        *,
        model_name: str,
        api_base: str,
        auth_env: str,
        timeout: float,
        mp3_sample_rate: int,
        response_format: str,
    ) -> None:
        self.model_name = model_name
        self.api_base = api_base
        self.auth_env = auth_env
        self.timeout = timeout
        self.mp3_sample_rate = mp3_sample_rate
        self.response_format = response_format

    def ensure_downloaded(self) -> None:
        """No model to fetch; fail early and clearly if the API key is missing."""
        if not os.environ.get(self.auth_env, "").strip():
            raise RuntimeError(
                f"{self.auth_env} is not set; it is required for the 'openrouter-stt' backend"
            )

    def _effective_timeout(self, duration_seconds: float | None) -> float:
        """Floor for short clips, growing with audio length. Generous because this only
        bounds how long we wait — the endpoint returned a 65-minute file in ~16 s."""
        if not duration_seconds:
            return self.timeout
        return min(1800.0, max(self.timeout, 90.0 + duration_seconds * 0.25))

    def transcribe_file(
        self, path: Path, language: str, *, duration_seconds: float | None = None
    ) -> str:
        import requests

        from .audio import encode_to_mp3

        api_key = os.environ.get(self.auth_env, "").strip()
        if not api_key:
            raise RuntimeError(f"{self.auth_env} is required for the 'openrouter-stt' backend")
        timeout = self._effective_timeout(duration_seconds)
        with tempfile.TemporaryDirectory(prefix="speech-note-or-stt-") as tmp_dir:
            mp3_path = Path(tmp_dir) / "audio.mp3"
            encode_to_mp3(path, mp3_path, sample_rate=self.mp3_sample_rate)
            data: dict[str, str] = {
                "model": self.model_name,
                "response_format": self.response_format,
            }
            if language and language.lower() not in {"auto", "und"}:
                data["language"] = language
            with mp3_path.open("rb") as handle:
                response = requests.post(
                    self.api_base,
                    headers={"Authorization": f"Bearer {api_key}"},
                    files={"file": ("audio.mp3", handle, "audio/mpeg")},
                    data=data,
                    timeout=timeout,
                )
        if not response.ok:
            detail = " ".join(response.text.split())[:400]
            raise RuntimeError(f"{self.model_name}: HTTP {response.status_code} {detail}")
        try:
            payload = response.json()
        except ValueError:
            raise RuntimeError(
                f"{self.model_name}: HTTP 200 with non-JSON body: "
                f"{' '.join(response.text.split())[:200]}"
            ) from None
        # A 200 can still carry a provider error and no transcript; say so rather than
        # returning "" (which the collection would report as "no speech detected").
        text = payload.get("text")
        if not isinstance(text, str):
            raise RuntimeError(
                f"{self.model_name}: response carried no transcript "
                f"({json.dumps(payload.get('error', payload))[:200]})"
            )
        return normalize_paragraphs(text)


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
