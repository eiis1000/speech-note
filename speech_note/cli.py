"""Argument parsing and config resolution.

parse_args produces a raw namespace where "user didn't say" is None; resolve_config
turns that into an immutable Config with every default applied exactly once.
After resolution nothing mutates the config — archive mode derives a new one.
"""

from __future__ import annotations

import argparse
import atexit
import dataclasses
import os
import shutil
import sys
import tempfile
from pathlib import Path

from . import config as defaults
from .devices import coerce_input_device, print_input_devices, select_input_device
from .terminal import read_single_choice


@dataclasses.dataclass(frozen=True)
class Config:
    # capture
    sample_rate: int
    frame_ms: int
    vad_mode: int
    start_padding_ms: int
    end_silence_ms: int
    min_speech_ms: int
    max_segment_seconds: float
    input_device: int | str | None
    replay_speed: float
    # inputs
    input_file: Path | None
    input_archive: Path | None
    replay_input_file: Path | None
    dry_run_text: str | None
    primary_transcript: Path | None
    secondary_transcript: Path | None
    extra_transcripts: tuple[Path, ...]
    # outputs
    output: Path | None
    artifacts_dir: Path
    archive_dir: Path
    full_auto: bool
    # primary ASR
    asr_backend: str
    asr_model: str
    asr_device: str
    asr_compute_type: str
    asr_cpu_threads: int
    live_asr_cpu_threads: int
    whisper_cpp_binary: Path | None
    whisper_cpp_model: Path | None
    whisper_cpp_device: int
    whisper_cpp_gpu: bool
    # secondary ASR
    secondary_asr_enabled: bool
    secondary_asr_backend: str
    secondary_asr_model: str
    secondary_asr_device: str
    secondary_asr_strip_times: bool
    secondary_only: bool
    # shared ASR
    language: str
    auto_download: bool
    download_root: Path | None
    parallel_final_asr: bool | None  # None = auto (parallel when devices differ)
    # organizer
    organizer_mode: str
    organizer_provider: str
    organizer_api_base: str
    organizer_models: tuple[str, ...]
    organizer_auth_env: str | None
    organizer_timeout: float
    organizer_context_tokens: int
    organizer_max_output_tokens: int
    organizer_gpu_layers: str
    organizer_kv_offload: bool
    organizer_gguf: Path | None
    organizer_server_command: tuple[str, ...] | None

    def with_archive_contents(self, audio_path: Path, transcript_paths: list[Path]) -> "Config":
        derived = dataclasses.replace(
            self,
            input_archive=None,
            input_file=audio_path,
            extra_transcripts=self.extra_transcripts + tuple(transcript_paths),
        )
        if derived.full_auto and derived.output is None:
            from .naming import full_auto_output_path

            # Name from the archive (zip stem + source tags), not the internal
            # audio filename: derive the name from a config that still knows it
            # came from an archive and carries the bundled transcripts.
            naming_config = dataclasses.replace(
                self, extra_transcripts=self.extra_transcripts + tuple(transcript_paths)
            )
            derived = dataclasses.replace(
                derived, output=full_auto_output_path(naming_config, Path.cwd())
            )
        return derived


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="speech-note",
        description="Capture or load audio, transcribe locally, clean the transcript with an LM.",
    )
    # capture
    parser.add_argument("--sample-rate", type=int, default=defaults.SAMPLE_RATE)
    parser.add_argument("--frame-ms", type=int, choices=[10, 20, 30], default=defaults.FRAME_MS)
    parser.add_argument("--vad-mode", type=int, choices=[0, 1, 2, 3], default=1)
    parser.add_argument("--start-padding-ms", type=int, default=500)
    parser.add_argument("--end-silence-ms", type=int, default=1200)
    parser.add_argument("--min-speech-ms", type=int, default=200)
    parser.add_argument("--max-segment-seconds", type=float, default=12.0)
    parser.add_argument("--input-device", default=None)
    parser.add_argument("--replay-speed", type=float, default=1.0)
    # discovery / one-shot modes
    parser.add_argument("--list-input-devices", action="store_true")
    parser.add_argument("--select-mic", action="store_true")
    # inputs
    parser.add_argument("--input-audio", dest="input_file", type=Path, default=None,
                        help="Audio file to transcribe (m4a/mp3/wav/...).")
    parser.add_argument(
        "--input-archive",
        type=Path,
        default=None,
        help="Zip containing one recording plus transcript file(s).",
    )
    parser.add_argument(
        "--replay-input-file",
        type=Path,
        default=None,
        help="Feed an audio file through the live capture path as a virtual mic.",
    )
    parser.add_argument(
        "--input-text",
        dest="dry_run_text",
        default=None,
        help="Bypass mic and ASR; feed text (chunks split by ||) to the cleanup stage.",
    )
    parser.add_argument(
        "--primary-transcript",
        type=Path,
        default=None,
        help="Use this file as the primary transcript and skip ASR.",
    )
    parser.add_argument(
        "--secondary-transcript",
        type=Path,
        default=None,
        help="Use this file as the secondary transcript (requires --primary-transcript).",
    )
    parser.add_argument(
        "--extra-transcript",
        type=Path,
        action="append",
        default=[],
        help="Additional transcript for the cleanup stage. Repeatable.",
    )
    # outputs
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Also write the cleaned transcript to this path.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("."),
        help="Directory for raw.latest / clean.latest / diagnostics and the logs/ archive.",
    )
    parser.add_argument(
        "--full-auto",
        action="store_true",
        help=(
            "Non-interactive: write only an auto-named cleaned transcript to the current "
            "directory (plus a diagnostics file on errors); keep other artifacts in a "
            "temporary directory."
        ),
    )
    # primary ASR
    parser.add_argument(
        "--asr-backend",
        choices=["faster-whisper", "whisper-cpp"],
        default=defaults.DEFAULT_ASR_BACKEND,
    )
    parser.add_argument("--asr-model", default=defaults.DEFAULT_ASR_MODEL)
    parser.add_argument("--asr-device", default="cpu")
    parser.add_argument("--asr-compute-type", default="int8")
    parser.add_argument("--asr-cpu-threads", type=int, default=defaults.DEFAULT_ASR_CPU_THREADS)
    parser.add_argument("--live-asr-cpu-threads", type=int, default=2)
    parser.add_argument("--whisper-cpp-binary", type=Path, default=None)
    parser.add_argument("--whisper-cpp-model", type=Path, default=None)
    parser.add_argument(
        "--whisper-cpp-device",
        default="0",
        help="Whisper.cpp device: 'cpu', or a GPU index (default 0 = first GPU).",
    )
    # secondary ASR
    parser.add_argument(
        "--secondary-asr", dest="secondary_asr_enabled",
        action=argparse.BooleanOptionalAction, default=True,
        help="Run a secondary ASR pass alongside Whisper (default: on).",
    )
    parser.add_argument(
        "--secondary-asr-backend",
        choices=sorted(defaults.SECONDARY_BACKENDS),
        default=defaults.DEFAULT_SECONDARY_ASR_BACKEND,
    )
    parser.add_argument(
        "--secondary-asr-model",
        default=None,
        help="Defaults to the standard model for the chosen backend.",
    )
    parser.add_argument(
        "--secondary-asr-device",
        default=None,
        help=(
            "Device for in-process secondary backends. Default: cpu for sherpa and ctc "
            "(so they overlap the GPU primary pass and avoid ROCm init overhead), auto otherwise."
        ),
    )
    parser.add_argument(
        "--secondary-asr-strip-times", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--secondary-only",
        action="store_true",
        help="Run only the secondary ASR backend on --input-audio and print its transcript.",
    )
    # shared ASR
    parser.add_argument("--language", default="en")
    parser.add_argument(
        "--auto-download",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow model loaders to download missing files (default: local caches only).",
    )
    parser.add_argument("--download-root", type=Path, default=None)
    parser.add_argument(
        "--parallel-asr",
        dest="parallel_final_asr",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Default: automatic — parallel when primary and secondary use different devices.",
    )
    # connectivity presets: bundle organizer provider + model list. ASR stays local
    # in every mode (no usable OpenRouter ASR beats local whisper+sherpa). Explicit
    # --organizer-provider / --organizer-model still override. Work with/without --full-auto.
    connectivity = parser.add_mutually_exclusive_group()
    connectivity.add_argument(
        "--offline", dest="connectivity", action="store_const", const="offline",
        help="Local cleanup (bundled GGUF) + local ASR; nothing leaves the machine.",
    )
    connectivity.add_argument(
        "--online-free", dest="connectivity", action="store_const", const="online-free",
        help="OpenRouter cleanup with free models (may log/train on inputs); local ASR.",
    )
    connectivity.add_argument(
        "--online-paid", dest="connectivity", action="store_const", const="online-paid",
        help="OpenRouter cleanup with paid models (deepseek-v3.2 / gemini-3-flash, not "
             "logged); local ASR (no usable paid ASR beats local). Needs OPENROUTER_API_KEY.",
    )
    parser.set_defaults(connectivity=None)
    # organizer
    parser.add_argument("--organizer-mode", choices=["llama", "heuristic", "off"], default="llama")
    parser.add_argument(
        "--organizer-provider",
        choices=["local", "openrouter"],
        default=defaults.DEFAULT_ORGANIZER_PROVIDER,
    )
    parser.add_argument("--organizer-api-base", default=None)
    parser.add_argument(
        "--organizer-model",
        default=None,
        help="Model name, or a comma-separated fallback list.",
    )
    parser.add_argument("--organizer-timeout", type=float, default=12.0)
    parser.add_argument("--organizer-context-tokens", type=int, default=None)
    parser.add_argument("--organizer-max-output-tokens", type=int, default=None)
    parser.add_argument("--organizer-gpu-layers", default="auto")
    parser.add_argument(
        "--organizer-kv-offload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Offload the cleanup LM's KV cache to the GPU (default: on; ~1.5x faster "
            "generation). --no-organizer-kv-offload restores the workaround for older "
            "llama.cpp/Vulkan stacks that hang during slot init."
        ),
    )
    parser.add_argument(
        "--organizer-gguf",
        type=Path,
        default=None,
        help=(
            "Path to a local cleanup-LM GGUF, overriding the bundled gemma-4-E2B. Use a "
            "stronger model if your hardware allows — required to clean up the looping that "
            "--asr-model large-v3-turbo-q5_k tends to produce (gemma-E2B is too small for it). "
            "Ignored when --organizer-server-command is given."
        ),
    )
    parser.add_argument(
        "--organizer-server-command",
        nargs="*",
        default=None,
        help="Command to launch a local OpenAI-compatible server (default: bundled llama-server).",
    )
    return parser.parse_args(argv)


def _parse_whisper_cpp_device(raw: str) -> tuple[bool, int]:
    """Map the --whisper-cpp-device string to (use_gpu, gpu_index).

    'cpu'/'none' disables the GPU; an integer selects that GPU index (and enables
    the GPU). Defaults to GPU index 0.
    """
    text = raw.strip().lower()
    if text in ("cpu", "none", ""):
        return False, 0
    try:
        return True, int(text)
    except ValueError:
        raise SystemExit(
            f"--whisper-cpp-device must be 'cpu' or a GPU index, got {raw!r}"
        ) from None


def resolve_config(args: argparse.Namespace) -> Config:
    # Connectivity preset: bundle organizer provider + default model list. An explicit
    # --organizer-provider/--organizer-model still wins (handled below / in the openrouter
    # branch). ASR is untouched — every mode uses local whisper+sherpa.
    connectivity = getattr(args, "connectivity", None)
    if connectivity == "offline":
        args.organizer_provider = "local"
    elif connectivity in ("online-free", "online-paid"):
        args.organizer_provider = "openrouter"

    secondary_model = args.secondary_asr_model
    if secondary_model is None:
        secondary_model = defaults.SECONDARY_DEFAULT_MODELS[args.secondary_asr_backend]

    secondary_device = args.secondary_asr_device
    if secondary_device is None:
        # sherpa and CTC run fastest on the CPU here: "auto" would land them on the
        # ROCm GPU, which shares the iGPU with the Vulkan primary (so they serialize)
        # and pays heavy init overhead. CPU is faster and overlaps the GPU primary.
        cpu_default_backends = {"sherpa", "ctc"}
        secondary_device = (
            "cpu"
            if args.secondary_asr_backend in cpu_default_backends
            else defaults.DEFAULT_SECONDARY_ASR_DEVICE
        )

    whisper_cpp_gpu, whisper_cpp_device = _parse_whisper_cpp_device(args.whisper_cpp_device)

    if args.organizer_provider == "openrouter":
        api_base = args.organizer_api_base or defaults.DEFAULT_OPENROUTER_API_BASE
        if connectivity == "online-paid":
            mode_models = defaults.OPENROUTER_PAID_MODELS
        elif connectivity == "online-free":
            mode_models = defaults.OPENROUTER_FREE_MODELS
        else:
            mode_models = defaults.OPENROUTER_PREFERRED_MODELS
        models = (
            tuple(part.strip() for part in args.organizer_model.split(",") if part.strip())
            if args.organizer_model
            else tuple(mode_models)
        )
        auth_env: str | None = defaults.OPENROUTER_API_KEY_ENV
        context_tokens = args.organizer_context_tokens or defaults.DEFAULT_OPENROUTER_CONTEXT_TOKENS
        max_output_tokens = (
            args.organizer_max_output_tokens or defaults.DEFAULT_OPENROUTER_MAX_OUTPUT_TOKENS
        )
    else:
        api_base = args.organizer_api_base or defaults.DEFAULT_LOCAL_API_BASE
        models = (
            tuple(part.strip() for part in args.organizer_model.split(",") if part.strip())
            if args.organizer_model
            else (defaults.DEFAULT_LOCAL_MODEL_LABEL,)
        )
        auth_env = None
        context_tokens = args.organizer_context_tokens or defaults.DEFAULT_ORGANIZER_CONTEXT_TOKENS
        max_output_tokens = (
            args.organizer_max_output_tokens or defaults.DEFAULT_ORGANIZER_MAX_OUTPUT_TOKENS
        )

    artifacts_dir: Path = args.artifacts_dir
    output: Path | None = args.output
    if args.full_auto:
        scratch = Path(tempfile.mkdtemp(prefix="speech-note-full-auto-"))
        atexit.register(shutil.rmtree, scratch, ignore_errors=True)
        artifacts_dir = scratch

    config = Config(
        sample_rate=args.sample_rate,
        frame_ms=args.frame_ms,
        vad_mode=args.vad_mode,
        start_padding_ms=args.start_padding_ms,
        end_silence_ms=args.end_silence_ms,
        min_speech_ms=args.min_speech_ms,
        max_segment_seconds=args.max_segment_seconds,
        input_device=coerce_input_device(args.input_device),
        replay_speed=args.replay_speed,
        input_file=args.input_file,
        input_archive=args.input_archive,
        replay_input_file=args.replay_input_file,
        dry_run_text=args.dry_run_text,
        primary_transcript=args.primary_transcript,
        secondary_transcript=args.secondary_transcript,
        extra_transcripts=tuple(args.extra_transcript),
        output=output,
        artifacts_dir=artifacts_dir,
        archive_dir=artifacts_dir / "logs",
        full_auto=args.full_auto,
        asr_backend=args.asr_backend,
        asr_model=args.asr_model,
        asr_device=args.asr_device,
        asr_compute_type=args.asr_compute_type,
        asr_cpu_threads=args.asr_cpu_threads,
        live_asr_cpu_threads=args.live_asr_cpu_threads,
        whisper_cpp_binary=args.whisper_cpp_binary,
        whisper_cpp_model=args.whisper_cpp_model,
        whisper_cpp_device=whisper_cpp_device,
        whisper_cpp_gpu=whisper_cpp_gpu,
        secondary_asr_enabled=args.secondary_asr_enabled,
        secondary_asr_backend=args.secondary_asr_backend,
        secondary_asr_model=secondary_model,
        secondary_asr_device=secondary_device,
        secondary_asr_strip_times=args.secondary_asr_strip_times,
        secondary_only=args.secondary_only,
        language=args.language,
        auto_download=args.auto_download,
        download_root=args.download_root,
        parallel_final_asr=args.parallel_final_asr,
        organizer_mode=args.organizer_mode,
        organizer_provider=args.organizer_provider,
        organizer_api_base=api_base,
        organizer_models=models,
        organizer_auth_env=auth_env,
        organizer_timeout=args.organizer_timeout,
        organizer_context_tokens=context_tokens,
        organizer_max_output_tokens=max_output_tokens,
        organizer_gpu_layers=args.organizer_gpu_layers,
        organizer_kv_offload=args.organizer_kv_offload,
        organizer_gguf=args.organizer_gguf,
        organizer_server_command=(
            tuple(args.organizer_server_command)
            if args.organizer_server_command is not None
            else None
        ),
    )

    if config.full_auto and config.output is None and config.input_archive is None:
        from .naming import full_auto_output_path

        config = dataclasses.replace(config, output=full_auto_output_path(config, Path.cwd()))
    return config


def validate(config: Config) -> None:
    inputs = [
        config.input_file,
        config.input_archive,
        config.replay_input_file,
        config.dry_run_text,
        config.primary_transcript,
    ]
    if sum(value is not None for value in inputs) > 1:
        raise SystemExit(
            "--input-audio, --input-archive, --replay-input-file, --input-text and "
            "--primary-transcript are mutually exclusive"
        )
    if config.secondary_transcript is not None and config.primary_transcript is None:
        raise SystemExit("--secondary-transcript requires --primary-transcript")
    if config.secondary_only and config.input_file is None:
        raise SystemExit("--secondary-only requires --input-audio")
    if config.full_auto and all(value is None for value in inputs) and config.input_device is None:
        raise SystemExit(
            "--full-auto needs --input-audio, --input-archive, --replay-input-file, "
            "--input-text, --primary-transcript, or an explicit --input-device"
        )
    if (
        config.organizer_mode == "llama"
        and config.organizer_auth_env is not None
        and not os.environ.get(config.organizer_auth_env, "").strip()
    ):
        raise SystemExit(
            f"{config.organizer_auth_env} is not set; it is required for "
            f"--organizer-provider {config.organizer_provider}"
        )


def _interactive_capture_setup(args: argparse.Namespace) -> None:
    """Terminal-dictation prompts: cleanup provider, then microphone."""
    if (
        args.organizer_mode == "llama"
        and args.organizer_provider == defaults.DEFAULT_ORGANIZER_PROVIDER
        and sys.stdin
        and sys.stdin.isatty()
    ):
        print("Select cleanup model:")
        print("[1] local model [default]")
        print("[2] OpenRouter cleanup models")
        choice = read_single_choice(
            {"1", "2"},
            "Press key for cleanup model. Enter selects local. Ctrl-C cancels.",
            default_key="1",
        )
        if choice == "2":
            args.organizer_provider = "openrouter"
    args.input_device = select_input_device()


def main(argv: list[str] | None = None) -> int:
    # Load ~/.config/speech-note/env first so OPENROUTER_API_KEY (etc.) is available no
    # matter which directory speech-note runs from — a shell export still takes priority.
    defaults.load_user_env()
    args = parse_args(argv)
    if args.list_input_devices:
        print_input_devices()
        return 0
    if args.select_mic:
        selected = select_input_device()
        if selected is not None:
            print(selected)
        return 0

    from .hardware import warn_on_unsupported_gpu

    warn_on_unsupported_gpu()

    capture_mode = all(
        getattr(args, name) is None
        for name in ("input_file", "input_archive", "replay_input_file", "dry_run_text", "primary_transcript")
    )
    if capture_mode and args.input_device is None and not args.full_auto:
        _interactive_capture_setup(args)

    config = resolve_config(args)
    validate(config)

    from . import pipeline

    if config.secondary_only:
        session = pipeline.run_secondary_only(config)
    elif config.primary_transcript is not None:
        session = pipeline.run_transcript_pipeline(config)
    elif config.input_archive is not None:
        session = pipeline.run_archive_pipeline(config)
    elif config.input_file is not None:
        session = pipeline.run_file_pipeline(config)
    elif config.dry_run_text is not None:
        session = pipeline.run_dry_text_pipeline(config)
    else:
        from .capture import run_capture_pipeline

        session = run_capture_pipeline(config)
    return 1 if session.run_failed else 0
