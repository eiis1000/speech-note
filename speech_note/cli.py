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
import zipfile
from pathlib import Path

from . import __version__
from . import config as defaults
from .config import AsrSource
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
    extra_transcripts: tuple[Path, ...]
    no_asr: bool
    # outputs
    output: Path | None
    export_sources: Path | None
    artifacts_dir: Path
    archive_dir: Path
    full_auto: bool
    # ASR: an ordered collection of sources, all fed to the cleanup LM as peers.
    asr_sources: tuple[AsrSource, ...]
    asr_compute_type: str  # faster-whisper compute type (applies to faster-whisper sources)
    asr_cpu_threads: int
    live_asr_cpu_threads: int
    whisper_cpp_binary: Path | None
    whisper_cpp_model: Path | None  # explicit ggml path, applied to whisper-cpp sources
    strip_asr_timestamps: bool
    language: str
    auto_download: bool
    download_root: Path | None
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
    organizer_prewarm: bool

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
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    # capture
    parser.add_argument("--sample-rate", type=int, default=defaults.SAMPLE_RATE)
    parser.add_argument("--frame-ms", type=int, choices=[10, 20, 30], default=defaults.FRAME_MS)
    parser.add_argument("--vad-mode", type=int, choices=[0, 1, 2, 3], default=1)
    parser.add_argument("--start-padding-ms", type=int, default=500)
    parser.add_argument("--end-silence-ms", type=int, default=1200)
    parser.add_argument("--min-speech-ms", type=int, default=200)
    parser.add_argument("--max-segment-seconds", type=float, default=12.0)
    parser.add_argument("-d", "--input-device", default=None)
    parser.add_argument("--replay-speed", type=float, default=1.0)
    # discovery / one-shot modes
    parser.add_argument("--list-input-devices", action="store_true")
    parser.add_argument("-m", "--select-mic", action="store_true")
    # inputs
    parser.add_argument(
        "-i", "--input",
        dest="input",
        type=Path,
        default=None,
        help=(
            "Audio file (m4a/mp3/wav/...) OR an archive zip (recording + transcript "
            "file(s)) to process. The kind is detected from the file, so one flag covers "
            "both."
        ),
    )
    parser.add_argument(
        "--replay-input-file",
        type=Path,
        default=None,
        help="Feed an audio file through the live capture path as a virtual mic.",
    )
    parser.add_argument(
        "-t", "--input-text",
        dest="dry_run_text",
        default=None,
        help="Bypass mic and ASR; feed text (chunks split by ||) to the cleanup stage.",
    )
    parser.add_argument(
        "--no-asr",
        dest="no_asr",
        action="store_true",
        help=(
            "Skip transcription and clean up only the transcripts you provide "
            "(--extra-transcript, or those bundled in an archive). With no audio at all, "
            "--extra-transcript files are cleaned up directly, so this flag only matters "
            "when audio IS present and you want to reuse existing transcripts instead of "
            "re-running ASR. All transcripts are equal peers — there is no primary/secondary."
        ),
    )
    parser.add_argument(
        "-x", "--extra-transcript",
        type=Path,
        action="append",
        default=[],
        help="A transcript file fed to the cleanup stage as an equal peer source. Repeatable.",
    )
    # outputs
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Also write the cleaned transcript to this path.",
    )
    parser.add_argument(
        "--export-sources",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Also write every transcript fed to the cleanup LM — each ASR pass and each "
            "--extra-transcript — as its own file in DIR (NN-<source>.txt), plus the "
            "cleaned result as clean.txt. Lets you compare what each source heard, or "
            "reuse a single source later via --extra-transcript. Works in every mode."
        ),
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("."),
        help="Directory for raw.latest / clean.latest / diagnostics and the logs/ archive.",
    )
    parser.add_argument(
        "-f", "--full-auto",
        action="store_true",
        help=(
            "Non-interactive: write only an auto-named cleaned transcript to the current "
            "directory (plus a diagnostics file on errors); keep other artifacts in a "
            "temporary directory."
        ),
    )
    # ASR collection
    parser.add_argument(
        "-a", "--asr",
        action="append",
        default=None,
        metavar="BACKEND[:MODEL][@DEVICE]",
        help=(
            "An ASR source to add to the collection. Repeatable, and each value may be a "
            "comma-separated list, e.g. --asr whisper-cpp:medium-q8_0@gpu --asr sherpa, or "
            "--asr 'whisper-cpp,ctc@cpu'. Model and device are optional (backend defaults "
            "apply). Backends: " + ", ".join(sorted(defaults.ASR_BACKENDS)) + ". "
            "If omitted, the collection comes from the user config file "
            f"({defaults.USER_ASR_FILE}) or the built-in default (whisper-cpp + sherpa). "
            "All sources are fed to the cleanup LM as peers; order is only a soft preference "
            "for which transcript is the raw fallback."
        ),
    )
    parser.add_argument("--asr-compute-type", default="int8")
    parser.add_argument("--asr-cpu-threads", type=int, default=defaults.DEFAULT_ASR_CPU_THREADS)
    parser.add_argument("--live-asr-cpu-threads", type=int, default=2)
    parser.add_argument("--whisper-cpp-binary", type=Path, default=None)
    parser.add_argument(
        "--whisper-cpp-model", type=Path, default=None,
        help="Explicit ggml model file, applied to whisper-cpp sources.",
    )
    parser.add_argument(
        "--strip-asr-timestamps", action=argparse.BooleanOptionalAction, default=True,
        help="Strip Parakeet [hh:mm:ss] timestamps from ASR output (default: on).",
    )
    parser.add_argument("-l", "--language", default="en")
    parser.add_argument(
        "--auto-download",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow model loaders to download missing files (default: local caches only).",
    )
    parser.add_argument("--download-root", type=Path, default=None)
    # connectivity presets: bundle organizer provider + model list. ASR stays local
    # in every mode (no usable OpenRouter ASR beats local whisper+sherpa). Explicit
    # --organizer-provider / --organizer-model still override. Work with/without --full-auto.
    connectivity = parser.add_mutually_exclusive_group()
    connectivity.add_argument(
        "-O", "--offline", dest="connectivity", action="store_const", const="offline",
        help="Local cleanup (bundled GGUF) + local ASR; nothing leaves the machine.",
    )
    connectivity.add_argument(
        "-F", "--online-free", dest="connectivity", action="store_const", const="online-free",
        help="OpenRouter cleanup with free models (may log/train on inputs); local ASR.",
    )
    connectivity.add_argument(
        "-P", "--online-paid", dest="connectivity", action="store_const", const="online-paid",
        help="OpenRouter cleanup with paid models (deepseek-v3.2 / gemini-3-flash, not "
             "logged) AND a Gemini full-file ASR source added by default, alongside local "
             "whisper+sherpa. Needs OPENROUTER_API_KEY.",
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
    # Undocumented: skip the background prewarm so the local cleanup LM is started
    # only after ASR completes, instead of loading concurrently. On a shared iGPU
    # (Vulkan whisper + Vulkan llama-server) the LM's slot init can stall while ASR
    # holds the GPU; sequencing them avoids that contention at the cost of the
    # prewarm overlap. Left as an escape hatch; default keeps the prewarm on.
    parser.add_argument(
        "--organizer-no-prewarm",
        dest="organizer_prewarm",
        action="store_false",
        default=True,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def resolve_asr_sources(
    args: argparse.Namespace,
    default_sources: tuple[AsrSource, ...] = defaults.DEFAULT_ASR_SOURCES,
) -> tuple[AsrSource, ...]:
    """The ASR collection: --asr (CLI) > user config file > the given default.

    default_sources lets a connectivity preset supply its own default collection
    (e.g. --online-paid leads with the OpenRouter Gemini source) while an explicit
    --asr or the user ASR config file still override it.
    """
    if args.asr:
        try:
            return defaults.parse_asr_sources(args.asr)
        except ValueError as exc:
            raise SystemExit(f"--asr: {exc}") from None
    from_file = defaults.load_user_asr_sources()
    if from_file:
        return from_file
    return tuple(source.resolved() for source in default_sources)


def _looks_like_archive(path: Path) -> bool:
    """A zip → archive pipeline; anything else → audio file. Detect by content when the
    file exists (handles a misnamed zip), else fall back to the .zip extension."""
    if path.suffix.lower() == ".zip":
        return True
    return path.exists() and zipfile.is_zipfile(path)


def fold_unified_input(args: argparse.Namespace) -> None:
    """Split the unified -i/--input into the internal input_file / input_archive by
    detected type. These two are an implementation detail of the pipeline dispatch,
    not CLI flags, so they are derived here rather than parsed."""
    chosen = getattr(args, "input", None)
    args.input_file = None
    args.input_archive = None
    if chosen is None:
        return
    if _looks_like_archive(chosen):
        args.input_archive = chosen
    else:
        args.input_file = chosen


def resolve_config(args: argparse.Namespace) -> Config:
    fold_unified_input(args)
    # Connectivity preset: bundle organizer provider + default model list. An explicit
    # --organizer-provider/--organizer-model still wins (handled below / in the openrouter
    # branch). ASR is untouched — every mode uses local whisper+sherpa.
    connectivity = getattr(args, "connectivity", None)
    if connectivity == "offline":
        args.organizer_provider = "local"
    elif connectivity in ("online-free", "online-paid"):
        args.organizer_provider = "openrouter"

    # --online-paid also changes the *default* ASR collection (an audio-LLM can
    # transcribe a whole long recording in one request, unlike the chunk-only local
    # backends); an explicit --asr or the user ASR file still wins. Other modes keep
    # the local whisper+sherpa default.
    default_asr = (
        defaults.ONLINE_PAID_ASR_SOURCES
        if connectivity == "online-paid"
        else defaults.DEFAULT_ASR_SOURCES
    )
    asr_sources = resolve_asr_sources(args, default_asr)

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
        extra_transcripts=tuple(args.extra_transcript),
        no_asr=args.no_asr,
        output=output,
        export_sources=args.export_sources,
        artifacts_dir=artifacts_dir,
        archive_dir=artifacts_dir / "logs",
        full_auto=args.full_auto,
        asr_sources=asr_sources,
        asr_compute_type=args.asr_compute_type,
        asr_cpu_threads=args.asr_cpu_threads,
        live_asr_cpu_threads=args.live_asr_cpu_threads,
        whisper_cpp_binary=args.whisper_cpp_binary,
        whisper_cpp_model=args.whisper_cpp_model,
        strip_asr_timestamps=args.strip_asr_timestamps,
        language=args.language,
        auto_download=args.auto_download,
        download_root=args.download_root,
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
        organizer_prewarm=args.organizer_prewarm,
    )

    if config.full_auto and config.output is None and config.input_archive is None:
        from .naming import full_auto_output_path

        config = dataclasses.replace(config, output=full_auto_output_path(config, Path.cwd()))
    return config


def validate(config: Config) -> None:
    # The single-audio/text inputs are mutually exclusive; --extra-transcript is not —
    # it supplements ASR, or (with no audio) is the input on its own.
    inputs = [
        config.input_file,
        config.input_archive,
        config.replay_input_file,
        config.dry_run_text,
    ]
    if sum(value is not None for value in inputs) > 1:
        raise SystemExit(
            "--input, --replay-input-file and --input-text are mutually exclusive"
        )
    if (
        config.no_asr
        and config.input_file is not None
        and not config.extra_transcripts
    ):
        raise SystemExit(
            "--no-asr skips transcription, so an audio file alone has nothing to clean "
            "up; add --extra-transcript or drop --no-asr"
        )
    has_input = any(value is not None for value in inputs) or bool(config.extra_transcripts)
    if config.full_auto and not has_input and config.input_device is None:
        raise SystemExit(
            "--full-auto needs --input, --replay-input-file, --input-text, "
            "--extra-transcript, or an explicit --input-device"
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
    if any(source.backend == "openrouter" for source in config.asr_sources) and not os.environ.get(
        defaults.OPENROUTER_API_KEY_ENV, ""
    ).strip():
        raise SystemExit(
            f"{defaults.OPENROUTER_API_KEY_ENV} is not set; it is required for the "
            "'openrouter' ASR backend (selected via --asr or --online-paid)"
        )


def _interactive_capture_setup(args: argparse.Namespace) -> None:
    """Terminal-dictation prompts: cleanup provider, then microphone."""
    if (
        args.organizer_mode == "llama"
        # A connectivity preset (--offline / --online-free / --online-paid) already
        # fixes the provider in resolve_config, so don't ask — the answer would be
        # silently overridden by the preset anyway.
        and getattr(args, "connectivity", None) is None
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

    capture_mode = (
        all(getattr(args, name) is None for name in ("input", "replay_input_file", "dry_run_text"))
        and not args.extra_transcript  # transcript-only run, not a mic capture
    )
    if capture_mode and args.input_device is None and not args.full_auto:
        _interactive_capture_setup(args)

    config = resolve_config(args)
    validate(config)

    from . import pipeline

    if config.input_archive is not None:
        session = pipeline.run_archive_pipeline(config)
    elif config.input_file is not None:
        session = pipeline.run_file_pipeline(config)
    elif config.dry_run_text is not None:
        session = pipeline.run_dry_text_pipeline(config)
    elif config.extra_transcripts and config.replay_input_file is None:
        # No audio source, only provided transcript file(s): clean them up directly.
        session = pipeline.run_transcript_pipeline(config)
    else:
        from .capture import run_capture_pipeline

        session = run_capture_pipeline(config)
    return 1 if session.run_failed else 0
