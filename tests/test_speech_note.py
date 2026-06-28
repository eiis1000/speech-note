"""Tests for the speech_note package.

Configs are built through the real parse/resolve path (never hand-made
namespaces), and assertions target observable behavior, not implementation
strings.
"""

from __future__ import annotations

import io
import json
import os
import struct
import tempfile
import types
import unittest
import wave
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from speech_note import audio, naming, textproc
from speech_note.capture import SegmentCollector
from speech_note.cli import Config, parse_args, resolve_config, validate
from speech_note.model import Transcript
from speech_note.organizer import (
    ChatClient,
    ChatResponse,
    Organizer,
    build_user_prompt,
    default_server_command,
    ensure_default_cleanup_model,
    request_timeout_seconds,
)
from speech_note.cli import _interactive_capture_setup, fold_unified_input
from speech_note.pipeline import (
    cleanup_sources,
    discover_archive_inputs,
    extract_subtitle_text,
    extract_zip_safely,
    review_panels,
    run_dry_text_pipeline,
    start_organizer_prewarm,
)
from speech_note.asr import (
    build_subprocess_command,
    build_transcriber,
    ctc_chunk_config,
    prepare_sources,
    run_asr_collection,
    run_source,
)
from speech_note.config import ASR_BACKENDS, parse_asr_source
from speech_note.session import ArtifactStore, Session
from speech_note.transcribers import (
    FasterWhisperTranscriber,
    OpenRouterTranscriber,
    SherpaTranscriber,
    default_whisper_cpp_model_path,
    whisper_cpp_model_name,
)


def make_config(*argv: str, tmp_path: Path | None = None) -> Config:
    args = ["--organizer-mode", "heuristic"]
    if tmp_path is not None:
        args += ["--artifacts-dir", str(tmp_path)]
    args += list(argv)
    return resolve_config(parse_args(args))


class TextprocTests(unittest.TestCase):
    def test_normalize_spacing(self) -> None:
        self.assertEqual(textproc.normalize_spacing("  a \n b\t c "), "a b c")

    def test_normalize_paragraphs_keeps_breaks(self) -> None:
        text = "first  line\nstill first\n\n  second   paragraph "
        self.assertEqual(
            textproc.normalize_paragraphs(text),
            "first line still first\n\nsecond paragraph",
        )

    def test_strip_parakeet_timestamps(self) -> None:
        text = "[ 0.0, 3.4]: hello there [3.5, 4.0]: world"
        self.assertEqual(textproc.strip_parakeet_timestamps(text), "hello there world")

    def test_heuristic_cleanup(self) -> None:
        self.assertEqual(
            textproc.heuristic_cleanup("um I think uh this is good"),
            "I think this is good.",
        )

    def test_format_elapsed(self) -> None:
        self.assertEqual(textproc.format_elapsed(5), "00:05")
        self.assertEqual(textproc.format_elapsed(65), "01:05")
        self.assertEqual(textproc.format_elapsed(3700), "01:01:40")


class AudioTests(unittest.TestCase):
    def test_pcm_stats(self) -> None:
        pcm = struct.pack("<hhhh", 1000, -1000, 2000, -2000)
        stats = audio.pcm_stats(pcm)
        self.assertEqual(stats.peak_abs, 2000)
        self.assertAlmostEqual(stats.rms_abs, 1581.14, places=1)
        self.assertLess(stats.peak_dbfs, 0)

    def test_pcm_stats_empty(self) -> None:
        stats = audio.pcm_stats(b"")
        self.assertEqual(stats.peak_abs, 0)
        self.assertIsNone(stats.peak_dbfs)

    def _write_test_wav(self, path: Path, amplitude: int, samples: int = 1600) -> None:
        frames = struct.pack(f"<{samples}h", *([amplitude, -amplitude] * (samples // 2)))
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(frames)

    def test_normalize_pcm_wav_boosts_quiet_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "in.wav"
            target = Path(tmp) / "out.wav"
            self._write_test_wav(source, amplitude=100)
            result = audio.normalize_pcm_wav(source, target)
            self.assertTrue(result.applied)
            self.assertGreater(result.gain, 1.0)
            with wave.open(str(target), "rb") as handle:
                out_frames = handle.readframes(handle.getnframes())
            out_stats = audio.pcm_stats(out_frames)
            self.assertGreater(out_stats.peak_abs, 100)
            # Never clip past the ceiling.
            self.assertLessEqual(out_stats.peak_abs, 32767)

    def test_normalize_pcm_wav_silent_input_copied_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "in.wav"
            target = Path(tmp) / "out.wav"
            self._write_test_wav(source, amplitude=0)
            result = audio.normalize_pcm_wav(source, target)
            self.assertFalse(result.applied)
            self.assertEqual(result.gain, 1.0)
            self.assertEqual(source.read_bytes(), target.read_bytes())

    def test_normalize_respects_max_gain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "in.wav"
            target = Path(tmp) / "out.wav"
            self._write_test_wav(source, amplitude=2)
            result = audio.normalize_pcm_wav(source, target)
            self.assertLessEqual(result.gain, audio.NORMALIZE_MAX_GAIN + 1e-9)


class SegmentCollectorTests(unittest.TestCase):
    def make_collector(self) -> SegmentCollector:
        collector = SegmentCollector(
            sample_rate=16000,
            frame_ms=30,
            vad_mode=1,
            start_padding_ms=90,  # 3 frames of pre-roll
            end_silence_ms=90,  # 3 frames of silence end a segment
            min_speech_ms=60,  # 2 voiced frames minimum
            max_segment_seconds=0.3,  # 10 frames hard cap
        )
        return collector

    def scripted_vad(self, collector: SegmentCollector, decisions: list[bool]) -> None:
        iterator = iter(decisions)
        collector.vad = mock.Mock(is_speech=lambda frame, rate: next(iterator))

    def frame(self, value: int = 1000) -> bytes:
        samples = 16000 * 30 // 1000
        return struct.pack(f"<{samples}h", *([value] * samples))

    def test_segment_emitted_after_end_silence(self) -> None:
        collector = self.make_collector()
        # 1 silent, 4 voiced, 3 silent -> one segment
        self.scripted_vad(collector, [False] + [True] * 4 + [False] * 3)
        segments: list[bytes] = []
        for _ in range(8):
            segments.extend(collector.process_frame(self.frame()))
        self.assertEqual(len(segments), 1)
        # 1 pre-roll frame + 4 voiced + 3 silent = 8 frames... pre-roll holds the
        # silent frame, so the segment spans all frames seen since it.
        self.assertEqual(len(segments[0]), 8 * collector.frame_bytes)

    def test_short_blip_is_not_a_segment(self) -> None:
        collector = self.make_collector()
        self.scripted_vad(collector, [True] + [False] * 10)
        segments: list[bytes] = []
        for _ in range(11):
            segments.extend(collector.process_frame(self.frame()))
        # One voiced frame < min_speech_ms: finalization requires min speech.
        self.assertEqual(segments, [])

    def test_max_segment_cap_forces_finalize(self) -> None:
        collector = self.make_collector()
        self.scripted_vad(collector, [True] * 30)
        segments: list[bytes] = []
        for _ in range(15):
            segments.extend(collector.process_frame(self.frame()))
        self.assertTrue(segments)
        self.assertLessEqual(len(segments[0]), collector.max_segment_frames * collector.frame_bytes)

    def test_flush_returns_pending_speech(self) -> None:
        collector = self.make_collector()
        self.scripted_vad(collector, [True] * 4)
        for _ in range(4):
            collector.process_frame(self.frame())
        flushed = collector.flush()
        self.assertEqual(len(flushed), 1)
        self.assertEqual(collector.flush(), [])

    def test_wrong_size_frame_ignored(self) -> None:
        collector = self.make_collector()
        self.assertEqual(collector.process_frame(b"\x00\x01"), [])


class NamingTests(unittest.TestCase):
    def test_transcript_source_tag(self) -> None:
        self.assertEqual(naming.transcript_source_tag(Path("meeting.google.txt"), "extra"), "google")
        self.assertEqual(naming.transcript_source_tag(Path("meeting.parakeet.txt"), "extra"), "parakeet")
        self.assertEqual(naming.transcript_source_tag(Path("human-edited.txt"), "extra"), "edited")

    def test_unique_output_path_avoids_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "note.txt").write_text("x")
            (directory / "note-2.txt").write_text("x")
            self.assertEqual(
                naming.unique_output_path(directory, "note", ".txt"),
                directory / "note-3.txt",
            )

    def test_full_auto_stem_for_transcript_inputs(self) -> None:
        config = make_config(
            "--primary-transcript", "/tmp/compute-orientation.whisper.txt",
            "--secondary-transcript", "/tmp/compute-orientation.parakeet.txt",
            "--extra-transcript", "/tmp/compute-orientation.google.txt",
        )
        self.assertEqual(
            naming.full_auto_source_stem(config),
            "compute-orientation-whisper-parakeet-google",
        )

    def test_full_auto_stem_for_archive(self) -> None:
        config = make_config(
            "--input", "/tmp/cosmicwatch.zip",
            "--extra-transcript", "/tmp/cosmicwatch.google.txt",
        )
        self.assertEqual(
            naming.full_auto_source_stem(config),
            "cosmicwatch-whisper-parakeet-google",
        )


class ConfigResolutionTests(unittest.TestCase):
    def test_default_collection_is_whisper_plus_sherpa(self) -> None:
        sources = make_config().asr_sources
        self.assertEqual([s.backend for s in sources], ["whisper-cpp", "sherpa"])
        self.assertEqual(sources[0].model, "medium-q8_0")
        self.assertEqual(sources[1].model, "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8")
        # Different devices → they can overlap.
        self.assertEqual([s.device_kind for s in sources], ["gpu", "cpu"])

    def test_asr_flag_overrides_collection(self) -> None:
        sources = make_config("--asr", "ctc@cpu").asr_sources
        self.assertEqual([(s.backend, s.model, s.device) for s in sources],
                         [("ctc", "nvidia/parakeet-ctc-0.6b", "cpu")])

    def test_asr_flag_repeatable_and_comma_separated(self) -> None:
        sources = make_config("--asr", "whisper-cpp:medium-q8_0", "--asr", "sherpa,ctc@cpu").asr_sources
        self.assertEqual([s.backend for s in sources], ["whisper-cpp", "sherpa", "ctc"])

    def test_model_defaults_per_backend(self) -> None:
        self.assertEqual(parse_asr_source("onnx").model, "nemo-parakeet-tdt-0.6b-v2")
        self.assertEqual(parse_asr_source("crispasr").model, "parakeet-tdt-0.6b-v2-q4_k.gguf")

    def test_offline_flag_forces_local_provider(self) -> None:
        config = make_config("--offline", "--organizer-mode", "llama")
        self.assertEqual(config.organizer_provider, "local")

    def test_online_free_flag_uses_free_chain(self) -> None:
        from speech_note import config as cfg
        config = make_config("--online-free")
        self.assertEqual(config.organizer_provider, "openrouter")
        self.assertEqual(config.organizer_models, tuple(cfg.OPENROUTER_FREE_MODELS))
        self.assertGreaterEqual(len(config.organizer_models), 5)  # deep fallback chain
        self.assertTrue(all(m.endswith(":free") for m in config.organizer_models))

    def test_online_paid_flag_leads_paid_then_falls_back_to_free(self) -> None:
        from speech_note import config as cfg
        config = make_config("--online-paid")
        self.assertEqual(config.organizer_provider, "openrouter")
        self.assertEqual(config.organizer_models[0], "deepseek/deepseek-v3.2")
        # paid leads, but the free chain is appended as a fallback for paid outages
        self.assertTrue(any(m.endswith(":free") for m in config.organizer_models))
        for free in cfg.OPENROUTER_FREE_MODELS:
            self.assertIn(free, config.organizer_models)

    def test_explicit_model_overrides_connectivity_mode(self) -> None:
        config = make_config("--online-paid", "--organizer-model", "custom/model")
        self.assertEqual(config.organizer_models, ("custom/model",))

    def test_connectivity_modes_are_mutually_exclusive(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--offline", "--online-paid"])

    def test_load_user_env_sets_default_without_override(self) -> None:
        from speech_note import config as cfg
        with tempfile.TemporaryDirectory() as d:
            envf = Path(d) / "env"
            envf.write_text("# comment\nOPENROUTER_API_KEY='sk-or-test'\nEMPTY\n", encoding="utf-8")
            with mock.patch.dict("os.environ", {}, clear=True):
                cfg.load_user_env(envf)
                self.assertEqual(os.environ["OPENROUTER_API_KEY"], "sk-or-test")
            # an explicit shell export must win over the file
            with mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "shell-key"}, clear=True):
                cfg.load_user_env(envf)
                self.assertEqual(os.environ["OPENROUTER_API_KEY"], "shell-key")

    def test_openrouter_defaults(self) -> None:
        config = make_config("--organizer-provider", "openrouter")
        self.assertIn("openrouter.ai", config.organizer_api_base)
        self.assertEqual(config.organizer_context_tokens, 262144)
        self.assertEqual(config.organizer_max_output_tokens, 32768)
        self.assertEqual(config.organizer_auth_env, "OPENROUTER_API_KEY")
        self.assertTrue(config.organizer_models)

    def test_explicit_context_tokens_survive_provider_defaults(self) -> None:
        config = make_config(
            "--organizer-provider", "openrouter",
            "--organizer-context-tokens", "65536",
        )
        self.assertEqual(config.organizer_context_tokens, 65536)

    def test_full_auto_redirects_artifacts_and_names_output(self) -> None:
        config = make_config("--full-auto", "--input", "/recordings/My Lecture.m4a")
        self.assertNotEqual(config.artifacts_dir, Path("."))
        self.assertIsNotNone(config.output)
        self.assertEqual(config.output.name, "my-lecture-clean.txt")

    def test_full_auto_respects_explicit_output(self) -> None:
        config = make_config(
            "--full-auto",
            "--input", "/recordings/x.m4a",
            "--output", "/tmp/chosen.txt",
        )
        self.assertEqual(config.output, Path("/tmp/chosen.txt"))

    def test_validate_rejects_multiple_inputs(self) -> None:
        config = make_config("--input", "/a.wav", "--input-text", "hello")
        with self.assertRaises(SystemExit):
            validate(config)

    def test_validate_secondary_transcript_requires_primary(self) -> None:
        config = make_config("--secondary-transcript", "/tmp/x.txt")
        with self.assertRaises(SystemExit):
            validate(config)

    def test_validate_requires_openrouter_key(self) -> None:
        args = parse_args(["--organizer-provider", "openrouter", "--input-text", "x"])
        config = resolve_config(args)
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(SystemExit):
                validate(config)

    def test_validate_full_auto_requires_input(self) -> None:
        config = make_config("--full-auto", "--input", "/a.wav")
        validate(config)  # ok
        with self.assertRaises(SystemExit):
            validate(make_config("--full-auto"))


class AsrSourceTests(unittest.TestCase):
    def test_device_kind_for_scheduling(self) -> None:
        # whisper.cpp on GPU + sherpa on CPU -> different device groups, so they
        # overlap; ctc defaults to CPU; crispasr forced to cpu is a cpu source.
        self.assertEqual(parse_asr_source("whisper-cpp@gpu").device_kind, "gpu")
        self.assertEqual(parse_asr_source("sherpa").device_kind, "cpu")
        self.assertEqual(parse_asr_source("ctc").device_kind, "cpu")
        self.assertEqual(parse_asr_source("ctc@cuda:0").device_kind, "gpu")
        self.assertEqual(parse_asr_source("crispasr@cpu").device_kind, "cpu")
        self.assertEqual(parse_asr_source("crispasr").device_kind, "gpu")

    def test_default_device_per_backend(self) -> None:
        # No @device -> the backend's natural device string.
        self.assertEqual(parse_asr_source("whisper-cpp").device, "0")
        self.assertEqual(parse_asr_source("sherpa").device, "cpu")
        self.assertEqual(parse_asr_source("crispasr").device, "vulkan")

    def test_unknown_backend_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_asr_source("not-a-backend")

    def test_user_config_file_supplies_collection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "asr"
            cfg.write_text("# my collection\nwhisper-cpp:small.en@gpu\nctc@cpu\n")
            from speech_note import config as config_module

            sources = config_module.load_user_asr_sources(cfg)
        self.assertEqual([(s.backend, s.model, s.device, s.device_kind) for s in sources],
                         [("whisper-cpp", "small.en", "gpu", "gpu"),
                          ("ctc", "nvidia/parakeet-ctc-0.6b", "cpu", "cpu")])


class WhisperCppNamingTests(unittest.TestCase):
    def test_model_alias_mapping(self) -> None:
        self.assertEqual(whisper_cpp_model_name("Systran/faster-whisper-medium.en"), "medium.en")
        self.assertTrue(
            str(default_whisper_cpp_model_path("medium-q8_0")).endswith("ggml-medium-q8_0.bin")
        )


class AsrBackendTests(unittest.TestCase):
    def test_ctc_chunk_window_under_position_cliff(self) -> None:
        # CTC long-form striding: window stays under the model's ~400s position
        # limit, stride overlaps it. No length cap — any length is transcribed.
        chunk_length, stride = ctc_chunk_config()
        self.assertEqual(chunk_length, 240.0)
        self.assertEqual(stride, 5.0)
        self.assertLess(stride, chunk_length)

    def test_registry_marks_subprocess_backends(self) -> None:
        self.assertFalse(ASR_BACKENDS["onnx"].in_process)
        self.assertFalse(ASR_BACKENDS["crispasr"].in_process)
        self.assertTrue(ASR_BACKENDS["sherpa"].in_process)
        self.assertTrue(ASR_BACKENDS["whisper-cpp"].in_process)

    def test_sherpa_builds_in_process_not_subprocess(self) -> None:
        source = parse_asr_source("sherpa")
        # In-process backends have no subprocess command.
        with self.assertRaises(ValueError):
            build_subprocess_command(make_config(), source, Path("/tmp/audio.wav"))
        transcriber = build_transcriber(make_config(), source)
        assert isinstance(transcriber, SherpaTranscriber)
        self.assertEqual(transcriber.device, "cpu")
        self.assertEqual(
            transcriber.model_name, "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
        )

    def test_onnx_command(self) -> None:
        command = build_subprocess_command(
            make_config(), parse_asr_source("onnx"), Path("/tmp/audio.wav")
        )
        self.assertEqual(command[0], "onnx-asr")
        self.assertIn("nemo-parakeet-tdt-0.6b-v2", command)
        self.assertIn("/tmp/audio.wav", command)

    def test_crispasr_command_device_mapping(self) -> None:
        command = build_subprocess_command(
            make_config(), parse_asr_source("crispasr"), Path("/tmp/audio.wav")
        )
        self.assertEqual(command[0], "crispasr")
        self.assertIn("vulkan", command)
        cpu_command = build_subprocess_command(
            make_config(), parse_asr_source("crispasr@cpu"), Path("/tmp/audio.wav")
        )
        self.assertIn("cpu", cpu_command)

    def test_in_process_backend_has_no_subprocess_command(self) -> None:
        with self.assertRaises(ValueError):
            build_subprocess_command(make_config(), parse_asr_source("ctc"), Path("/tmp/a.wav"))

    def test_run_source_strips_parakeet_timestamps(self) -> None:
        source = parse_asr_source("onnx")
        completed = mock.Mock(returncode=0, stdout="[ 0.0, 1.0]: hello [1.0, 2.0]: world\n", stderr="")
        with mock.patch("speech_note.asr.subprocess.run", return_value=completed):
            outcome = run_source(
                make_config(), source, Path("/tmp/a.wav"),
                label="asr1", duration=2.0, transcriber=None,
            )
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.transcript.text, "hello world")
        self.assertEqual(outcome.transcript.label, "asr1")
        self.assertEqual(outcome.transcript.model, "nemo-parakeet-tdt-0.6b-v2")
        self.assertIsNone(outcome.transcript.quality_hint)  # un-noted source: no reliability claim
        self.assertIsNotNone(outcome.realtime_factor)

    def test_run_source_can_keep_timestamps(self) -> None:
        config = make_config("--no-strip-asr-timestamps")
        source = parse_asr_source("onnx")
        completed = mock.Mock(returncode=0, stdout="[ 0.0, 1.0]: hello\n", stderr="")
        with mock.patch("speech_note.asr.subprocess.run", return_value=completed):
            outcome = run_source(
                config, source, Path("/tmp/a.wav"),
                label="asr1", duration=1.0, transcriber=None,
            )
        self.assertEqual(outcome.transcript.text, "[ 0.0, 1.0]: hello")

    def test_run_source_failure_is_an_error_outcome(self) -> None:
        source = parse_asr_source("onnx")
        completed = mock.Mock(returncode=3, stdout="", stderr="model exploded")
        with mock.patch("speech_note.asr.subprocess.run", return_value=completed):
            outcome = run_source(
                make_config(), source, Path("/tmp/a.wav"),
                label="asr1", duration=1.0, transcriber=None,
            )
        self.assertFalse(outcome.ok)
        self.assertIn("model exploded", outcome.error)
        self.assertIsNone(outcome.skip_reason)


class AsrPrepareTests(unittest.TestCase):
    """Model download/consent happens up front (prepare_sources), on the main
    thread — never under a status spinner or in an ASR worker thread."""

    def test_faster_whisper_construction_is_lazy(self) -> None:
        # A built-but-unrun source must not load or download: construction stays
        # cheap and offline so prepare_sources is the single, visible fetch point.
        transcriber = FasterWhisperTranscriber(
            model_name="definitely/not-a-real-model",
            device="cpu",
            compute_type="int8",
            cpu_threads=1,
            download_root=None,
        )
        self.assertIsNone(transcriber.model)

    def test_prepare_invokes_ensure_downloaded(self) -> None:
        source = parse_asr_source("sherpa")
        transcriber = mock.Mock(spec=["ensure_downloaded"])
        prepared = prepare_sources([(source, transcriber)])
        transcriber.ensure_downloaded.assert_called_once_with()
        self.assertEqual(prepared[0][0], "asr1")
        self.assertIsNone(prepared[0][3])  # no prep error

    def test_failed_prepare_records_error_without_running_transcribe(self) -> None:
        source = parse_asr_source("sherpa")

        class _Boom:
            def ensure_downloaded(self) -> None:
                raise RuntimeError("download declined")

            def transcribe_file(self, *args: object, **kwargs: object) -> str:
                raise AssertionError("must not transcribe after a failed prepare")

        prepared = prepare_sources([(source, _Boom())])
        self.assertEqual(prepared[0][3], "download declined")
        session = Session(make_config())
        run_asr_collection(
            make_config(), session, Path("/tmp/none.wav"), prepared=prepared, duration=1.0
        )
        self.assertEqual(len(session.asr_outcomes), 1)
        self.assertFalse(session.asr_outcomes[0].ok)
        self.assertIn("download declined", session.asr_outcomes[0].error or "")
        self.assertEqual(session.asr_transcripts(), [])


class TranscriptInputViewTests(unittest.TestCase):
    """A transcript-only input (no ASR/live pass) is still surfaced as the raw
    transcript and as a review panel, not just fed silently to cleanup."""

    def test_external_transcript_is_the_raw_text(self) -> None:
        session = Session(make_config())
        session.add_transcript(
            Transcript(label="primary", model="draft.txt", kind="external", text="rough draft")
        )
        self.assertEqual(session.raw_text(), "rough draft")

    def test_external_transcript_appears_as_review_panel(self) -> None:
        session = Session(make_config())
        session.add_transcript(
            Transcript(label="primary", model="draft.txt", kind="external", text="rough draft")
        )
        panels = review_panels(make_config(), session)
        self.assertIn(("draft.txt", "rough draft"), panels)


class InteractiveCaptureSetupTests(unittest.TestCase):
    def test_connectivity_preset_skips_provider_prompt(self) -> None:
        # --offline / --online-* already fix the provider, so the interactive
        # capture flow must not also ask (the answer would be overridden anyway).
        args = parse_args(["--offline"])
        with mock.patch("speech_note.cli.select_input_device", return_value=0), \
                mock.patch("speech_note.cli.read_single_choice") as choice:
            _interactive_capture_setup(args)
        choice.assert_not_called()

    def test_no_preset_prompts_for_provider(self) -> None:
        args = parse_args([])
        with mock.patch("speech_note.cli.select_input_device", return_value=0), \
                mock.patch("speech_note.cli.sys.stdin") as stdin, \
                mock.patch("speech_note.cli.read_single_choice", return_value="2") as choice, \
                redirect_stdout(io.StringIO()):
            stdin.isatty.return_value = True
            _interactive_capture_setup(args)
        choice.assert_called_once()
        self.assertEqual(args.organizer_provider, "openrouter")


class OrganizerPrewarmTests(unittest.TestCase):
    @staticmethod
    def _organizer() -> tuple[object, mock.Mock]:
        supervisor = mock.Mock()
        return types.SimpleNamespace(supervisor=supervisor), supervisor

    def test_no_prewarm_flag_skips_background_load(self) -> None:
        config = make_config("--organizer-mode", "llama", "--organizer-no-prewarm")
        self.assertFalse(config.organizer_prewarm)
        organizer, supervisor = self._organizer()
        self.assertIsNone(start_organizer_prewarm(config, organizer))
        supervisor.ensure_running.assert_not_called()

    def test_default_prewarms_in_background(self) -> None:
        config = make_config("--organizer-mode", "llama")
        self.assertTrue(config.organizer_prewarm)
        organizer, supervisor = self._organizer()
        with redirect_stderr(io.StringIO()):
            thread = start_organizer_prewarm(config, organizer)
            self.assertIsNotNone(thread)
            thread.join(timeout=5)
        supervisor.ensure_running.assert_called_once()


class OrganizerPromptTests(unittest.TestCase):
    def test_prompt_names_sources_and_models(self) -> None:
        sources = [
            Transcript("primary", "ggml-medium-q8_0.bin", "asr-final", "one " * 100,
                       quality_hint="primary ASR pass"),
            Transcript("secondary", "parakeet", "asr-final", "two " * 200),
            Transcript("extra:google.txt", "google.txt", "external", "three " * 150),
        ]
        prompt = build_user_prompt(sources)
        self.assertIn("Source 1 — primary (ggml-medium-q8_0.bin)", prompt)
        self.assertIn("primary ASR pass", prompt)
        self.assertIn("Source 2 — secondary (parakeet)", prompt)
        self.assertIn("Source 3 — extra:google.txt (google.txt)", prompt)
        # Reference length comes from the shortest non-empty source.
        self.assertIn("The shortest source is 100 words", prompt)

    def test_timeout_scales_with_request_size(self) -> None:
        small = request_timeout_seconds(
            configured_timeout=12.0, estimated_prompt_tokens=500, requested_output_tokens=1024
        )
        large = request_timeout_seconds(
            configured_timeout=12.0, estimated_prompt_tokens=5000, requested_output_tokens=8192
        )
        self.assertGreaterEqual(small, 45.0)
        self.assertGreater(large, small)
        self.assertLessEqual(large, 1800.0)


def fake_response(status_code: int, payload: dict) -> mock.Mock:
    response = mock.Mock()
    response.ok = status_code == 200
    response.status_code = status_code
    response.text = json.dumps(payload)
    response.json.return_value = payload
    return response


def make_llm_organizer(models: str = "model-a") -> Organizer:
    client = ChatClient(
        api_base="http://127.0.0.1:9/v1/chat/completions",
        models=models.split(","),
        timeout=0.01,
    )
    return Organizer(
        mode="llama", client=client, supervisor=None, context_tokens=16384, max_output_tokens=4096
    )


class OrganizerLlmTests(unittest.TestCase):
    def setUp(self) -> None:
        # No catalog endpoint in tests: candidate_models falls back to configured.
        patcher = mock.patch(
            "speech_note.organizer.requests.get", side_effect=Exception("no catalog")
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def chat_payload(self, content: str, *, model: str = "served-model", finish: str = "stop") -> dict:
        return {
            "model": model,
            "choices": [{"message": {"content": content}, "finish_reason": finish}],
        }

    def test_successful_cleanup_records_served_model(self) -> None:
        organizer = make_llm_organizer()
        sources = [Transcript("primary", "whisper", "asr-final", "word " * 100)]
        response = fake_response(200, self.chat_payload("clean " * 89 + "done."))
        with mock.patch("speech_note.organizer.requests.post", return_value=response):
            outcome = organizer.cleanup(sources)
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.method, "llama")
        self.assertEqual(outcome.served_model, "served-model")
        self.assertFalse(outcome.flagged_short)

    def test_fallback_across_models_on_transient_error(self) -> None:
        organizer = make_llm_organizer("first,second")
        calls: list[str] = []

        def fake_post(_url, *, data, **_kwargs):
            model = json.loads(data)["model"]
            calls.append(model)
            if model == "first":
                return fake_response(429, {"error": {"message": "rate-limited"}})
            return fake_response(200, self.chat_payload("clean " * 90, model="second"))

        sources = [Transcript("primary", "whisper", "asr-final", "word " * 100)]
        with mock.patch("speech_note.organizer.requests.post", side_effect=fake_post):
            outcome = organizer.cleanup(sources)
        self.assertEqual(calls, ["first", "second"])
        self.assertEqual(outcome.served_model, "second")

    def test_model_404_falls_through_for_authed_providers(self) -> None:
        client = ChatClient(
            api_base="https://openrouter.ai/api/v1/chat/completions",
            models=["stale:free", "working:free"],
            timeout=0.01,
            auth_env="OPENROUTER_API_KEY",
        )
        organizer = Organizer(
            mode="llama", client=client, supervisor=None,
            context_tokens=16384, max_output_tokens=4096,
        )
        calls: list[str] = []

        def fake_post(_url, *, data, **_kwargs):
            model = json.loads(data)["model"]
            calls.append(model)
            if model == "stale:free":
                return fake_response(404, {"error": {"message": "no longer exists"}})
            return fake_response(200, self.chat_payload("clean " * 90, model="working:free"))

        sources = [Transcript("primary", "whisper", "asr-final", "word " * 100)]
        with mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "k"}):
            with mock.patch("speech_note.organizer.requests.post", side_effect=fake_post):
                outcome = organizer.cleanup(sources)
        self.assertEqual(calls, ["stale:free", "working:free"])
        self.assertTrue(outcome.ok)

    def test_truncated_output_is_an_error(self) -> None:
        organizer = make_llm_organizer()
        sources = [Transcript("primary", "whisper", "asr-final", "word " * 100)]
        response = fake_response(200, self.chat_payload("clean " * 90, finish="length"))
        with mock.patch("speech_note.organizer.requests.post", return_value=response):
            outcome = organizer.cleanup(sources)
        self.assertTrue(outcome.flagged_short)
        self.assertIn("truncated", outcome.error)

    def test_short_output_is_flagged_but_kept(self) -> None:
        organizer = make_llm_organizer()
        sources = [Transcript("primary", "whisper", "asr-final", "word " * 100)]
        response = fake_response(200, self.chat_payload("too short"))
        with mock.patch("speech_note.organizer.requests.post", return_value=response):
            outcome = organizer.cleanup(sources)
        self.assertEqual(outcome.text, "too short")
        self.assertTrue(outcome.flagged_short)
        self.assertIsNone(outcome.error)
        self.assertIn("short", outcome.warning)

    def test_mid_sentence_output_is_flagged_but_kept(self) -> None:
        organizer = make_llm_organizer()
        sources = [Transcript("primary", "whisper", "asr-final", "word " * 100)]
        # Long enough to pass the length ratio, but cut off mid-sentence (no terminal
        # punctuation) — the rita-c failure mode that finish_reason="stop" hid.
        cut_off = "clean " * 89 + "and then I"
        response = fake_response(200, self.chat_payload(cut_off))
        with mock.patch("speech_note.organizer.requests.post", return_value=response):
            outcome = organizer.cleanup(sources)
        self.assertEqual(outcome.text, cut_off.strip())  # output preserved, not deleted
        self.assertTrue(outcome.flagged_short)
        self.assertIsNone(outcome.error)  # a flag, never an error/deletion
        self.assertIn("mid-sentence", outcome.warning)

    def test_complete_output_is_not_flagged_mid_sentence(self) -> None:
        # A faithful trail-off that the speaker actually made still ends in '.', '?', '!'
        # or '…', so it must NOT be flagged.
        for ending in ("the time is...", "good night.", "what now?", "part of the like…", 'he said "hello."'):
            organizer = make_llm_organizer()
            sources = [Transcript("primary", "whisper", "asr-final", "word " * 10)]
            response = fake_response(200, self.chat_payload("clean " * 20 + ending))
            with mock.patch("speech_note.organizer.requests.post", return_value=response):
                outcome = organizer.cleanup(sources)
            self.assertFalse(outcome.flagged_short, msg=f"false positive on {ending!r}")
            self.assertIsNone(outcome.warning, msg=f"false positive on {ending!r}")

    def test_oversized_prompt_is_skipped(self) -> None:
        client = ChatClient(api_base="http://x/v1/chat/completions", models=["m"], timeout=0.01)
        organizer = Organizer(
            mode="llama", client=client, supervisor=None, context_tokens=512, max_output_tokens=256
        )
        sources = [Transcript("primary", "whisper", "asr-final", "word " * 2000)]
        outcome = organizer.cleanup(sources)
        self.assertEqual(outcome.method, "skipped-too-large")
        self.assertFalse(outcome.ok)

    def test_duplicate_sources_are_deduplicated(self) -> None:
        organizer = make_llm_organizer()
        prompts: list[str] = []

        def fake_post(_url, *, data, **_kwargs):
            prompts.append(json.loads(data)["messages"][1]["content"])
            return fake_response(200, self.chat_payload("clean " * 90))

        same = "word " * 100
        sources = [
            Transcript("primary", "whisper", "asr-final", same),
            Transcript("secondary", "parakeet", "asr-final", same),
        ]
        with mock.patch("speech_note.organizer.requests.post", side_effect=fake_post):
            organizer.cleanup(sources)
        self.assertIn("Source 1", prompts[0])
        self.assertNotIn("Source 2", prompts[0])


class ChatClientCatalogTests(unittest.TestCase):
    def test_stale_models_filtered_against_catalog(self) -> None:
        client = ChatClient(
            api_base="https://openrouter.ai/api/v1/chat/completions",
            models=["gone:free", "alive:free"],
            timeout=0.01,
        )
        catalog = fake_response(200, {"data": [{"id": "alive:free"}, {"id": "other"}]})
        catalog.raise_for_status = mock.Mock()
        with mock.patch("speech_note.organizer.requests.get", return_value=catalog):
            self.assertEqual(client.candidate_models(), ["alive:free"])

    def test_catalog_failure_keeps_configured_list(self) -> None:
        client = ChatClient(
            api_base="http://127.0.0.1:9/v1/chat/completions", models=["a", "b"], timeout=0.01
        )
        with mock.patch(
            "speech_note.organizer.requests.get", side_effect=Exception("offline")
        ):
            self.assertEqual(client.candidate_models(), ["a", "b"])

    def test_fully_stale_catalog_keeps_configured_list(self) -> None:
        client = ChatClient(
            api_base="http://127.0.0.1:9/v1/chat/completions", models=["a"], timeout=0.01
        )
        catalog = fake_response(200, {"data": [{"id": "other"}]})
        catalog.raise_for_status = mock.Mock()
        with mock.patch("speech_note.organizer.requests.get", return_value=catalog):
            self.assertEqual(client.candidate_models(), ["a"])


class ServerCommandTests(unittest.TestCase):
    def test_default_server_command_kv_offload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_text("fake")
            with mock.patch("speech_note.organizer.DEFAULT_GGUF_MODEL", model):
                with mock.patch("speech_note.organizer.shutil.which", return_value="/bin/llama-server"):
                    # KV offload is on by default (fixed upstream; ~1.5x faster).
                    command = default_server_command(context_tokens=4096)
                    self.assertNotIn("--no-kv-offload", command)
                    # The workaround for older stacks is still reachable.
                    command = default_server_command(context_tokens=4096, kv_offload=False)
                    self.assertIn("--no-kv-offload", command)

    def test_default_server_command_missing_model(self) -> None:
        with mock.patch("speech_note.organizer.DEFAULT_GGUF_MODEL", Path("/nonexistent.gguf")):
            self.assertIsNone(default_server_command(context_tokens=4096))

    def test_default_server_command_model_path_override(self) -> None:
        # A stronger local cleanup model can be dropped in via model_path (the
        # --organizer-gguf flag), overriding the bundled gemma-E2B default.
        with tempfile.TemporaryDirectory() as tmp:
            stronger = Path(tmp) / "gemma-31b.gguf"
            stronger.write_text("fake")
            with mock.patch("speech_note.organizer.DEFAULT_GGUF_MODEL", Path("/nonexistent.gguf")):
                with mock.patch("speech_note.organizer.shutil.which", return_value="/bin/llama-server"):
                    command = default_server_command(context_tokens=4096, model_path=stronger)
                    self.assertIsNotNone(command)
                    self.assertIn(str(stronger), command)

    def test_cli_organizer_gguf_flag(self) -> None:
        config = make_config("--organizer-gguf", "/models/big.gguf")
        self.assertEqual(config.organizer_gguf, Path("/models/big.gguf"))
        self.assertIsNone(make_config().organizer_gguf)

    def test_ensure_default_cleanup_model_already_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "cleanup.gguf"
            model.write_text("fake")
            with mock.patch("speech_note.organizer.DEFAULT_GGUF_MODEL", model):
                # Present already: returns True without importing/calling the downloader.
                self.assertTrue(ensure_default_cleanup_model(auto_yes=True))

    def test_ensure_default_cleanup_model_declined(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "missing.gguf"
            with mock.patch("speech_note.organizer.DEFAULT_GGUF_MODEL", model):
                # Non-interactive without --auto-download declines, no download attempted.
                self.assertFalse(
                    ensure_default_cleanup_model(auto_yes=False, interactive=False)
                )

    def test_ensure_default_cleanup_model_downloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "sub" / "cleanup.gguf"

            def fake_download(repo: str, filename: str, *, local_dir: str) -> str:
                dest = Path(local_dir) / filename
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text("downloaded")
                # the real model path is local_dir/<DEFAULT_GGUF_MODEL.name>; here we
                # mirror that by writing the patched model path directly
                model.write_text("downloaded")
                return str(dest)

            fake_hub = types.ModuleType("huggingface_hub")
            fake_hub.hf_hub_download = fake_download  # type: ignore[attr-defined]
            with mock.patch("speech_note.organizer.DEFAULT_GGUF_MODEL", model):
                with mock.patch.dict("sys.modules", {"huggingface_hub": fake_hub}):
                    self.assertTrue(ensure_default_cleanup_model(auto_yes=True))
            self.assertTrue(model.exists())


class ArchiveTests(unittest.TestCase):
    def test_discover_archive_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "rec").mkdir()
            (root / "xn").mkdir()
            audio_file = root / "rec" / "note.m4a"
            transcript = root / "xn" / "note.txt"
            audio_file.write_bytes(b"fake audio")
            transcript.write_text("transcript")
            found_audio, transcripts = discover_archive_inputs(root)
            self.assertEqual(found_audio, audio_file)
            self.assertEqual(transcripts, [transcript])

    def test_extract_zip_rejects_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_path = root / "bad.zip"
            extract_dir = root / "extract"
            extract_dir.mkdir()
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../escape.txt", "no")
            with self.assertRaisesRegex(SystemExit, "escapes extraction"):
                extract_zip_safely(archive_path, extract_dir)

    def test_extract_subtitle_text(self) -> None:
        srt = "1\n00:00:01,000 --> 00:00:02,000\nhello there\n\n2\n00:00:02,000 --> 00:00:03,000\nworld\n"
        self.assertEqual(extract_subtitle_text(srt), "hello there\nworld")
        vtt = "WEBVTT\n\n00:01.000 --> 00:02.000\nhi\n"
        self.assertEqual(extract_subtitle_text(vtt), "hi")


class CleanupSourceSelectionTests(unittest.TestCase):
    def test_live_transcript_only_used_when_primary_missing(self) -> None:
        config = make_config(tmp_path=Path("."))
        session = Session(config)
        live = Transcript("live", "tiny", "asr-live", "live text")
        session.add_transcript(live)
        self.assertEqual(cleanup_sources(session), [live])
        primary = Transcript("primary", "medium", "asr-final", "final text")
        session.add_transcript(primary)
        self.assertEqual(cleanup_sources(session), [primary])

    def test_user_transcripts_not_duplicated(self) -> None:
        config = make_config(tmp_path=Path("."))
        session = Session(config)
        session.add_transcript(Transcript("primary", "dry-run-text", "user", "hello"))
        self.assertEqual(len(cleanup_sources(session)), 1)


class PipelineEndToEndTests(unittest.TestCase):
    """Dry-run text through the real pipeline: cleanup, single artifact commit,
    diagnostics shape — no audio hardware or models involved."""

    def run_dry(self, tmp: Path, *argv: str) -> Session:
        config = make_config(
            "--input-text", "um hello there || this is uh a test",
            *argv,
            tmp_path=tmp,
        )
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            session = run_dry_text_pipeline(config)
        return session

    def test_dry_run_writes_all_artifacts_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            session = self.run_dry(tmp)
            self.assertFalse(session.run_failed)
            self.assertEqual(
                (tmp / "raw.latest").read_text().strip(),
                "um hello there\nthis is uh a test",
            )
            clean = (tmp / "clean.latest").read_text().strip()
            self.assertNotIn("um", clean.split())
            archive_files = sorted(path.name for path in (tmp / "logs").iterdir())
            self.assertEqual(len([n for n in archive_files if n.endswith("-raw")]), 1)
            self.assertEqual(len([n for n in archive_files if n.endswith("-clean")]), 1)
            self.assertEqual(len([n for n in archive_files if n.endswith("-diagnostics.json")]), 1)

    def test_diagnostics_payload_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            self.run_dry(tmp)
            payload = json.loads((tmp / "diagnostics.latest.json").read_text())
            self.assertEqual(payload["schema"], 2)
            self.assertEqual(payload["cleanup"]["method"], "heuristic")
            self.assertFalse(payload["audio_levels"]["measured"])
            self.assertEqual(payload["transcripts"][0]["label"], "primary")
            self.assertEqual(payload["transcripts"][0]["kind"], "user")
            self.assertIn("config", payload)
            self.assertEqual(payload["errors"], [])

    def test_output_flag_writes_clean_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            output = tmp / "out" / "note.txt"
            session = self.run_dry(tmp, "--output", str(output))
            self.assertTrue(output.exists())
            self.assertEqual(output.read_text().strip(), session.cleanup.text)

    def test_extra_transcripts_reach_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            extra = tmp / "google.txt"
            extra.write_text("google transcript text")
            session = self.run_dry(tmp, "--extra-transcript", str(extra))
            labels = [t.label for t in session.transcripts]
            self.assertIn("extra:google.txt", labels)


class FullAutoTests(unittest.TestCase):
    def test_full_auto_failure_leaves_no_clean_file_but_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            output = tmp / "note-clean.txt"
            config = resolve_config(
                parse_args(
                    [
                        "--input-text", "hello there",
                        "--full-auto",
                        "--output", str(output),
                        "--organizer-mode", "llama",
                        "--organizer-api-base", "http://127.0.0.1:9/v1/chat/completions",
                        "--organizer-server-command",  # empty: no server launch
                    ]
                )
            )
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with mock.patch(
                    "speech_note.organizer.requests.get", side_effect=Exception("offline")
                ):
                    with mock.patch(
                        "speech_note.organizer.requests.post",
                        side_effect=Exception("connection refused"),
                    ):
                        from speech_note.pipeline import run_dry_text_pipeline

                        session = run_dry_text_pipeline(config)
            self.assertTrue(session.run_failed)
            self.assertFalse(output.exists())
            error_diag = tmp / "note-clean-diagnostics.json"
            self.assertTrue(error_diag.exists())
            payload = json.loads(error_diag.read_text())
            self.assertTrue(payload["errors"])

    def test_full_auto_success_writes_only_clean_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            output = tmp / "note-clean.txt"
            config = resolve_config(
                parse_args(
                    [
                        "--input-text", "um hello there",
                        "--full-auto",
                        "--output", str(output),
                        "--organizer-mode", "heuristic",
                    ]
                )
            )
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                session = run_dry_text_pipeline(config)
            self.assertFalse(session.run_failed)
            self.assertTrue(output.exists())
            self.assertEqual(
                [path.name for path in tmp.iterdir()], ["note-clean.txt"]
            )


class SessionTests(unittest.TestCase):
    def test_audio_measurement_flag(self) -> None:
        session = Session(make_config(tmp_path=Path(".")))
        self.assertFalse(session.audio_measured)
        session.observe_audio_frame(struct.pack("<hh", 100, -100))
        self.assertTrue(session.audio_measured)
        self.assertEqual(session.audio_peak_abs_max, 100)

    def test_quiet_audio_warning(self) -> None:
        session = Session(make_config(tmp_path=Path(".")))
        session.observe_audio_frame(struct.pack("<hh", 100, -100))
        self.assertIn("very quiet", session.audio_level_warning())

    def test_artifact_store_writes_recording(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            config = make_config(tmp_path=tmp)
            session = Session(config)
            session.add_transcript(Transcript("primary", "m", "asr-final", "some text"))
            session.observe_audio_frame(struct.pack("<hh", 1000, -1000))
            ArtifactStore(config.artifacts_dir, config.archive_dir).commit(session)
            self.assertTrue((tmp / "recording.latest.wav").exists())
            self.assertTrue((tmp / "raw.latest").read_text().startswith("some text"))


class AsrSourcePresentationTests(unittest.TestCase):
    def test_unnoted_model_uses_filename_and_no_hint(self) -> None:
        from speech_note.config import asr_source_presentation

        # An un-annotated source carries no reliability claim — no source is
        # privileged as "most accurate".
        display, hint = asr_source_presentation("medium-q8_0", "ggml-medium-q8_0.bin")
        self.assertEqual(display, "ggml-medium-q8_0.bin")
        self.assertIsNone(hint)

    def test_turbo_q5_k_gets_display_name_and_repetition_warning(self) -> None:
        from speech_note.config import asr_source_presentation

        display, hint = asr_source_presentation("large-v3-turbo-q5_k", "ggml-large-v3-turbo-q5_k.bin")
        self.assertEqual(display, "Whisper Large V3 Turbo Q5_K")
        assert hint is not None
        self.assertIn("repeat", hint.lower())
        self.assertIn("ignore it", hint.lower())

    def test_warning_reaches_the_cleanup_prompt(self) -> None:
        from speech_note.config import asr_source_presentation
        from speech_note.organizer import build_user_prompt

        display, hint = asr_source_presentation("large-v3-turbo-q5_k", "x.bin")
        prompt = build_user_prompt([
            Transcript("asr1", display, "asr-final", "hello world", quality_hint=hint),
        ])
        self.assertIn("Whisper Large V3 Turbo Q5_K", prompt)
        self.assertIn("tendency to repeat", prompt)


class ModelConsentTests(unittest.TestCase):
    def test_auto_yes_skips_prompt(self) -> None:
        from speech_note import models

        self.assertTrue(models.consent_to_download("m", "/d", auto_yes=True))

    def test_non_interactive_refuses(self) -> None:
        from speech_note import models

        self.assertFalse(
            models.consent_to_download("m", "/d", auto_yes=False, interactive=False)
        )

    def test_require_consent_raises_when_declined(self) -> None:
        from speech_note import models

        with self.assertRaises(models.ModelInstallDeclined):
            models.require_consent("m", "/d", auto_yes=False, interactive=False)

    def test_load_or_install_uses_cache_without_consent(self) -> None:
        from speech_note import models

        calls: list[bool] = []
        result = models.load_or_install(
            lambda local_only: (calls.append(local_only), "loaded")[1],
            label="m", dest="/d", auto_yes=False,
        )
        self.assertEqual(result, "loaded")
        self.assertEqual(calls, [True])  # only the offline attempt, no download

    def test_load_or_install_downloads_on_miss_with_auto_yes(self) -> None:
        from speech_note import models

        calls: list[bool] = []

        def loader(local_only: bool) -> str:
            calls.append(local_only)
            if local_only:
                raise FileNotFoundError("cache miss")
            return "downloaded"

        result = models.load_or_install(loader, label="m", dest="/d", auto_yes=True)
        self.assertEqual(result, "downloaded")
        self.assertEqual(calls, [True, False])  # offline miss, then download

    def test_load_or_install_raises_when_declined(self) -> None:
        from speech_note import models

        def loader(local_only: bool) -> str:
            if local_only:
                raise FileNotFoundError("cache miss")
            return "downloaded"

        with self.assertRaises(models.ModelInstallDeclined):
            models.load_or_install(
                loader, label="m", dest="/d", auto_yes=False, interactive=False
            )


class HardwareWarningTests(unittest.TestCase):
    def test_detect_runs_without_error(self) -> None:
        from speech_note import hardware

        result = hardware.detect_unsupported_gpu()
        self.assertTrue(result is None or isinstance(result, str))

    def test_warning_points_to_readme_when_detected(self) -> None:
        from speech_note import hardware

        buf = io.StringIO()
        with mock.patch.object(hardware, "detect_unsupported_gpu", return_value="an NVIDIA GPU"):
            hardware.warn_on_unsupported_gpu(out=buf)
        text = buf.getvalue()
        self.assertIn("NVIDIA", text)
        self.assertIn("README", text)

    def test_no_warning_when_nothing_detected(self) -> None:
        from speech_note import hardware

        buf = io.StringIO()
        with mock.patch.object(hardware, "detect_unsupported_gpu", return_value=None):
            hardware.warn_on_unsupported_gpu(out=buf)
        self.assertEqual(buf.getvalue(), "")

    def test_warning_suppressed_by_env(self) -> None:
        from speech_note import hardware

        buf = io.StringIO()
        with mock.patch.object(hardware, "detect_unsupported_gpu", return_value="an NVIDIA GPU"), \
             mock.patch.dict(os.environ, {"SPEECH_NOTE_NO_GPU_WARNING": "1"}):
            hardware.warn_on_unsupported_gpu(out=buf)
        self.assertEqual(buf.getvalue(), "")


class OpenRouterAsrTests(unittest.TestCase):
    """The 'openrouter' ASR backend: a whole-file audio-LLM source over the network."""

    def _transcriber(self) -> OpenRouterTranscriber:
        return OpenRouterTranscriber(
            model_name="google/gemini-3-flash-preview",
            api_base="https://openrouter.ai/api/v1/chat/completions",
            auth_env="OPENROUTER_API_KEY",
            timeout=120.0,
            mp3_sample_rate=16_000,
            max_output_tokens=16_384,
        )

    def test_registry_entry_is_in_process_network_source(self) -> None:
        spec = ASR_BACKENDS["openrouter"]
        self.assertTrue(spec.in_process)
        self.assertEqual(spec.device_kind, "net")

    def test_source_resolves_to_net_device(self) -> None:
        source = parse_asr_source("openrouter")
        self.assertEqual(source.device, "net")
        self.assertEqual(source.device_kind, "net")  # own scheduling group → overlaps local
        self.assertEqual(source.model, "google/gemini-3-flash-preview")

    def test_build_transcriber_constructs_openrouter(self) -> None:
        transcriber = build_transcriber(make_config(), parse_asr_source("openrouter"))
        assert isinstance(transcriber, OpenRouterTranscriber)
        self.assertEqual(transcriber.model_name, "google/gemini-3-flash-preview")
        self.assertEqual(transcriber.mp3_sample_rate, 16_000)

    def test_transcribe_sends_audio_chat_request(self) -> None:
        captured: dict = {}

        class FakeClient:
            def __init__(self, **kwargs) -> None:
                captured["init"] = kwargs

            def chat(self, messages, *, max_tokens, timeout, **_kwargs):
                captured["messages"] = messages
                captured["max_tokens"] = max_tokens
                captured["timeout"] = timeout
                return ChatResponse(
                    content="  hello   world  ",
                    served_model="google/gemini-3-flash-preview",
                    finish_reason="stop",
                )

        def fake_encode(source, target, *, sample_rate) -> None:
            self.assertEqual(sample_rate, 16_000)
            Path(target).write_bytes(b"FAKEAUDIO")

        with mock.patch("speech_note.organizer.ChatClient", FakeClient), \
             mock.patch("speech_note.audio.encode_to_mp3", fake_encode):
            text = self._transcriber().transcribe_file(
                Path("/tmp/x.wav"), "en", duration_seconds=240.0
            )

        self.assertEqual(text, "hello world")  # normalize_spacing applied
        self.assertEqual(captured["init"]["models"], ["google/gemini-3-flash-preview"])
        messages = captured["messages"]
        self.assertEqual(messages[0]["role"], "system")
        audio_parts = [p for p in messages[1]["content"] if p["type"] == "input_audio"]
        self.assertEqual(len(audio_parts), 1)
        self.assertEqual(audio_parts[0]["input_audio"]["format"], "mp3")
        import base64

        self.assertEqual(base64.b64decode(audio_parts[0]["input_audio"]["data"]), b"FAKEAUDIO")
        # Timeout grows past the floor with duration but stays bounded.
        self.assertGreaterEqual(captured["timeout"], 120.0)
        self.assertLessEqual(captured["timeout"], 900.0)

    def test_ensure_downloaded_requires_key(self) -> None:
        transcriber = self._transcriber()
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                transcriber.ensure_downloaded()
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-x"}, clear=True):
            transcriber.ensure_downloaded()  # no raise


class OnlinePaidAsrTests(unittest.TestCase):
    def test_online_paid_leads_with_openrouter_keeping_local_peers(self) -> None:
        config = make_config("--online-paid")
        backends = [s.backend for s in config.asr_sources]
        self.assertEqual(backends[0], "openrouter")  # default lead = the paid source
        self.assertIn("whisper-cpp", backends)  # local peers still corroborate / fall back
        self.assertIn("sherpa", backends)
        self.assertEqual(config.organizer_provider, "openrouter")

    def test_other_modes_keep_local_only_default(self) -> None:
        self.assertEqual(
            [s.backend for s in make_config().asr_sources], ["whisper-cpp", "sherpa"]
        )
        self.assertNotIn(
            "openrouter", [s.backend for s in make_config("--online-free").asr_sources]
        )

    def test_explicit_asr_overrides_online_paid_default(self) -> None:
        config = make_config("--online-paid", "--asr", "sherpa")
        self.assertEqual([s.backend for s in config.asr_sources], ["sherpa"])

    def test_openrouter_asr_requires_key(self) -> None:
        config = make_config("--asr", "openrouter", "--input-text", "x")
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SystemExit):
                validate(config)
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-x"}, clear=True):
            validate(config)  # no raise


class UnifiedInputTests(unittest.TestCase):
    def test_audio_path_routes_to_input_file(self) -> None:
        config = make_config("--input", "/tmp/rec.m4a")
        self.assertEqual(config.input_file, Path("/tmp/rec.m4a"))
        self.assertIsNone(config.input_archive)

    def test_zip_extension_routes_to_archive(self) -> None:
        config = make_config("-i", "/tmp/bundle.zip")
        self.assertEqual(config.input_archive, Path("/tmp/bundle.zip"))
        self.assertIsNone(config.input_file)

    def test_real_zip_detected_without_zip_extension(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle.dat"
            with zipfile.ZipFile(bundle, "w") as archive:
                archive.writestr("note.txt", "hi")
            config = make_config("--input", str(bundle))
        self.assertEqual(config.input_archive, bundle)
        self.assertIsNone(config.input_file)

    def test_fold_clears_dests_when_no_input(self) -> None:
        args = parse_args([])
        fold_unified_input(args)
        self.assertIsNone(args.input_file)
        self.assertIsNone(args.input_archive)


class ShortOptionTests(unittest.TestCase):
    def test_common_short_flags(self) -> None:
        args = parse_args(["-f", "-o", "out.txt", "-i", "rec.m4a", "-l", "es", "-a", "sherpa"])
        self.assertTrue(args.full_auto)
        self.assertEqual(args.output, Path("out.txt"))
        self.assertEqual(args.input, Path("rec.m4a"))
        self.assertEqual(args.language, "es")
        self.assertEqual(args.asr, ["sherpa"])

    def test_connectivity_short_flags(self) -> None:
        self.assertEqual(parse_args(["-P"]).connectivity, "online-paid")
        self.assertEqual(parse_args(["-F"]).connectivity, "online-free")
        self.assertEqual(parse_args(["-O"]).connectivity, "offline")

    def test_input_text_short_flag(self) -> None:
        self.assertEqual(parse_args(["-t", "hello"]).dry_run_text, "hello")


if __name__ == "__main__":
    unittest.main()
