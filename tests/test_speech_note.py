"""Tests for the speech_note package.

Configs are built through the real parse/resolve path (never hand-made
namespaces), and assertions target observable behavior, not implementation
strings.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import re
import struct
import tempfile
import types
import unittest
import wave
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import cast
from unittest import mock

from speech_note import audio, naming, textproc
from speech_note.capture import SegmentCollector
from speech_note.cli import Config, parse_args, resolve_config, validate
from speech_note.model import Transcript
from speech_note.chat import ChatClient, ChatResponse, request_timeout_seconds
from speech_note.llama_server import default_server_command, ensure_default_cleanup_model
from speech_note.organizer import (
    SYSTEM_PROMPT,
    Organizer,
    build_user_prompt,
    cleanup_request_plan,
)
from speech_note.cli import _interactive_capture_setup, fold_unified_input
from speech_note.pipeline import (
    _batch_item_config,
    cleanup_sources,
    discover_archive_inputs,
    discover_directory_inputs,
    extract_subtitle_text,
    extract_zip_safely,
    review_panels,
    run_dry_text_pipeline,
    run_transcript_pipeline,
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
from speech_note.terminal import StatusDisplay, StatusTask, render_status_line
from speech_note.session import ArtifactStore, Session
from speech_note.transcribers import (
    FasterWhisperTranscriber,
    OpenRouterTranscriber,
    SherpaTranscriber,
    Transcriber,
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
        assert stats.peak_dbfs is not None
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
            ceiling = 10 ** (audio.NORMALIZE_MAX_GAIN_DB / 20.0)
            self.assertLessEqual(result.gain, ceiling + 1e-9)

    def _write_samples(self, path: Path, samples: "list[int]") -> None:
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(struct.pack(f"<{len(samples)}h", *samples))

    def test_normalize_is_not_vetoed_by_one_transient(self) -> None:
        """A single full-scale sample must not stop quiet speech being amplified.

        Regression test: the old global gain took its peak term from the absolute
        maximum sample, so one bump or click drove the gain below 1 and the pipeline
        handed ASR audio *quieter* than the input.
        """
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "in.wav"
            target = Path(tmp) / "out.wav"
            quiet = [300 if index % 2 else -300 for index in range(16000)]
            quiet[8000] = 32767  # the transient
            self._write_samples(source, quiet)
            result = audio.normalize_pcm_wav(source, target)
            self.assertTrue(result.applied)
            self.assertGreater(result.gain, 1.0, "one transient must not veto the gain")
            with wave.open(str(target), "rb") as handle:
                out = handle.readframes(handle.getnframes())
            # The quiet body got louder even though the input already peaked at full scale.
            body = struct.unpack(f"<{len(out) // 2}h", out)[:4000]
            self.assertGreater(max(abs(value) for value in body), 300)

    @staticmethod
    def _tone(amplitude: int, seconds: float) -> "list[int]":
        count = int(16000 * seconds)
        return [amplitude if index % 2 else -amplitude for index in range(count)]

    @staticmethod
    def _rms(values) -> float:
        return (sum(float(v) * v for v in values) / max(1, len(values))) ** 0.5

    def _normalized(self, samples: "list[int]", tmp: str):
        source, target = Path(tmp) / "in.wav", Path(tmp) / "out.wav"
        self._write_samples(source, samples)
        result = audio.normalize_pcm_wav(source, target)
        with wave.open(str(target), "rb") as handle:
            out = struct.unpack(f"<{handle.getnframes()}h", handle.readframes(handle.getnframes()))
        return result, out

    def test_normalize_corrects_a_sustained_level_change(self) -> None:
        """Minutes of quiet end up at normal level — the pocketed-mic case.

        The gain ride is deliberately slow, so the change has to persist on that
        timescale; a few seconds would not (and should not) move it.
        """
        with tempfile.TemporaryDirectory() as tmp:
            samples = self._tone(6000, 180) + self._tone(600, 180)
            result, out = self._normalized(samples, tmp)
            self.assertTrue(result.applied)
            loud = self._rms(out[16000 * 60 : 16000 * 90])
            faint = self._rms(out[16000 * 270 : 16000 * 300])
            after_ratio = max(loud, faint) / max(1.0, min(loud, faint))
            self.assertLess(after_ratio, 10.0 / 2, "the sustained gap should mostly close")

    def test_normalize_rides_through_a_brief_dip(self) -> None:
        """A couple of seconds of quiet must NOT be chased up to full level.

        Chasing short excursions is what produces pumping, and it tells the ASR feature
        extractor that a dropped syllable was a whole sentence.
        """
        with tempfile.TemporaryDirectory() as tmp:
            samples = self._tone(6000, 120) + self._tone(600, 3) + self._tone(6000, 120)
            _result, out = self._normalized(samples, tmp)
            steady = self._rms(out[16000 * 30 : 16000 * 60])
            dip = self._rms(out[16000 * 121 : 16000 * 122])
            self.assertLess(dip, steady / 3, "the brief dip should stay quiet")

    def test_normalize_never_gates_or_cuts_audio(self) -> None:
        """Speech is often quieter than the noise around it and is still recoverable.

        Nothing may be classified as non-speech and removed: the output must be the input
        times a gain curve, so its shape is preserved sample for sample.
        """
        with tempfile.TemporaryDirectory() as tmp:
            rng = __import__("random").Random(0)
            # Quiet "speech" riding under much louder "noise".
            samples = [
                max(-32000, min(32000, int(rng.gauss(0, 4000)) + (300 if i % 2 else -300)))
                for i in range(16000 * 20)
            ]
            _result, out = self._normalized(samples, tmp)
            self.assertEqual(len(out), len(samples))
            # Every non-zero input sample stays non-zero and keeps its sign: a gate would
            # zero whole stretches, and nothing here may be silenced.
            checked = 0
            for index in range(0, len(samples), 997):
                if abs(samples[index]) > 500:
                    self.assertNotEqual(out[index], 0, f"sample {index} was silenced")
                    self.assertEqual(
                        samples[index] > 0, out[index] > 0, f"sample {index} changed sign"
                    )
                    checked += 1
            self.assertGreater(checked, 100)

    def test_normalize_compresses_peaks_instead_of_dropping_the_gain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            samples = self._tone(400, 30)
            samples[16000 * 15] = 32767  # one full-scale transient
            result, _out = self._normalized(samples, tmp)
            self.assertGreater(result.gain, 1.0, "one transient must not veto the gain")
            self.assertGreater(result.compressed_db, 0.0, "the peak should be compressed")


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
            "--extra-transcript", "/tmp/compute-orientation.whisper.txt",
            "--extra-transcript", "/tmp/compute-orientation.parakeet.txt",
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
        self.assertEqual(config.organizer.provider, "local")

    def test_online_free_flag_uses_free_chain(self) -> None:
        from speech_note import config as cfg
        config = make_config("--online-free")
        self.assertEqual(config.organizer.provider, "openrouter")
        self.assertEqual(config.organizer.models, tuple(cfg.OPENROUTER_FREE_MODELS))
        self.assertGreaterEqual(len(config.organizer.models), 5)  # deep fallback chain
        self.assertTrue(all(m.endswith(":free") for m in config.organizer.models))

    def test_online_paid_flag_leads_paid_then_falls_back_to_free(self) -> None:
        from speech_note import config as cfg
        config = make_config("--online-paid")
        self.assertEqual(config.organizer.provider, "openrouter")
        self.assertEqual(config.organizer.models[0], "deepseek/deepseek-v3.2")
        # paid leads, but the free chain is appended as a fallback for paid outages
        self.assertTrue(any(m.endswith(":free") for m in config.organizer.models))
        for free in cfg.OPENROUTER_FREE_MODELS:
            self.assertIn(free, config.organizer.models)

    def test_explicit_model_overrides_connectivity_mode(self) -> None:
        config = make_config("--online-paid", "--organizer-model", "custom/model")
        self.assertEqual(config.organizer.models, ("custom/model",))

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
        self.assertIn("openrouter.ai", config.organizer.api_base)
        self.assertEqual(config.organizer.context_tokens, 262144)
        self.assertEqual(config.organizer.max_output_tokens, 65536)
        self.assertEqual(config.organizer.auth_env, "OPENROUTER_API_KEY")
        self.assertTrue(config.organizer.models)

    def test_explicit_context_tokens_survive_provider_defaults(self) -> None:
        config = make_config(
            "--organizer-provider", "openrouter",
            "--organizer-context-tokens", "65536",
        )
        self.assertEqual(config.organizer.context_tokens, 65536)

    def test_full_auto_redirects_artifacts_and_names_output(self) -> None:
        config = make_config("--full-auto", "--input", "/recordings/My Lecture.m4a")
        self.assertNotEqual(config.artifacts_dir, Path("."))
        assert config.output is not None
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

    def test_validate_no_asr_with_audio_needs_a_transcript(self) -> None:
        # --no-asr skips transcription, so an audio file with nothing else is empty.
        with self.assertRaises(SystemExit):
            validate(make_config("--no-asr", "--input", "/a.wav"))
        # ...but audio + a transcript to clean up is fine.
        validate(make_config("--no-asr", "--input", "/a.wav", "-x", "/t.txt"))

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


class DirectoryBatchTests(unittest.TestCase):
    """Undocumented batch mode: -i on a directory processes each audio/zip inside."""

    def _dir(self) -> Path:
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        (tmp / "a.wav").write_bytes(b"")
        (tmp / "b.zip").write_bytes(b"")
        (tmp / "sub").mkdir()
        (tmp / "sub" / "nested.wav").write_bytes(b"")  # subdir ignored
        # Decoys of the kind a batch writes back and must never re-ingest:
        (tmp / "previous-clean.txt").write_text("out")
        (tmp / "previous-diagnostics.json").write_text("{}")
        return tmp

    def test_discovery_takes_only_top_level_audio_and_zip(self) -> None:
        names = [p.name for p in discover_directory_inputs(self._dir())]
        self.assertEqual(names, ["a.wav", "b.zip"])  # no .txt/.json, no subdir

    def test_directory_input_requires_full_auto(self) -> None:
        tmp = self._dir()
        with self.assertRaises(SystemExit):
            validate(make_config("--input", str(tmp)))
        validate(make_config("--full-auto", "--input", str(tmp)))  # ok

    def test_batch_item_config_routes_and_names_into_dir(self) -> None:
        tmp = self._dir()
        base = make_config("--full-auto", "--input", str(tmp))
        self.assertIsNotNone(base.input_dir)
        self.assertIsNone(base.output)  # batch config itself isn't auto-named

        zip_item = _batch_item_config(base, tmp / "b.zip")
        self.assertEqual(zip_item.input_archive, tmp / "b.zip")
        self.assertIsNone(zip_item.input_file)
        self.assertIsNone(zip_item.input_dir)
        self.assertIsNone(zip_item.output)  # archive self-names post-extraction
        self.assertEqual(zip_item.full_auto_output_dir, tmp)

        wav_item = _batch_item_config(base, tmp / "a.wav")
        self.assertEqual(wav_item.input_file, tmp / "a.wav")
        self.assertIsNone(wav_item.input_archive)
        assert wav_item.output is not None
        self.assertEqual(wav_item.output.parent, tmp)  # named into the directory
        self.assertEqual(wav_item.output.name, "a-clean.txt")


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
        assert outcome.transcript is not None
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
        assert outcome.transcript is not None
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
        assert outcome.error is not None
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
        self.assertEqual(prepared[0].label, "asr1")
        self.assertIsNone(prepared[0].error)

    def test_failed_prepare_records_error_without_running_transcribe(self) -> None:
        source = parse_asr_source("sherpa")

        class _Boom(Transcriber):
            def ensure_downloaded(self) -> None:
                raise RuntimeError("download declined")

            def transcribe_file(self, *args: object, **kwargs: object) -> str:
                raise AssertionError("must not transcribe after a failed prepare")

        prepared = prepare_sources([(source, _Boom())])
        self.assertEqual(prepared[0].error, "download declined")
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
        panels = review_panels(session)
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
    def _organizer() -> tuple[Organizer, mock.Mock]:
        supervisor = mock.Mock()
        fake = cast(Organizer, types.SimpleNamespace(supervisor=supervisor))
        return fake, supervisor

    def test_no_prewarm_flag_skips_background_load(self) -> None:
        config = make_config("--organizer-mode", "llama", "--organizer-no-prewarm")
        self.assertFalse(config.organizer.prewarm)
        organizer, supervisor = self._organizer()
        self.assertIsNone(start_organizer_prewarm(config, organizer))
        supervisor.ensure_running.assert_not_called()

    def test_default_prewarms_in_background(self) -> None:
        config = make_config("--organizer-mode", "llama")
        self.assertTrue(config.organizer.prewarm)
        organizer, supervisor = self._organizer()
        with redirect_stderr(io.StringIO()):
            thread = start_organizer_prewarm(config, organizer)
            assert thread is not None
            thread.join(timeout=5)
        supervisor.ensure_running.assert_called_once()


class OrganizerPromptTests(unittest.TestCase):
    def test_cleanup_plan_reserves_room_for_hidden_reasoning(self) -> None:
        sources = [
            Transcript("external", "archive.txt", "external", "word " * 510),
            Transcript("whisper", "medium", "asr-final", "word " * 420),
            Transcript("parakeet", "parakeet", "asr-final", "word " * 460),
        ]
        plan = cleanup_request_plan(
            sources,
            context_tokens=262_144,
            max_output_tokens=65_536,
        )
        self.assertEqual(plan.requested_output_tokens, 4_096)
        self.assertFalse(plan.output_cap_limited)

    def test_cleanup_plan_treats_insufficient_reasoning_reserve_as_cap_risk(self) -> None:
        sources = [Transcript("external", "archive.txt", "external", "word " * 140)]
        plan = cleanup_request_plan(
            sources,
            context_tokens=262_144,
            max_output_tokens=2_048,
        )
        self.assertEqual(plan.requested_output_tokens, 2_048)
        self.assertTrue(plan.output_cap_limited)

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
        # Length is anchored to the MEAN source length (robust to one filler-heavy
        # source): mean of 100/200/150 words is 150.
        self.assertIn("On average the sources are 150 words", prompt)

    def test_prompt_is_union_recall_not_consensus(self) -> None:
        """Single-source content must be KEPT (sources differ by sensitivity, not
        reliability) — the prompt must not tell the model to drop minority content."""
        sources = [
            Transcript("whisper", "whisper", "asr-final", "word " * 80),
            Transcript("gemini", "gemini", "asr-final", "word " * 160),
        ]
        prompt = build_user_prompt(sources)
        lowered = prompt.lower()
        # Keeps single-source content rather than treating consensus as truth.
        self.assertIn("single source", lowered)
        self.assertIn("sensitivity", lowered)
        self.assertIn("keep that content", lowered)
        # Anchored to the mean source length (mean of 80/160 is 120), never the shortest.
        self.assertIn("on average the sources are 120 words", lowered)
        self.assertNotIn("shortest source", lowered)
        # Two orthogonal jobs: strip disfluencies, keep content.
        self.assertIn("disfluencies", lowered)

    def test_system_prompt_frames_union_reconstruction(self) -> None:
        lowered = SYSTEM_PROMPT.lower()
        self.assertIn("sensitivity", lowered)
        self.assertIn("union", lowered)
        # The asymmetric cost: dropping real content is worse than keeping uncertain.
        self.assertIn("losing real content", lowered)
        self.assertNotIn("conservative transcript repair", lowered)

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
            "speech_note.chat.requests.get", side_effect=Exception("no catalog")
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
        with mock.patch("speech_note.chat.requests.post", return_value=response):
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
        with mock.patch("speech_note.chat.requests.post", side_effect=fake_post):
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
            with mock.patch("speech_note.chat.requests.post", side_effect=fake_post):
                outcome = organizer.cleanup(sources)
        self.assertEqual(calls, ["stale:free", "working:free"])
        self.assertTrue(outcome.ok)

    def test_http_200_error_body_falls_through_to_next_model(self) -> None:
        # OpenRouter can return HTTP 200 with an {"error": ...} body and no choices
        # (upstream rate limit / outage). That must fall through, not crash on
        # data["choices"].
        organizer = make_llm_organizer("first,second")
        calls: list[str] = []

        def fake_post(_url, *, data, **_kwargs):
            model = json.loads(data)["model"]
            calls.append(model)
            if model == "first":
                return fake_response(200, {"error": {"message": "rate-limited upstream", "code": 429}})
            return fake_response(200, self.chat_payload("clean " * 90, model="second"))

        sources = [Transcript("primary", "whisper", "asr-final", "word " * 100)]
        with mock.patch("speech_note.chat.requests.post", side_effect=fake_post):
            outcome = organizer.cleanup(sources)
        self.assertEqual(calls, ["first", "second"])
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.served_model, "second")

    def test_http_200_error_body_on_last_model_is_error_not_crash(self) -> None:
        organizer = make_llm_organizer("only")
        response = fake_response(200, {"error": {"message": "upstream down"}})
        sources = [Transcript("primary", "whisper", "asr-final", "word " * 100)]
        with mock.patch("speech_note.chat.requests.post", return_value=response):
            outcome = organizer.cleanup(sources)
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.method, "error")
        assert outcome.error is not None
        self.assertIn("no choices", outcome.error)

    def test_http_200_non_json_body_falls_through(self) -> None:
        organizer = make_llm_organizer("first,second")

        def fake_post(_url, *, data, **_kwargs):
            model = json.loads(data)["model"]
            if model == "first":
                bad = mock.Mock()
                bad.ok = True
                bad.status_code = 200
                bad.text = "<html>502 Bad Gateway</html>"
                bad.json.side_effect = ValueError("not json")
                return bad
            return fake_response(200, self.chat_payload("clean " * 90, model="second"))

        sources = [Transcript("primary", "whisper", "asr-final", "word " * 100)]
        with mock.patch("speech_note.chat.requests.post", side_effect=fake_post):
            outcome = organizer.cleanup(sources)
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.served_model, "second")

    def test_truncated_output_is_an_error(self) -> None:
        organizer = make_llm_organizer()
        sources = [Transcript("primary", "whisper", "asr-final", "word " * 100)]
        response = fake_response(200, self.chat_payload("clean " * 90, finish="length"))
        with mock.patch("speech_note.chat.requests.post", return_value=response):
            outcome = organizer.cleanup(sources)
        self.assertTrue(outcome.flagged_short)
        assert outcome.error is not None
        self.assertIn("truncated", outcome.error)

    def test_short_output_is_flagged_but_kept(self) -> None:
        organizer = make_llm_organizer()
        sources = [Transcript("primary", "whisper", "asr-final", "word " * 100)]
        response = fake_response(200, self.chat_payload("too short"))
        with mock.patch("speech_note.chat.requests.post", return_value=response):
            outcome = organizer.cleanup(sources)
        self.assertEqual(outcome.text, "too short")
        self.assertTrue(outcome.flagged_short)
        self.assertIsNone(outcome.error)
        assert outcome.warning is not None
        self.assertIn("short", outcome.warning)

    def test_mid_sentence_output_is_flagged_but_kept(self) -> None:
        organizer = make_llm_organizer()
        sources = [Transcript("primary", "whisper", "asr-final", "word " * 100)]
        # Long enough to pass the length ratio, but cut off mid-sentence (no terminal
        # punctuation) — the rita-c failure mode that finish_reason="stop" hid.
        cut_off = "clean " * 89 + "and then I"
        response = fake_response(200, self.chat_payload(cut_off))
        with mock.patch("speech_note.chat.requests.post", return_value=response):
            outcome = organizer.cleanup(sources)
        self.assertEqual(outcome.text, cut_off.strip())  # output preserved, not deleted
        self.assertTrue(outcome.flagged_short)
        self.assertIsNone(outcome.error)  # a flag, never an error/deletion
        assert outcome.warning is not None
        self.assertIn("mid-sentence", outcome.warning)

    def test_complete_output_is_not_flagged_mid_sentence(self) -> None:
        # A faithful trail-off that the speaker actually made still ends in '.', '?', '!'
        # or '…', so it must NOT be flagged.
        for ending in ("the time is...", "good night.", "what now?", "part of the like…", 'he said "hello."'):
            organizer = make_llm_organizer()
            sources = [Transcript("primary", "whisper", "asr-final", "word " * 10)]
            response = fake_response(200, self.chat_payload("clean " * 20 + ending))
            with mock.patch("speech_note.chat.requests.post", return_value=response):
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
        with mock.patch("speech_note.chat.requests.post", side_effect=fake_post):
            organizer.cleanup(sources)
        self.assertIn("Source 1", prompts[0])
        self.assertNotIn("Source 2", prompts[0])


class ChatClientCatalogTests(unittest.TestCase):
    def test_reasoning_effort_is_sent_when_configured(self) -> None:
        client = ChatClient(
            api_base="https://openrouter.ai/api/v1/chat/completions",
            models=["model"],
            timeout=0.01,
            reasoning_effort="none",
        )
        response = fake_response(
            200,
            {
                "model": "model",
                "choices": [{"message": {"content": "clean"}, "finish_reason": "stop"}],
            },
        )
        with mock.patch("speech_note.chat.requests.post", return_value=response) as post:
            client.chat([{"role": "user", "content": "transcript"}], max_tokens=4096, timeout=1)
        payload = json.loads(post.call_args.kwargs["data"])
        self.assertEqual(payload["reasoning"], {"effort": "none"})

    def test_stale_models_filtered_against_catalog(self) -> None:
        client = ChatClient(
            api_base="https://openrouter.ai/api/v1/chat/completions",
            models=["gone:free", "alive:free"],
            timeout=0.01,
        )
        catalog = fake_response(200, {"data": [{"id": "alive:free"}, {"id": "other"}]})
        catalog.raise_for_status = mock.Mock()
        with mock.patch("speech_note.chat.requests.get", return_value=catalog):
            self.assertEqual(client.candidate_models(), ["alive:free"])

    def test_catalog_failure_keeps_configured_list(self) -> None:
        client = ChatClient(
            api_base="http://127.0.0.1:9/v1/chat/completions", models=["a", "b"], timeout=0.01
        )
        with mock.patch(
            "speech_note.chat.requests.get", side_effect=Exception("offline")
        ):
            self.assertEqual(client.candidate_models(), ["a", "b"])

    def test_fully_stale_catalog_keeps_configured_list(self) -> None:
        client = ChatClient(
            api_base="http://127.0.0.1:9/v1/chat/completions", models=["a"], timeout=0.01
        )
        catalog = fake_response(200, {"data": [{"id": "other"}]})
        catalog.raise_for_status = mock.Mock()
        with mock.patch("speech_note.chat.requests.get", return_value=catalog):
            self.assertEqual(client.candidate_models(), ["a"])


class ServerCommandTests(unittest.TestCase):
    def test_default_server_command_kv_offload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_text("fake")
            with mock.patch("speech_note.llama_server.DEFAULT_GGUF_MODEL", model):
                with mock.patch("speech_note.llama_server.shutil.which", return_value="/bin/llama-server"):
                    # KV offload is on by default (fixed upstream; ~1.5x faster).
                    command = default_server_command(context_tokens=4096)
                    assert command is not None
                    self.assertNotIn("--no-kv-offload", command)
                    # The workaround for older stacks is still reachable.
                    command = default_server_command(context_tokens=4096, kv_offload=False)
                    assert command is not None
                    self.assertIn("--no-kv-offload", command)

    def test_default_server_command_missing_model(self) -> None:
        with mock.patch("speech_note.llama_server.DEFAULT_GGUF_MODEL", Path("/nonexistent.gguf")):
            self.assertIsNone(default_server_command(context_tokens=4096))

    def test_default_server_command_model_path_override(self) -> None:
        # A stronger local cleanup model can be dropped in via model_path (the
        # --organizer-gguf flag), overriding the bundled gemma-E2B default.
        with tempfile.TemporaryDirectory() as tmp:
            stronger = Path(tmp) / "gemma-31b.gguf"
            stronger.write_text("fake")
            with mock.patch("speech_note.llama_server.DEFAULT_GGUF_MODEL", Path("/nonexistent.gguf")):
                with mock.patch("speech_note.llama_server.shutil.which", return_value="/bin/llama-server"):
                    command = default_server_command(context_tokens=4096, model_path=stronger)
                    assert command is not None
                    self.assertIn(str(stronger), command)

    def test_cli_organizer_gguf_flag(self) -> None:
        config = make_config("--organizer-gguf", "/models/big.gguf")
        self.assertEqual(config.organizer.gguf, Path("/models/big.gguf"))
        self.assertIsNone(make_config().organizer.gguf)

    def test_ensure_default_cleanup_model_already_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "cleanup.gguf"
            model.write_text("fake")
            with mock.patch("speech_note.llama_server.DEFAULT_GGUF_MODEL", model):
                # Present already: returns True without importing/calling the downloader.
                self.assertTrue(ensure_default_cleanup_model(auto_yes=True))

    def test_ensure_default_cleanup_model_declined(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "missing.gguf"
            with mock.patch("speech_note.llama_server.DEFAULT_GGUF_MODEL", model):
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
            with mock.patch("speech_note.llama_server.DEFAULT_GGUF_MODEL", model):
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
            self.assertEqual(payload["schema"], 4)
            self.assertEqual(payload["cleanup"]["method"], "heuristic")
            self.assertFalse(payload["audio_levels"]["measured"])
            self.assertEqual(payload["transcripts"][0]["label"], "user")
            self.assertEqual(payload["transcripts"][0]["kind"], "user")
            self.assertIn("config", payload)
            self.assertEqual(payload["errors"], [])

    def test_output_flag_writes_clean_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            output = tmp / "out" / "note.txt"
            session = self.run_dry(tmp, "--output", str(output))
            self.assertTrue(output.exists())
            assert session.cleanup is not None
            self.assertEqual(output.read_text().strip(), session.cleanup.text)

    def test_export_sources_writes_every_fed_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            extra = tmp / "google.txt"
            extra.write_text("a google transcript that was also fed in")
            export = tmp / "sources"
            session = self.run_dry(
                tmp, "--extra-transcript", str(extra), "--export-sources", str(export)
            )
            files = sorted(p.name for p in export.iterdir())
            # One file per fed source (the dry-run "primary" user text + the extra),
            # plus the cleaned result. Heuristic cleanup sends no organizer prompt.
            self.assertIn("clean.txt", files)
            source_files = [n for n in files if n != "clean.txt"]
            self.assertEqual(len(source_files), 2)
            self.assertTrue(all(n[:2].isdigit() for n in source_files), files)
            # The extra source round-trips verbatim (body is the transcript alone).
            extra_export = next(export.glob("*google*.txt"))
            self.assertEqual(
                extra_export.read_text().strip(), "a google transcript that was also fed in"
            )
            assert session.cleanup is not None
            self.assertEqual((export / "clean.txt").read_text().strip(), session.cleanup.text)

    def test_no_export_sources_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            session = self.run_dry(tmp)
            self.assertNotIn("exported_sources", session.paths)

    def test_export_sources_writes_cleanup_prompt_for_llm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            export = tmp / "sources"
            config = resolve_config(
                parse_args(
                    [
                        "--input-text", "hello world",
                        "--organizer-mode", "llama",
                        "--organizer-api-base", "http://127.0.0.1:9/v1/chat/completions",
                        "--export-sources", str(export),
                        "--organizer-server-command",  # empty: no server launch
                    ]
                )
            )
            posted: dict[str, object] = {}

            def fake_post(_url, *, data, **_kwargs):
                posted.update(json.loads(data))
                return fake_response(200, {"choices": [
                    {"message": {"content": "hello world."}, "finish_reason": "stop"}
                ]})

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with mock.patch(
                    "speech_note.chat.requests.get",
                    return_value=fake_response(503, {"error": "offline"}),
                ):
                    with mock.patch("speech_note.chat.requests.post", side_effect=fake_post):
                        run_dry_text_pipeline(config)
            prompt = (export / "cleanup-prompt.txt").read_text()
            messages = cast("list[dict[str, str]]", posted["messages"])
            self.assertEqual(prompt, f"[system]\n{messages[0]['content']}\n\n[user]\n{messages[1]['content']}\n")

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
                    "speech_note.chat.requests.get", side_effect=Exception("offline")
                ):
                    with mock.patch(
                        "speech_note.chat.requests.post",
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

    def test_full_auto_cap_risk_exports_sources_and_skips_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            output = tmp / "note-clean.txt"
            config = resolve_config(
                parse_args(
                    [
                        "--input-text", "word " * 1200,
                        "--full-auto",
                        "--output", str(output),
                        "--organizer-mode", "llama",
                        "--organizer-max-output-tokens", "1024",
                        "--organizer-server-command",  # empty: no server launch
                    ]
                )
            )
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with mock.patch("speech_note.chat.requests.post") as post:
                    session = run_dry_text_pipeline(config)
            post.assert_not_called()
            self.assertTrue(session.run_failed)
            self.assertFalse(output.exists())
            sources = tmp / "note-sources"
            self.assertTrue(sources.is_dir())
            self.assertTrue((sources / "cleanup-prompt.txt").exists())
            prompt = (sources / "cleanup-prompt.txt").read_text()
            self.assertIn("[system]", prompt)
            self.assertIn("[user]", prompt)
            self.assertIn("Source 1", prompt)
            source_files = sorted(
                path.name
                for path in sources.glob("*.txt")
                if path.name != "cleanup-prompt.txt"
            )
            self.assertEqual(len(source_files), 1)
            self.assertFalse((sources / "clean.txt").exists())
            error_diag = tmp / "note-clean-diagnostics.json"
            payload = json.loads(error_diag.read_text())
            self.assertEqual(payload["cleanup"]["method"], "skipped-too-large")
            self.assertIn("output token cap", payload["cleanup"]["error"])
            self.assertEqual(payload["paths"]["exported_sources"], str(sources))

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
        warning = session.audio_level_warning()
        assert warning is not None
        self.assertIn("very quiet", warning)

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
        from speech_note import install as models

        self.assertTrue(models.consent_to_download("m", "/d", auto_yes=True))

    def test_non_interactive_refuses(self) -> None:
        from speech_note import install as models

        self.assertFalse(
            models.consent_to_download("m", "/d", auto_yes=False, interactive=False)
        )

    def test_require_consent_raises_when_declined(self) -> None:
        from speech_note import install as models

        with self.assertRaises(models.ModelInstallDeclined):
            models.require_consent("m", "/d", auto_yes=False, interactive=False)

    def test_load_or_install_uses_cache_without_consent(self) -> None:
        from speech_note import install as models

        calls: list[bool] = []
        result = models.load_or_install(
            lambda local_only: (calls.append(local_only), "loaded")[1],
            label="m", dest="/d", auto_yes=False,
        )
        self.assertEqual(result, "loaded")
        self.assertEqual(calls, [True])  # only the offline attempt, no download

    def test_load_or_install_downloads_on_miss_with_auto_yes(self) -> None:
        from speech_note import install as models

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
        from speech_note import install as models

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

        with mock.patch("speech_note.chat.ChatClient", FakeClient), \
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


class OpenRouterSttTranscriberTests(unittest.TestCase):
    """The dedicated hosted-ASR backend (/audio/transcriptions)."""

    def _transcriber(self):
        from speech_note import config as cfg
        from speech_note.transcribers import OpenRouterSttTranscriber

        return OpenRouterSttTranscriber(
            model_name=cfg.OPENROUTER_STT_MODEL,
            api_base=cfg.OPENROUTER_STT_API_BASE,
            auth_env="OPENROUTER_API_KEY",
            timeout=120.0,
            mp3_sample_rate=16_000,
            response_format=cfg.OPENROUTER_STT_RESPONSE_FORMAT,
        )

    def _run(self, transcriber, response, *, duration=240.0) -> tuple[str, dict]:
        captured: dict = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            captured.update(kwargs)
            # requests streams the handle, so read it before the file closes.
            captured["body"] = kwargs["files"]["file"][1].read()
            return response

        def fake_encode(source, target, *, sample_rate) -> None:
            Path(target).write_bytes(b"FAKEAUDIO")

        with mock.patch("requests.post", fake_post), \
             mock.patch("speech_note.audio.encode_to_mp3", fake_encode), \
             mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-x"}, clear=True):
            text = transcriber.transcribe_file(Path("/tmp/x.wav"), "en", duration_seconds=duration)
        return text, captured

    def test_uploads_multipart_not_base64(self) -> None:
        """A 65-minute recording is ~23 MB as base64 JSON, which the gateway 502s on."""
        response = mock.Mock(ok=True, status_code=200)
        response.json.return_value = {"text": "  hello   world  "}
        text, captured = self._run(self._transcriber(), response)
        self.assertEqual(text, "hello world")
        self.assertIn("audio/transcriptions", captured["url"])
        self.assertEqual(captured["body"], b"FAKEAUDIO")  # raw bytes, not base64
        self.assertIsNone(captured.get("json"))
        self.assertEqual(captured["data"]["response_format"], "json")
        self.assertEqual(captured["data"]["language"], "en")
        self.assertGreaterEqual(captured["timeout"], 120.0)

    def test_http_error_raises_so_the_source_fails_not_the_run(self) -> None:
        response = mock.Mock(ok=False, status_code=402, text='{"error":"insufficient balance"}')
        with self.assertRaises(RuntimeError) as caught:
            self._run(self._transcriber(), response)
        self.assertIn("402", str(caught.exception))

    def test_ok_response_without_transcript_is_an_error_not_silence(self) -> None:
        """Returning "" here would be reported as 'no speech detected', which is wrong."""
        response = mock.Mock(ok=True, status_code=200)
        response.json.return_value = {"error": {"message": "upstream failed"}}
        with self.assertRaises(RuntimeError) as caught:
            self._run(self._transcriber(), response)
        self.assertIn("no transcript", str(caught.exception))

    def test_ensure_downloaded_requires_key(self) -> None:
        transcriber = self._transcriber()
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                transcriber.ensure_downloaded()
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-x"}, clear=True):
            transcriber.ensure_downloaded()  # no raise


class OnlinePaidAsrTests(unittest.TestCase):
    def test_online_paid_runs_whisper_and_parakeet_as_hosted_models(self) -> None:
        """--online-paid moves ASR off this machine, to dedicated hosted recognizers.

        Not to the audio-LLM: that fabricates on unintelligible audio and degenerates
        into a repeated word on long files (see config.catalog.OPENROUTER_ASR_MODEL).
        """
        from speech_note import config as cfg

        config = make_config("--online-paid")
        backends = [s.backend for s in config.asr_sources]
        self.assertEqual(backends, ["openrouter-stt", "openrouter-stt", "openrouter"])
        models = [s.model for s in config.asr_sources]
        self.assertEqual(
            models,
            [cfg.OPENROUTER_STT_MODEL, cfg.OPENROUTER_STT_SECOND_MODEL, cfg.OPENROUTER_ASR_MODEL],
        )
        # The audio-LLM is last, so it is never the raw/fallback transcript — list order
        # is the soft preference for that.
        self.assertEqual(backends[-1], "openrouter")
        self.assertEqual(config.organizer.provider, "openrouter")

    def test_hosted_asr_sources_each_get_their_own_scheduling_group(self) -> None:
        """Network sources hold no local device, so they must overlap rather than queue."""
        from speech_note import config as cfg
        from speech_note.asr import PreparedSource, _scheduling_groups

        prepared = [
            PreparedSource(f"asr{index}", source.resolved(), None)
            for index, source in enumerate(cfg.ONLINE_PAID_ASR_SOURCES, start=1)
        ]
        groups = _scheduling_groups(prepared)
        self.assertEqual(len(groups), 3, "hosted calls should not serialize")
        # Local sources still share one group per device kind.
        local = [
            PreparedSource(f"asr{index}", source.resolved(), None)
            for index, source in enumerate(
                (cfg.AsrSource("sherpa", "", "cpu"), cfg.AsrSource("ctc", "", "cpu")), start=1
            )
        ]
        self.assertEqual(len(_scheduling_groups(local)), 1)

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
        for backend in ("openrouter", "openrouter-stt"):
            config = make_config("--asr", backend, "--input-text", "x")
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


class NoAsrTranscriptTests(unittest.TestCase):
    """--no-asr replaces the old primary/secondary-transcript flags: all transcripts
    are equal peers, ASR is just an optional source."""

    def test_no_asr_flag_default_and_set(self) -> None:
        self.assertFalse(make_config().no_asr)
        self.assertTrue(make_config("--no-asr", "-x", "/t.txt").no_asr)

    def test_no_primary_secondary_flags_remain(self) -> None:
        for flag in ("--primary-transcript", "--secondary-transcript"):
            with self.assertRaises(SystemExit):
                parse_args([flag, "/tmp/x.txt"])

    def test_transcript_only_pipeline_cleans_extra_transcripts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            transcript = Path(tmp) / "rec.whisper.txt"
            transcript.write_text("um so i think uh this is the plan")
            config = make_config("-x", str(transcript), tmp_path=Path(tmp))
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                session = run_transcript_pipeline(config)
            self.assertFalse(session.run_failed)
            self.assertTrue(session.raw_text().strip())

    def test_no_asr_file_run_skips_asr_and_uses_transcript(self) -> None:
        # --no-asr with an audio file must not build/run any ASR source.
        with tempfile.TemporaryDirectory() as tmp:
            transcript = Path(tmp) / "rec.whisper.txt"
            transcript.write_text("the meeting is on tuesday")
            config = make_config(
                "--no-asr", "--input", "/nonexistent/rec.m4a", "-x", str(transcript),
                tmp_path=Path(tmp),
            )
            from speech_note import pipeline

            with mock.patch.object(pipeline, "run_final_asr") as ran_asr, \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                session = pipeline.run_file_pipeline(config)
            ran_asr.assert_not_called()  # ASR skipped despite the audio input
            self.assertIn("tuesday", session.raw_text())


class StatusDisplayTests(unittest.TestCase):
    """The single-owner status line: one phase, possibly many concurrent tasks."""

    def test_render_no_tasks_has_spinner_and_right_aligned_total(self) -> None:
        line = render_status_line(
            "Normalizing audio", [], phase_started=100.0, now=102.0, width=40, spinner="⠋"
        )
        self.assertTrue(line.startswith("⠋ Normalizing audio"))
        self.assertTrue(line.endswith("00:02"))
        self.assertEqual(len(line), 39)  # width-1, total hugging the right edge

    def test_render_concurrent_tasks_mixed_states(self) -> None:
        tasks = [
            StatusTask("whisper", started_at=100.0, done_at=105.0),
            StatusTask("parakeet", started_at=100.0),
            StatusTask("gemini", started_at=100.0, done_at=103.0, error=True),
        ]
        line = render_status_line(
            "ASR", tasks, phase_started=100.0, now=112.0, width=200, spinner="⠹"
        )
        self.assertIn("whisper ✓00:05", line)  # done -> ✓
        self.assertIn("parakeet 00:12", line)  # running -> no glyph
        self.assertIn("gemini ✗00:03", line)  # errored -> ✗
        self.assertTrue(line.startswith("⠹ ASR"))
        self.assertTrue(line.endswith("00:12"))

    def test_render_done_uses_resolved_glyph(self) -> None:
        line = render_status_line("ASR", [], phase_started=0.0, now=5.0, width=40, spinner="✓")
        self.assertTrue(line.startswith("✓ ASR"))

    def test_render_thin_terminal_keeps_left_and_total_no_wrap(self) -> None:
        tasks = [StatusTask(f"source{i}", started_at=0.0) for i in range(8)]
        line = render_status_line(
            "ASR", tasks, phase_started=0.0, now=1.0, width=30, spinner="⠋"
        )
        self.assertLessEqual(len(line), 29)  # never exceeds width-1, so no wrap
        self.assertTrue(line.startswith("⠋ ASR"))  # leftmost content survives
        self.assertTrue(line.endswith("00:01"))  # the time survives
        self.assertIn("…", line)  # the middle is elided

    def test_non_tty_prints_one_milestone_line_no_carriage_returns(self) -> None:
        buf = io.StringIO()  # isatty() -> False, so the non-TTY path is exercised
        with redirect_stderr(buf):
            display = StatusDisplay()
            display.begin_phase("ASR")
            display.start_task("whisper")
            display.finish_task("whisper")
            display.start_task("sherpa")
            display.finish_task("sherpa", error=True)
            display.end_phase()
        out = buf.getvalue()
        self.assertNotIn("\r", out)  # no spinner spam when stderr isn't a terminal
        self.assertIn("ASR", out)
        self.assertIn("whisper", out)
        self.assertIn("sherpa", out)
        self.assertIn("(failed)", out)  # the errored task is flagged in the summary


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

    def test_extra_transcript_short_flag_repeatable(self) -> None:
        args = parse_args(["-x", "a.txt", "-x", "b.txt"])
        self.assertEqual(args.extra_transcript, [Path("a.txt"), Path("b.txt")])


if __name__ == "__main__":
    unittest.main()


class UncertaintyAnnotationTests(unittest.TestCase):
    """The second, JSON-only pass that marks unsupported claims (see annotate.py)."""

    @staticmethod
    def _annotating_organizer() -> tuple[Organizer, mock.Mock]:
        client = mock.Mock()
        client.timeout = 12.0  # a real number: request_timeout_seconds does max() on it
        organizer = Organizer(
            mode="llama",
            client=cast(ChatClient, client),
            supervisor=None,
            context_tokens=65_536,
            max_output_tokens=16_384,
            annotate=True,
        )
        return organizer, client

    @staticmethod
    def _sources() -> "list[Transcript]":
        return [
            Transcript(label="asr1", model="a", kind="asr-final", text="she liked the letter"),
            Transcript(label="asr2", model="b", kind="asr-final", text="she liked the grid"),
        ]

    def _decode(self, content: str):
        from speech_note.annotate import decode_response

        return decode_response(content)

    # --- the response contract -------------------------------------------------

    def test_response_schema_bounds_the_reply(self) -> None:
        """The schema is a correctness measure, not decoration: it is what makes a
        truncated reply impossible, so every string and array must be bounded."""
        from speech_note import annotate
        from speech_note.config import (
            ANNOTATION_MAX_ALTERNATIVES,
            ANNOTATION_MAX_NOTES,
            ANNOTATION_MAX_QUOTE_CHARS,
        )

        item = annotate.RESPONSE_SCHEMA["properties"]["uncertain"]
        self.assertEqual(item["maxItems"], ANNOTATION_MAX_NOTES)
        fields = item["items"]["properties"]
        self.assertEqual(fields["quote"]["maxLength"], ANNOTATION_MAX_QUOTE_CHARS)
        self.assertEqual(fields["alternatives"]["maxItems"], ANNOTATION_MAX_ALTERNATIVES)
        self.assertEqual(fields["alternatives"]["items"]["maxLength"], ANNOTATION_MAX_QUOTE_CHARS)
        self.assertEqual(annotate.RESPONSE_FORMAT["type"], "json_schema")

    def test_annotation_request_sends_the_schema(self) -> None:
        organizer, client = self._annotating_organizer()
        from speech_note.annotate import RESPONSE_FORMAT

        client.chat.side_effect = [
            ChatResponse(content="She liked the letter.", served_model="m", finish_reason="stop"),
            ChatResponse(content='{"uncertain": []}', served_model="m", finish_reason="stop"),
        ]
        organizer.cleanup(self._sources())
        self.assertEqual(client.chat.call_count, 2)
        self.assertEqual(client.chat.call_args.kwargs["response_format"], RESPONSE_FORMAT)

    def test_annotation_request_carries_the_worked_example(self) -> None:
        """The synthetic example rides as a real user/assistant pair — an assistant turn
        anchors the output format far harder than prose in the system prompt (models
        shown an in-prompt example imitated its surface and dropped the envelope). The
        real data must be the LAST turn, after the example."""
        organizer, client = self._annotating_organizer()
        from speech_note.annotate import EXAMPLE_ASSISTANT, EXAMPLE_USER

        client.chat.side_effect = [
            ChatResponse(content="She liked the letter.", served_model="m", finish_reason="stop"),
            ChatResponse(content='{"uncertain": []}', served_model="m", finish_reason="stop"),
        ]
        organizer.cleanup(self._sources())
        messages = client.chat.call_args.args[0]
        self.assertEqual(
            [m["role"] for m in messages], ["system", "user", "assistant", "user"]
        )
        self.assertEqual(messages[1]["content"], EXAMPLE_USER)
        self.assertEqual(messages[2]["content"], EXAMPLE_ASSISTANT)
        json.loads(EXAMPLE_ASSISTANT)  # the demonstrated reply must itself be valid JSON
        self.assertIn("she liked the grid", messages[3]["content"])  # real data, last turn

    def test_parses_well_formed_response(self) -> None:
        notes, understood = self._decode(
            '{"uncertain": [{"quote": "a letter to a friend", '
            '"alternatives": ["a letter to the grid", "  she liked the letter  "]}]}'
        )
        self.assertTrue(understood)
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].quote, "a letter to a friend")
        self.assertEqual(notes[0].alternatives, ("a letter to the grid", "she liked the letter"))

    def test_tolerates_code_fence(self) -> None:
        notes, understood = self._decode(
            '```json\n{"uncertain": [{"quote": "x y", "alternatives": ["z"]}]}\n```'
        )
        self.assertTrue(understood)
        self.assertEqual(len(notes), 1)

    def test_empty_list_is_distinguishable_from_unusable_output(self) -> None:
        """"nothing to flag" and "model ignored the contract" must not look identical."""
        self.assertEqual(self._decode('{"uncertain": []}'), ([], True))
        for bad in ("", "not json at all", "{", '{"uncertain": "nope"}', '{"other": []}'):
            _notes, understood = self._decode(bad)
            self.assertFalse(understood, f"{bad!r} should not read as understood")

    def test_valid_prefix_survives_trailing_junk(self) -> None:
        """Providers without schema enforcement were observed closing the envelope after
        the first entry and continuing anyway. The valid prefix is a real answer; slicing
        to the LAST brace (the old decode) turned it into nothing."""
        notes, understood = self._decode(
            '{"uncertain": [{"quote": "a b", "alternatives": ["c d"]}]}, '
            '{"quote": "e f", "alternatives": ["g h"]}]}'
        )
        self.assertTrue(understood)
        self.assertEqual([n.quote for n in notes], ["a b"])

    def test_malformed_entries_are_skipped(self) -> None:
        for bad in ('{"uncertain": [{"quote": "x"}]}',
                    '{"uncertain": [{"alternatives": ["x"]}]}',
                    '{"uncertain": [{"quote": "  ", "alternatives": ["x"]}]}'):
            notes, understood = self._decode(bad)
            self.assertTrue(understood)   # the shape was right...
            self.assertEqual(notes, [])   # ...but no usable entry in it

    def test_alternatives_capped(self) -> None:
        from speech_note.config import ANNOTATION_MAX_ALTERNATIVES

        notes, _ = self._decode(
            '{"uncertain": [{"quote": "q", "alternatives": ["a","b","c","d","e"]}]}'
        )
        self.assertEqual(len(notes[0].alternatives), ANNOTATION_MAX_ALTERNATIVES)

    def test_alternatives_that_repeat_the_quote_are_dropped(self) -> None:
        """A marker offering the reader the same words twice is worse than no marker.

        Measured with the bundled local model: it returned notes whose alternative was the
        quote verbatim, and duplicate alternatives.
        """
        notes, understood = self._decode(
            '{"uncertain": [{"quote": "I did.", "alternatives": ["I did.", "i did"]}]}'
        )
        self.assertTrue(understood)
        self.assertEqual(notes, [], "a note with no genuinely different reading is useless")

        notes, _ = self._decode(
            '{"uncertain": [{"quote": "She liked it.", '
            '"alternatives": ["She liked it", "She kicked it.", "she  kicked  it"]}]}'
        )
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].alternatives, ("She kicked it.",))

    # --- applying the notes ----------------------------------------------------

    def test_applies_markers_without_altering_the_transcript(self) -> None:
        from speech_note.annotate import UncertaintyNote, apply_notes

        text = "I arrived. She liked the letter. It was late."
        result = apply_notes(text, [UncertaintyNote("She liked the letter.", ("She liked the grid.",))])
        self.assertEqual(result.inline_count, 1)
        self.assertEqual(result.appended_count, 0)
        self.assertIn("She liked the letter.", result.text)
        self.assertIn('[unclear audio; also heard as "She liked the grid."]', result.text)
        # Removing the marker restores the input exactly.
        self.assertEqual(re.sub(r" \[unclear audio;[^\]]*\]", "", result.text), text)

    def test_matches_across_whitespace_differences(self) -> None:
        from speech_note.annotate import UncertaintyNote, apply_notes

        result = apply_notes("One two\nthree four.", [UncertaintyNote("two three", ("two free",))])
        self.assertEqual(result.inline_count, 1)

    def test_unplaceable_quote_is_listed_not_dropped(self) -> None:
        """A quote we cannot locate is still the auditor's finding — report it.

        Attaching the marker to a guessed span would be worse than a separate list, but
        silently discarding the finding would be worse than either.
        """
        from speech_note.annotate import APPENDIX_HEADING, UncertaintyNote, apply_notes

        text = "Only this sentence exists."
        result = apply_notes(text, [UncertaintyNote("something never said", ("maybe this",))])
        self.assertEqual(result.inline_count, 0)
        self.assertEqual(result.appended_count, 1)
        self.assertTrue(result.text.startswith(text))
        self.assertIn(APPENDIX_HEADING, result.text)
        self.assertIn('"something never said"', result.text)
        self.assertIn('"maybe this"', result.text)

    def test_overlapping_notes_are_listed_rather_than_nested(self) -> None:
        from speech_note.annotate import UncertaintyNote, apply_notes

        result = apply_notes(
            "alpha beta gamma",
            [UncertaintyNote("alpha beta", ("x",)), UncertaintyNote("beta gamma", ("y",))],
        )
        self.assertEqual(result.inline_count, 1)
        self.assertEqual(result.appended_count, 1)
        self.assertEqual(result.text.count("[unclear audio;"), 1)

    def test_multiple_notes_keep_their_positions(self) -> None:
        from speech_note.annotate import UncertaintyNote, apply_notes

        result = apply_notes(
            "first claim here. second claim here.",
            [UncertaintyNote("first claim", ("a",)), UncertaintyNote("second claim", ("b",))],
        )
        self.assertEqual(result.inline_count, 2)
        self.assertLess(result.text.index('"a"'), result.text.index("second"))
        self.assertLess(result.text.index("second"), result.text.index('"b"'))

    # --- the organizer's use of it ---------------------------------------------

    def test_organizer_annotates_after_a_successful_cleanup(self) -> None:
        organizer, client = self._annotating_organizer()
        client.chat.side_effect = [
            ChatResponse(content="She liked the letter.", served_model="m", finish_reason="stop"),
            ChatResponse(
                content='{"uncertain": [{"quote": "She liked the letter.", '
                '"alternatives": ["She liked the grid."]}]}',
                served_model="m",
                finish_reason="stop",
            ),
        ]
        outcome = organizer.cleanup(self._sources())
        self.assertEqual(outcome.annotation_count, 1)
        self.assertEqual(outcome.annotation_appendix_count, 0)
        self.assertEqual(outcome.text_before_annotation, "She liked the letter.")
        self.assertIn("[unclear audio;", outcome.text)

    def test_annotation_failure_leaves_the_transcript_untouched(self) -> None:
        organizer, client = self._annotating_organizer()
        client.chat.side_effect = [
            ChatResponse(content="Clean text.", served_model="m", finish_reason="stop"),
            RuntimeError("annotation endpoint down"),
        ]
        outcome = organizer.cleanup(self._sources())
        self.assertEqual(outcome.text, "Clean text.")
        self.assertEqual(outcome.annotation_count, 0)
        assert outcome.annotation_error is not None
        self.assertIn("annotation endpoint down", outcome.annotation_error)
        self.assertIsNone(outcome.error)  # the run itself did not fail

    def test_truncated_reply_is_reported_not_patched_up(self) -> None:
        """The schema makes this unreachable; if it happens anyway, say so plainly."""
        organizer, client = self._annotating_organizer()
        client.chat.side_effect = [
            ChatResponse(content="Clean text.", served_model="m", finish_reason="stop"),
            ChatResponse(
                content='{"uncertain": [{"quote": "Clean text.", "alternat',
                served_model="m",
                finish_reason="length",
            ),
        ]
        outcome = organizer.cleanup(self._sources())
        self.assertEqual(outcome.text, "Clean text.")
        assert outcome.annotation_error is not None
        self.assertIn("output limit", outcome.annotation_error)

    def test_organizer_records_unusable_annotation_output(self) -> None:
        organizer, client = self._annotating_organizer()
        client.chat.side_effect = [
            ChatResponse(content="Clean text.", served_model="m", finish_reason="stop"),
            ChatResponse(content="I could not find any issues!", served_model="m", finish_reason="stop"),
        ]
        outcome = organizer.cleanup(self._sources())
        self.assertEqual(outcome.text, "Clean text.")
        assert outcome.annotation_error is not None
        self.assertIn("did not return the requested JSON", outcome.annotation_error)

    def test_well_formed_empty_result_is_not_an_error(self) -> None:
        organizer, client = self._annotating_organizer()
        client.chat.side_effect = [
            ChatResponse(content="Clean text.", served_model="m", finish_reason="stop"),
            ChatResponse(content='{"uncertain": []}', served_model="m", finish_reason="stop"),
        ]
        outcome = organizer.cleanup(self._sources())
        self.assertIsNone(outcome.annotation_error)
        self.assertEqual(outcome.annotation_count, 0)

    def test_no_annotation_pass_with_a_single_source(self) -> None:
        """With one source there is no disagreement to detect, so don't pay for a call."""
        organizer, client = self._annotating_organizer()
        client.chat.return_value = ChatResponse(
            content="Clean text.", served_model="m", finish_reason="stop"
        )
        organizer.cleanup(
            [Transcript(label="asr1", model="a", kind="asr-final", text="clean text")]
        )
        self.assertEqual(client.chat.call_count, 1)

    def test_annotation_default_follows_the_cleanup_provider(self) -> None:
        """On for remote models; off for the bundled local quant."""
        self.assertFalse(make_config().organizer.annotate)  # local default
        self.assertFalse(make_config("--offline").organizer.annotate)
        self.assertTrue(make_config("--online-paid").organizer.annotate)
        self.assertTrue(make_config("--online-free").organizer.annotate)

    def test_annotation_flag_overrides_the_provider_default(self) -> None:
        self.assertTrue(make_config("--annotate-uncertainty").organizer.annotate)
        self.assertFalse(
            make_config("--online-paid", "--no-annotate-uncertainty").organizer.annotate
        )
