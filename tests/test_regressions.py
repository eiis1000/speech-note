"""Regressions found by the full-code correctness pass.

Exercise public construction and pipeline paths with synthetic inputs; no live
credentials, recordings, downloads, or model processes are needed.
"""

from __future__ import annotations

import dataclasses
import io
import json
import queue
import sys
import tempfile
import threading
import types
import unittest
import wave
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from speech_note import annotate, audio, capture, pipeline
from speech_note.asr import build_transcriber
from speech_note.chat import ChatClient, ChatResponse
from speech_note.cli import parse_args, resolve_config, validate
from speech_note.config import parse_asr_source
from speech_note.model import CleanupOutcome, Transcript
from speech_note.organizer import build_organizer
from speech_note.session import ArtifactStore, Session
from speech_note.transcribers import CTCTranscriber
from tools import annotation_eval, asr_eval, cleanup_eval, make_eval_case


def config(*args: str):
    with mock.patch("speech_note.cli.defaults.load_user_asr_sources", return_value=()), \
         mock.patch("speech_note.cli.preferred_model_selected", return_value=False):
        return resolve_config(parse_args(["--organizer-mode", "heuristic", *args]))


class ConfigAndModelTests(unittest.TestCase):
    def test_explicit_provider_overrides_each_preset_in_either_order(self):
        for preset in ("-O", "-F", "-P"):
            for provider in ("local", "openrouter"):
                for args in ((preset, "--organizer-provider", provider),
                             ("--organizer-provider", provider, preset)):
                    with self.subTest(args=args):
                        self.assertEqual(config(*args).organizer.provider, provider)

    def test_invalid_numeric_values_are_rejected(self):
        for flag, value in (
            ("--sample-rate", "0"), ("--replay-speed", "nan"),
            ("--max-segment-seconds", "inf"), ("--asr-cpu-threads", "-1"),
            ("--organizer-timeout", "nan"), ("--start-padding-ms", "-1"),
            ("--organizer-context-tokens", "0"),
            ("--organizer-max-output-tokens", "0"),
        ):
            with self.subTest(flag=flag), self.assertRaisesRegex(SystemExit, flag):
                validate(config("-t", "hello", flag, value))

    def test_empty_model_and_source_lists_are_rejected(self):
        for flag in ("--asr", "--organizer-model"):
            with self.subTest(flag=flag), self.assertRaisesRegex(SystemExit, flag):
                validate(config("-t", "hello", flag, ", ,"))

    def test_unused_remote_asr_needs_no_credentials(self):
        for args in (("-t", "hello"), ("-x", "draft.txt"),
                     ("--no-asr", "-i", "a.wav", "-x", "draft.txt")):
            with self.subTest(args=args), mock.patch.dict("os.environ", {}, clear=True):
                validate(config("--asr", "openrouter-stt", *args))

    def test_faster_whisper_receives_download_permission(self):
        for allowed in (False, True):
            cfg = config("--auto-download" if allowed else "--no-auto-download")
            transcriber = build_transcriber(cfg, parse_asr_source("faster-whisper"))
            self.assertEqual(transcriber.auto_download, allowed)

    def test_cpu_ctc_does_not_implicitly_select_a_gpu(self):
        transcriber = build_transcriber(config(), parse_asr_source("ctc@cpu"))
        self.assertIsInstance(transcriber, CTCTranscriber)
        fake_transformers = types.SimpleNamespace(
            AutoModelForCTC=mock.Mock(), AutoProcessor=mock.Mock(), pipeline=mock.Mock(),
        )
        fake_transformers.AutoModelForCTC.from_pretrained.return_value.config.inputs_to_logits_ratio = 1280
        with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
            transcriber.ensure_loaded()
        self.assertEqual(fake_transformers.pipeline.call_args.kwargs["device"], -1)

    def _local_organizer(self, root, *, preferred, shortfall=False):
        default = root / "default.gguf"
        upgrade = root / "upgrade.gguf"
        if preferred:
            upgrade.write_bytes(b"model")
        else:
            default.write_bytes(b"model")
        with ExitStack() as stack:
            for name, value in (
                ("speech_note.organizer.DEFAULT_GGUF_MODEL", default),
                ("speech_note.llama_server.DEFAULT_GGUF_MODEL", default),
                ("speech_note.llama_server.PREFERRED_GGUF_MODEL", upgrade),
            ):
                stack.enter_context(mock.patch(name, value))
            stack.enter_context(mock.patch("speech_note.llama_server.shutil.which", return_value="/bin/llama-server"))
            stack.enter_context(mock.patch("speech_note.llama_server.model_ram_shortfall", return_value=1024 if shortfall else None))
            download = stack.enter_context(mock.patch("speech_note.organizer.ensure_default_cleanup_model"))
            stack.enter_context(redirect_stderr(io.StringIO()))
            organizer = build_organizer(config("--organizer-mode", "llama"))
            download.assert_not_called()
            return organizer, upgrade

    def test_organizer_uses_preferred_model_without_downloading_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            organizer, preferred = self._local_organizer(Path(tmp), preferred=True)
            self.addCleanup(organizer.close)
            self.assertIn(str(preferred), organizer.supervisor.launch_command)

    def test_organizer_honors_the_default_ram_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            organizer, _ = self._local_organizer(Path(tmp), preferred=False, shortfall=True)
            self.addCleanup(organizer.close)
            self.assertIsNone(organizer.supervisor.launch_command)

    def test_custom_endpoint_does_not_spawn_default_port_server(self):
        with mock.patch("speech_note.organizer.default_server_command") as launch:
            organizer = build_organizer(config(
                "--organizer-mode", "llama", "--organizer-api-base",
                "http://127.0.0.1:9876/v1/chat/completions",
            ))
        self.addCleanup(organizer.close)
        launch.assert_not_called()
        self.assertIsNone(organizer.supervisor.launch_command)


class AudioTailTests(unittest.TestCase):
    def test_short_audio_has_finite_telemetry_and_keeps_every_sample(self):
        for length in (1, 159, 160, 319, 320, 321):
            with self.subTest(length=length), tempfile.TemporaryDirectory() as tmp:
                source, target = Path(tmp) / "in.wav", Path(tmp) / "out.wav"
                audio.write_wav(source, np.full(length, 100, dtype="<i2").tobytes(), 16000)
                result = audio.normalize_pcm_wav(source, target)
                json.dumps(result.as_dict(), allow_nan=False)
                self.assertGreater(result.gain, 1)
                with wave.open(str(target)) as handle:
                    self.assertEqual(handle.getnframes(), length)

    def test_final_partial_block_cannot_escape_the_limiter(self):
        samples = np.concatenate([np.full(1600, 1000.0), [1e6]])
        limited, blocks = audio._limit(samples)
        ceiling = audio.INT16_FULL_SCALE * 10 ** (audio.NORMALIZE_PEAK_CEILING_DBFS / 20)
        self.assertEqual(len(limited), len(samples))
        self.assertLessEqual(np.max(np.abs(limited)), ceiling + 1e-6)
        self.assertEqual(limited[0], samples[0])
        self.assertGreater(blocks, 0)

    def test_level_measurements_use_the_same_time_grid_at_different_rates(self):
        # The same envelope sampled at three rates must produce the same gain.
        levels = np.concatenate([np.full(50, 1000.0), np.full(50, 100.0)])
        summaries = []
        with tempfile.TemporaryDirectory() as tmp:
            for rate in (8000, 16000, 48000):
                samples = np.repeat(levels, rate // 50).astype("<i2")
                source, target = Path(tmp) / "in.wav", Path(tmp) / "out.wav"
                audio.write_wav(source, samples.tobytes(), rate)
                result = audio.normalize_pcm_wav(source, target)
                summaries.append((result.gain, result.gain_db_min, result.gain_db_max))
            self.assertEqual(summaries[0], summaries[1])
            self.assertEqual(summaries[1], summaries[2])

    def test_sherpa_feeds_every_complete_window_and_pads_only_the_tail(self):
        for length in (1, 512, 513, 1024, 1100):
            with self.subTest(length=length):
                samples = np.ones(length, dtype=np.float32)
                vad = mock.Mock()
                vad.empty.return_value = True
                transcriber = build_transcriber(config(), parse_asr_source("sherpa"))
                transcriber.recognizer = mock.Mock()
                fake_sherpa = types.SimpleNamespace(VoiceActivityDetector=mock.Mock(return_value=vad))
                fake_librosa = types.SimpleNamespace(load=mock.Mock(return_value=(samples, 16000)))
                with mock.patch.dict(sys.modules, {"sherpa_onnx": fake_sherpa, "librosa": fake_librosa}):
                    transcriber.transcribe_file(Path("unused.wav"), "en")
                fed = np.concatenate([call.args[0] for call in vad.accept_waveform.call_args_list])
                np.testing.assert_array_equal(fed[:length], samples)
                self.assertTrue(np.all(fed[length:] == 0))
                self.assertEqual(len(fed), ((length + 511) // 512) * 512)


class OutputAndCaptureTests(unittest.TestCase):
    def test_parallel_same_stem_outputs_are_both_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            barrier = threading.Barrier(2)
            cfg = dataclasses.replace(config("-f", "-t", "hello"), output=root / "same-clean.txt")

            def write(text):
                session = Session(cfg)
                session.cleanup = CleanupOutcome(text=text)
                barrier.wait(timeout=2)
                pipeline.write_output_file(cfg, session)
                return session.paths["output"]

            with ThreadPoolExecutor(max_workers=2) as pool:
                paths = list(pool.map(write, ("first recording", "second recording")))
            self.assertEqual(len(set(paths)), 2)
            self.assertEqual({Path(p).read_text().strip() for p in paths},
                             {"first recording", "second recording"})

    def test_plain_latest_is_removed_when_the_next_run_has_no_annotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config("--artifacts-dir", tmp)
            store = ArtifactStore(cfg.artifacts_dir, cfg.archive_dir)
            first = Session(cfg)
            first.cleanup = CleanupOutcome(text="Old.[1]", text_before_annotation="Old.")
            store.commit(first)
            second = Session(cfg)
            second.cleanup = CleanupOutcome(text="New.")
            store.commit(second)
            self.assertFalse((Path(tmp) / "clean.plain.latest").exists())
            self.assertEqual(Path(first.paths["archive_clean_plain"]).read_text(), "Old.\n")

    def test_full_auto_empty_heuristic_cleanup_is_a_failure(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            out = Path(tmp) / "out.txt"
            session = pipeline.run_dry_text_pipeline(config("-f", "-t", "um uh", "-o", str(out)))
            self.assertTrue(session.run_failed)
            self.assertFalse(out.exists())
            self.assertTrue((Path(tmp) / "out-diagnostics.json").exists())

    def test_live_worker_waits_for_the_final_segment_after_capture_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = object.__new__(capture.CaptureRunner)
            runner.config = config()
            runner.session = Session(runner.config)
            runner.segment_queue = queue.Queue()
            runner.stop_event = threading.Event()
            runner.stop_event.set()
            runner.live_transcriber_ready = threading.Event()
            runner.live_transcriber_ready.set()
            runner.live_transcriber = mock.Mock()
            runner.live_transcriber.transcribe_file.return_value = "final utterance"
            runner.live_chunks = []
            runner._note = mock.Mock()
            worker = threading.Thread(target=runner._transcribe_worker, daemon=True)
            worker.start()
            try:
                worker.join(timeout=0.6)  # the old worker exits after a 0.5 s empty poll
                self.assertTrue(worker.is_alive())
                segment = Path(tmp) / "tail.wav"
                segment.write_bytes(b"audio")
                runner.segment_queue.put(segment)
            finally:
                runner.segment_queue.put(None)
                worker.join(timeout=2)
            self.assertEqual(runner.live_chunks, ["final utterance"])
            self.assertFalse(worker.is_alive())

    def test_capture_interruption_preserves_audio_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config("-f", "--input-device", "0", "-o", str(Path(tmp) / "note.txt"))
            pcm = np.full(1600, 1000, dtype="<i2").tobytes()

            def build_runner(_cfg, session):
                runner = mock.Mock()
                def interrupt():
                    session.observe_audio_frame(pcm)
                    raise KeyboardInterrupt
                runner.run.side_effect = interrupt
                return runner

            with mock.patch("speech_note.capture.CaptureRunner", side_effect=build_runner), \
                 redirect_stderr(io.StringIO()), self.assertRaises(KeyboardInterrupt):
                capture.run_capture_pipeline(cfg)
            recovered = Path(tmp) / "note-recording.wav"
            with wave.open(str(recovered)) as handle:
                self.assertEqual(handle.readframes(handle.getnframes()), pcm)
            payload = json.loads((Path(tmp) / "note-diagnostics.json").read_text())
            self.assertTrue(payload["errors"])
            self.assertEqual(payload["paths"]["recovery_recording"], str(recovered))


class ResponseIntegrityTests(unittest.TestCase):
    def test_truncated_audio_llm_response_is_an_asr_error(self):
        transcriber = build_transcriber(config(), parse_asr_source("openrouter"))
        response = ChatResponse("incomplete transcript", "model", "length")
        def encode(_source, target, **_kwargs):
            target.write_bytes(b"synthetic audio")
        with mock.patch("speech_note.audio.encode_to_mp3", side_effect=encode), \
             mock.patch("speech_note.chat.ChatClient.chat", return_value=response), \
             self.assertRaisesRegex(RuntimeError, "truncated"):
            transcriber.transcribe_file(Path("unused.wav"), "en")

    def test_malformed_or_empty_chat_responses_fall_through(self):
        for body in ([], None, {"choices": {}}, {"choices": [None]},
                     {"choices": [{"message": None}]},
                     {"choices": [{"message": {"content": " "}}]}):
            with self.subTest(body=body):
                client = ChatClient(api_base="http://unused/v1/chat/completions", models=["bad", "good"], timeout=1)
                bad = mock.Mock(ok=True, status_code=200)
                bad.json.return_value = body
                good = mock.Mock(ok=True, status_code=200)
                good.json.return_value = {"model": "good", "choices": [{"message": {"content": "Complete."}, "finish_reason": "stop"}]}
                with mock.patch.object(client, "candidate_models", return_value=["bad", "good"]), \
                     mock.patch("speech_note.chat.requests.post", side_effect=[bad, good]):
                    response = client.chat([], max_tokens=100, timeout=1)
                self.assertEqual(response.content, "Complete.")
                self.assertEqual(response.served_model, "good")

    def test_citation_display_cannot_reorder_or_multiply_source_words(self):
        source = Transcript("asr1", "m", "asr-final", "Alice called Bob")
        for display in ("Bob called Alice", "Alice called Bob Bob"):
            citation = annotate.Citation(display, 1, source.text)
            notes, rejected = annotate.verify_notes([annotate.UncertaintyNote("q", (citation,))], [source])
            self.assertEqual(rejected, 0)
            self.assertEqual(notes[0].alternatives[0].display(), source.text)

    def test_citation_verification_preserves_non_ascii_words(self):
        source = Transcript("asr1", "m", "asr-final", "the house 大房子")
        citation = annotate.Citation("", 1, "the house 小房子")
        notes, rejected = annotate.verify_notes([annotate.UncertaintyNote("q", (citation,))], [source])
        self.assertEqual((notes, rejected), ([], 1))


if __name__ == "__main__":
    unittest.main()
