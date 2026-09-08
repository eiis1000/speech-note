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


if __name__ == "__main__":
    unittest.main()
