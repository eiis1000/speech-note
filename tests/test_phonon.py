"""Completeness, consent and isolation contracts for the optional backend."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import wave
from unittest.mock import patch

from speech_note.config import DEFAULT_ASR_SOURCES, parse_asr_source
from speech_note.install import download_prompts, ModelInstallDeclined
from speech_note.phonon import PhononTranscriber, validate_result


class PhononTests(unittest.TestCase):
    def result(self):
        return {"text": "first last", "audio_seconds": 60.0, "truncated": False,
                "segments": [{"start": 0, "end": 30, "text": "first"},
                             {"start": 30, "end": 60, "text": "last"}]}

    def test_complete_and_silent(self):
        self.assertEqual(validate_result(self.result(), 60), "first last")
        self.assertEqual(validate_result({"text": "", "audio_seconds": 60,
                                          "truncated": False, "segments": []}, 60), "")

    def test_reject_incomplete_results(self):
        cases = []
        for field, value in [("truncated", True), ("truncated", None),
                             ("audio_seconds", 30), ("audio_seconds", float("nan")),
                             ("segments", []), ("text", "first")]:
            row = self.result()
            row[field] = value
            cases.append(row)
        for field, value in [("start", 31), ("start", 29), ("end", 59),
                             ("end", float("inf")), ("text", None)]:
            row = copy.deepcopy(self.result())
            row["segments"][1][field] = value
            cases.append(row)
        for row in cases:
            with self.subTest(row=row), self.assertRaises(ValueError):
                validate_result(row, 60)

    def test_registry_and_defaults(self):
        self.assertEqual(parse_asr_source("phonon").device_kind, "cpu")
        for spec in ["phonon@gpu", "phonon:other"]:
            with self.assertRaises(ValueError):
                parse_asr_source(spec)
        self.assertNotIn("phonon", [s.backend for s in DEFAULT_ASR_SOURCES])

    def test_no_download_without_consent(self):
        transcriber = PhononTranscriber(cpu_threads=12, download_root=None)
        with patch.object(transcriber, "_invoke", return_value={"ready": False}) as invoke:
            with download_prompts(False), self.assertRaises(ModelInstallDeclined):
                transcriber.ensure_downloaded()
            invoke.assert_called_once_with("check")

    def test_explicit_download_and_cached_preparation(self):
        transcriber = PhononTranscriber(cpu_threads=12, download_root=None, auto_download=True)
        with patch.object(transcriber, "_invoke", side_effect=[{"ready": False}, {"ready": True}]) as invoke:
            with download_prompts(False):
                transcriber.ensure_downloaded()
                transcriber.ensure_downloaded()
            self.assertEqual([c.args[0] for c in invoke.call_args_list], ["check", "download"])

    def test_child_failure_and_bad_json(self):
        transcriber = PhononTranscriber(cpu_threads=12, download_root=Path("/tmp/models"))
        for code, output in [(1, json.dumps(self.result())), (0, "invalid"), (0, "[]")]:
            with patch("speech_note.phonon.shutil.which", return_value="worker"), patch(
                "speech_note.phonon.subprocess.run",
                return_value=subprocess.CompletedProcess([], code, output, "failed"),
            ) as run:
                with self.assertRaises(RuntimeError):
                    transcriber._invoke("transcribe", Path("audio.wav"))
                self.assertEqual(run.call_args.kwargs["env"]["FERMION_CPU_THREADS"], "12")
                self.assertEqual(run.call_args.kwargs["env"]["FERMION_CACHE_DIR"], "/tmp/models/phonon")

    def test_duration_uses_decoded_pcm_not_container_padding(self):
        transcriber = PhononTranscriber(cpu_threads=12, download_root=None)
        transcriber._prepared = True
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "normalized.wav"
            with wave.open(str(path), "wb") as audio:
                audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                audio.writeframes(b"\0\0" * 16000 * 60)
            with patch.object(transcriber, "_invoke", return_value=self.result()):
                self.assertEqual(transcriber.transcribe_file(path, "en", duration_seconds=60.04),
                                 "first last")
            with patch.object(transcriber, "_invoke", return_value=self.result()):
                with self.assertRaisesRegex(ValueError, "English"):
                    transcriber.transcribe_file(path, "fr")


if __name__ == "__main__":
    unittest.main()
