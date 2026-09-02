"""Full-stack, no-mock end-to-end tests.

These run the *real* pipeline: ffmpeg decode/normalize, the real Whisper
primary pass, the real secondary ASR backend, and (for llama mode) a real local
llama-server cleanup. Nothing is mocked. They are gated behind SPEECH_NOTE_E2E=1
because they load multi-hundred-MB / multi-GB models and take tens of seconds to
minutes; the fast logic suite lives in test_speech_note.py.

Run them with:

    SPEECH_NOTE_E2E=1 nix develop -c python -m unittest -v test_e2e

A test self-skips if a binary or model it needs is absent, so a partial install
still runs whatever it can.

These tests need two recordings you supply (not committed — provide your own
speech): tests/fixtures/short.wav (~15 s) and tests/fixtures/long.wav (~50 s).
Any clear English speech works; the long clip is looped internally to exercise
long-form transcription past the model's ~400 s position limit. Tests self-skip
when a fixture is absent.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import tempfile
import sys
import unittest
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

# Direct execution (python tests/test_e2e.py) puts tests/ on sys.path, not
# the repo root, so make the package importable either way — the __main__ guard at
# the bottom is otherwise decorative.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from speech_note.cli import parse_args, resolve_config
from speech_note.config import DEFAULT_GGUF_MODEL, WHISPER_CPP_MODEL_DIR
from speech_note.pipeline import run_archive_pipeline
from speech_note.transcribers import default_whisper_cpp_model_path

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "tests" / "fixtures"
FIXTURE_WAV = FIXTURES / "short.wav"
# A second, deliberately imperfect transcript to exercise the multi-source
# cleanup path with an external transcript (not a mock).
EXTRA_TRANSCRIPT = (
    "this is a rough draft transcript of the same audio "
    "with no punctuation and a few likely errors"
)

E2E_ENABLED = os.environ.get("SPEECH_NOTE_E2E") == "1"


def _have(binary: str) -> bool:
    return shutil.which(binary) is not None


def _build_archive(zip_path: Path, *, stem: str = "note", with_transcript: bool = True) -> None:
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(FIXTURE_WAV, arcname=f"recordings/{stem}.wav")
        if with_transcript:
            archive.writestr(f"transcripts/{stem}.google.txt", EXTRA_TRANSCRIPT)


@unittest.skipUnless(E2E_ENABLED, "set SPEECH_NOTE_E2E=1 to run the full-stack tests")
class ArchiveFullStackTests(unittest.TestCase):
    """One real archive, run through several modes with zero mocking."""

    @classmethod
    def setUpClass(cls) -> None:
        if not FIXTURE_WAV.exists():
            raise unittest.SkipTest(f"fixture recording missing: {FIXTURE_WAV}")
        if not _have("whisper-cli"):
            raise unittest.SkipTest("whisper-cli not on PATH (enter the Nix shell)")
        if not _have("ffmpeg") or not _have("ffprobe"):
            raise unittest.SkipTest("ffmpeg/ffprobe not on PATH")
        if not default_whisper_cpp_model_path("medium-q8_0").exists():
            raise unittest.SkipTest(
                f"primary model not installed under {WHISPER_CPP_MODEL_DIR}"
            )

    def _run(self, tmp: Path, zip_path: Path, *argv: str):
        args = [
            "--input", str(zip_path),
            "--artifacts-dir", str(tmp),
            *argv,
        ]
        config = resolve_config(parse_args(args))
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return run_archive_pipeline(config)

    def _assert_both_asr_sources_ran(self, session) -> None:
        # The default collection is two sources (whisper-cpp + sherpa); both should
        # produce a transcript, with no primary/secondary roles.
        transcripts = session.asr_transcripts()
        self.assertEqual(len(transcripts), 2, "expected two ASR transcripts")
        # The fixture says "testing ... live transcription"; the first source
        # (whisper) should hear it.
        self.assertIn("transcription", transcripts[0].text.lower())
        self.assertTrue(transcripts[1].text.strip())
        outcomes = [o for o in session.asr_outcomes if o.name.startswith("asr")]
        self.assertEqual(len(outcomes), 2)
        for outcome in outcomes:
            self.assertTrue(outcome.ok)
            self.assertIsNotNone(outcome.seconds)

    def test_off_mode_real_asr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            zip_path = tmp / "lecture.zip"
            _build_archive(zip_path)
            session = self._run(tmp, zip_path, "--organizer-mode", "off")
            self.assertFalse(session.run_failed)
            self._assert_both_asr_sources_ran(session)
            # "off" => clean output is the primary transcript verbatim.
            assert session.cleanup is not None
            self.assertEqual(session.cleanup.text, session.asr_transcripts()[0].text)
            self.assertIn("transcription", (tmp / "clean.latest").read_text().lower())
            self.assertEqual(json.loads((tmp / "diagnostics.latest.json").read_text())["schema"], 6)

    def test_heuristic_mode_real_asr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            zip_path = tmp / "lecture.zip"
            _build_archive(zip_path)
            session = self._run(tmp, zip_path, "--organizer-mode", "heuristic")
            self.assertFalse(session.run_failed)
            self._assert_both_asr_sources_ran(session)
            assert session.cleanup is not None
            self.assertEqual(session.cleanup.method, "heuristic")
            self.assertTrue(session.cleanup.text.strip())

    def test_extra_transcript_from_archive_reaches_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            zip_path = tmp / "lecture.zip"
            _build_archive(zip_path)
            session = self._run(tmp, zip_path, "--organizer-mode", "off")
            labels = {t.label for t in session.transcripts}
            self.assertIn("extra:note.google.txt", labels)

    @unittest.skipUnless(
        DEFAULT_GGUF_MODEL.exists() and _have("llama-server"),
        "local cleanup model / llama-server not available",
    )
    def test_llama_cleanup_real_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            zip_path = tmp / "lecture.zip"
            _build_archive(zip_path)
            session = self._run(tmp, zip_path, "--organizer-mode", "llama")
            self.assertFalse(session.run_failed)
            self._assert_both_asr_sources_ran(session)
            cleanup = session.cleanup
            assert cleanup is not None
            self.assertEqual(cleanup.method, "llama")
            self.assertIsNone(cleanup.error)
            self.assertTrue(cleanup.served_model)
            # The model actually rewrote the transcript into clean prose.
            self.assertTrue(cleanup.text.strip())
            self.assertIn("transcription", cleanup.text.lower())
            self.assertEqual((tmp / "clean.latest").read_text().strip(), cleanup.text)

    def test_full_auto_names_output_from_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            work = tmp / "cwd"
            work.mkdir()
            zip_path = tmp / "cosmicwatch.zip"
            _build_archive(zip_path, with_transcript=False)
            prior = os.getcwd()
            os.chdir(work)
            try:
                session = self._run(
                    tmp / "artifacts", zip_path, "--full-auto", "--organizer-mode", "heuristic"
                )
            finally:
                os.chdir(prior)
            self.assertFalse(session.run_failed)
            self.assertEqual(
                [p.name for p in work.iterdir()],
                ["cosmicwatch-whisper-parakeet-clean.txt"],
            )


CTC_MODEL_CACHE = (
    Path.home() / ".cache" / "huggingface" / "hub" / "models--nvidia--parakeet-ctc-0.6b"
)


SHERPA_MODEL_CACHE = (
    Path.home() / ".cache" / "huggingface" / "hub"
    / "models--csukuangfj--sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
)


# A user-supplied ~50s recording; looped it exceeds Parakeet-CTC's ~400s position cliff.
LONG_SOURCE = FIXTURES / "long.wav"


@unittest.skipUnless(E2E_ENABLED, "set SPEECH_NOTE_E2E=1 to run the full-stack tests")
@unittest.skipUnless(SHERPA_MODEL_CACHE.exists(), "sherpa parakeet-tdt bundle not in HF cache")
class SecondarySherpaFullStackTests(unittest.TestCase):
    """The default secondary backend on real audio: sherpa-onnx VAD-segments and
    decodes in C++ on the CPU. Proves it is the resolved default, covers a clip well
    past the single-pass attention OOM cliff, and runs nothing mocked."""

    @classmethod
    def setUpClass(cls) -> None:
        if not LONG_SOURCE.exists():
            raise unittest.SkipTest(f"source recording missing: {LONG_SOURCE}")
        if not _have("ffmpeg"):
            raise unittest.SkipTest("ffmpeg not on PATH")

    def test_sherpa_is_in_the_default_collection_on_cpu(self) -> None:
        config = resolve_config(parse_args(["--organizer-mode", "off"]))
        sherpa = next(s for s in config.asr_sources if s.backend == "sherpa")
        self.assertEqual(sherpa.device, "cpu")

    def test_transcribes_long_audio_with_full_coverage(self) -> None:
        from speech_note.asr import run_source
        from speech_note.config import parse_asr_source

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            clip = tmp / "long.wav"
            # ~455s: well past the ~400s single-pass attention cliff. VAD keeps each
            # decoded segment short, so there is no OOM and no length limit.
            subprocess.run(
                ["ffmpeg", "-nostdin", "-y", "-stream_loop", "8", "-i", str(LONG_SOURCE),
                 "-ac", "1", "-ar", "16000", str(clip)],
                check=True, capture_output=True,
            )
            duration = float(
                subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=nk=1:nw=1", str(clip)],
                    check=True, capture_output=True, text=True,
                ).stdout.strip()
            )
            self.assertGreater(duration, 400.0)
            config = resolve_config(parse_args(["--organizer-mode", "off"]))
            outcome = run_source(
                config, parse_asr_source("sherpa"), clip,
                label="asr1", duration=duration, transcriber=None,
            )
            self.assertTrue(
                outcome.ok,
                msg=f"sherpa failed on {duration:.0f}s: {outcome.error or outcome.skip_reason}",
            )
            # ~50s source -> ~54 words; looped 9x is ~480. Whole-clip coverage, not
            # just the first segment.
            assert outcome.transcript is not None
            words = len(outcome.transcript.text.split())
            self.assertGreaterEqual(words, 400, msg=f"under-covered: only {words} words for {duration:.0f}s")


@unittest.skipUnless(E2E_ENABLED, "set SPEECH_NOTE_E2E=1 to run the full-stack tests")
@unittest.skipUnless(CTC_MODEL_CACHE.exists(), "nvidia/parakeet-ctc-0.6b not in HF cache")
class SecondaryCtcLongFormFullStackTests(unittest.TestCase):
    """Prove the CTC backend transcribes audio with no length limit. A single
    forward pass over >400s raises (5000-position embedding cliff); the default
    config transcribes it via the HF pipeline's long-form striding. Real model,
    real ffmpeg, nothing mocked."""

    @classmethod
    def setUpClass(cls) -> None:
        if not LONG_SOURCE.exists():
            raise unittest.SkipTest(f"source recording missing: {LONG_SOURCE}")
        if not _have("ffmpeg"):
            raise unittest.SkipTest("ffmpeg not on PATH")

    def _make_long_clip(self, tmp: Path, loops: int) -> Path:
        out = tmp / "long.wav"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-stream_loop", str(loops), "-i", str(LONG_SOURCE),
             "-ac", "1", "-ar", "16000", str(out)],
            check=True, capture_output=True,
        )
        return out

    def test_ctc_chunk_window_under_position_cliff(self) -> None:
        from speech_note.asr import ctc_chunk_config

        chunk_length, stride = ctc_chunk_config()
        self.assertEqual(chunk_length, 240.0)
        self.assertLess(stride, chunk_length)

    def test_transcribes_audio_past_the_position_cliff(self) -> None:
        from speech_note.asr import run_source
        from speech_note.config import parse_asr_source

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            clip = self._make_long_clip(tmp, loops=8)  # ~455s, over the ~400s cliff
            duration = float(
                subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=nk=1:nw=1", str(clip)],
                    check=True, capture_output=True, text=True,
                ).stdout.strip()
            )
            self.assertGreater(duration, 400.0, "clip must exceed the position cliff")
            config = resolve_config(parse_args(["--asr", "ctc@cpu", "--organizer-mode", "off"]))
            outcome = run_source(
                config, parse_asr_source("ctc@cpu"), clip,
                label="asr1", duration=duration, transcriber=None,
            )
            self.assertTrue(
                outcome.ok, msg=f"CTC failed on {duration:.0f}s: {outcome.error or outcome.skip_reason}"
            )
            # Coverage check: long-form striding must cover the WHOLE clip, not
            # just the first window. The source ~50s clip transcribes to ~54
            # words; looped 9x that is ~480. Without the inputs_to_logits_ratio
            # fix the pipeline keeps only the first window (~250 words).
            assert outcome.transcript is not None
            words = len(outcome.transcript.text.split())
            self.assertGreaterEqual(words, 400, msg=f"under-covered: only {words} words for {duration:.0f}s")


if __name__ == "__main__":
    unittest.main()
