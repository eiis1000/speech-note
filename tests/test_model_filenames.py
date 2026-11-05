"""Auto output names reflect actual producers, not the configured fallback order."""
import tempfile
import unittest
from pathlib import Path

from speech_note.cli import parse_args, resolve_config
from speech_note.model import CleanupOutcome, Transcript
from speech_note.naming import model_filename_tag, produced_transcript_path
from speech_note.pipeline import write_output_file
from speech_note.session import Session


class ModelFilenameTests(unittest.TestCase):
    def session(self, root):
        config = resolve_config(parse_args(['-f', '-i', str(root / 'meeting.wav'),
                                            '--organizer-mode', 'heuristic']))
        # The normal pipeline places auto output in cwd; keep tests isolated.
        import dataclasses
        config = dataclasses.replace(config, output=root / 'meeting-clean.txt')
        session = Session(config)
        session.add_transcript(Transcript('asr1', 'openai/whisper-large-v3-turbo', 'asr-final', 'One.'))
        session.add_transcript(Transcript('asr3', 'microsoft/mai-transcribe-1.5', 'asr-final', 'Two.'))
        session.cleanup = CleanupOutcome(text='Final.', method='llama',
                                         served_model='nvidia/nemotron-3-ultra-550b-a55b:free')
        return config, session

    def test_actual_asr_and_fallback_models_in_saved_name(self):
        with tempfile.TemporaryDirectory() as temp:
            config, session = self.session(Path(temp))
            write_output_file(config, session)
            path = Path(session.paths['output'])
            self.assertEqual(path.name, 'meeting-whisper-large-v3-turbo-mai-transcribe-1-5-nemotron-3-ultra-550b-a55b-clean.txt')
            self.assertEqual(path.read_text(), 'Final.\n')
            write_output_file(config, session)
            self.assertNotEqual(session.paths['output'], str(path))
            self.assertTrue(path.exists())

    def test_rejected_name_uses_its_own_model_not_accepted_model(self):
        with tempfile.TemporaryDirectory() as temp:
            config, session = self.session(Path(temp))
            path = produced_transcript_path(config, session, model='google/gemini-3-flash-preview', rejected=True)
            self.assertIn('gemini-3-flash-preview-too-short-clean', path.name)
            self.assertNotIn('nemotron', path.name)

    def test_explicit_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            config = resolve_config(parse_args(['-f', '-t', 'Text.', '-o', str(Path(temp) / 'chosen.txt'),
                                                '--organizer-mode', 'heuristic']))
            session = Session(config)
            session.cleanup = CleanupOutcome(text='Final.', served_model='model')
            write_output_file(config, session)
            self.assertEqual(Path(session.paths['output']).name, 'chosen.txt')

    def test_long_input_names_leave_room_for_sidecars(self):
        with tempfile.TemporaryDirectory() as temp:
            config, session = self.session(Path(temp))
            import dataclasses
            config = dataclasses.replace(config, input_file=Path(temp) / ('long-' * 60 + '.wav'))
            path = produced_transcript_path(config, session)
            self.assertLess(len(path.stem + '-diagnostics-999.json'), 255)
            self.assertIn('nemotron-3-ultra-550b-a55b-clean', path.name)

    def test_tags_keep_versions(self):
        self.assertEqual(model_filename_tag('microsoft/mai-transcribe-1.5'), 'mai-transcribe-1-5')
        self.assertEqual(model_filename_tag('ggml-medium-q8_0.bin'), 'whisper-medium-q8-0')
