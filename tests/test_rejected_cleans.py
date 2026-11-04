"""Rejected short outputs survive both fallback success and total failure."""
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from speech_note.cli import parse_args, resolve_config
from speech_note.model import CleanupOutcome
from speech_note.pipeline import commit_artifacts, report
from speech_note.session import Session


def candidate(text, model='gemini'):
    return {'request': {'model': model},
            'response': {'model': model, 'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}]},
            'error': 'incomplete cleanup: suspiciously short (2 words < 100 expected)'}


class RejectedCleanTests(unittest.TestCase):
    def run_case(self, root, *, text='Accepted transcript.', attempts=None):
        config = resolve_config(parse_args(['-f', '-t', 'source', '--organizer-mode', 'heuristic',
                                             '-o', str(root / 'note-clean.txt')]))
        session = Session(config)
        session.cleanup = CleanupOutcome(text=text, error=None if text else 'all attempts rejected',
                                         attempts=attempts or [])
        session.run_failed = not bool(text)
        with patch.dict(os.environ, {'XDG_STATE_HOME': str(root / 'state')}):
            commit_artifacts(config, session)
        return config, session

    def test_rejected_candidate_saved_verbatim_on_fallback_success(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = '  Short result.\n\n'
            config, session = self.run_case(root, attempts=[candidate(raw)])
            path = root / 'note-too-short-clean.txt'
            self.assertEqual(path.read_text(), raw)
            self.assertEqual((root / 'note-clean.txt').read_text(), 'Accepted transcript.\n')
            data = json.loads(Path(session.paths['persistent_diagnostics']).read_text())
            self.assertEqual(data['facts']['too_short_cleans'][0]['model'], 'gemini')
            self.assertEqual(data['facts']['too_short_cleans'][0]['path'], str(path))
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                report(config, session)
            self.assertIn('saved rejected too-short clean:', stderr.getvalue())

    def test_all_rejections_saved_without_overwriting_existing_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old = root / 'note-too-short-clean.txt'
            old.write_text('Earlier run')
            _, session = self.run_case(root, text='', attempts=[candidate('First.'), candidate('Second.', 'other')])
            self.assertEqual(old.read_text(), 'Earlier run')
            self.assertEqual((root / 'note-too-short-clean-2.txt').read_text(), 'First.')
            self.assertEqual((root / 'note-too-short-clean-3.txt').read_text(), 'Second.')
            self.assertFalse((root / 'note-clean.txt').exists())
            self.assertTrue(session.run_failed)

    def test_successful_or_other_failed_attempts_are_not_mislabeled_short(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            success = candidate('Complete.')
            success.pop('error')
            failure = {'error': 'HTTP 429: rate limited', 'response': {'error': 'rate limited'}}
            self.run_case(root, attempts=[success, failure])
            self.assertFalse(list(root.glob('*too-short*')))

    def test_candidate_write_failure_is_not_silent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch('speech_note.pipeline.write_unique_output', side_effect=OSError('disk full')):
                _, session = self.run_case(root, attempts=[candidate('Short.')])
            self.assertTrue(session.run_failed)
            self.assertTrue(any('too-short cleanup' in error for error in session.errors))
            self.assertTrue((root / 'note-clean-diagnostics.json').exists())
