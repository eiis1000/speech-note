"""Cross-layer failure contracts from the second full review; synthetic inputs only."""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from speech_note import annotate, pipeline
from speech_note.asr import build_transcriber, run_source
from speech_note.chat import ChatClient
from speech_note.cli import parse_args, resolve_config
from speech_note.config import parse_asr_source
from tools import annotation_eval, asr_eval, cleanup_eval


def config(*args):
    with mock.patch('speech_note.cli.defaults.load_user_asr_sources', return_value=()), \
         mock.patch('speech_note.cli.preferred_model_selected', return_value=False):
        return resolve_config(parse_args(['--organizer-mode', 'heuristic', *args]))


def response(payload):
    reply = mock.Mock(ok=True, status_code=200, text=json.dumps(payload))
    reply.json.return_value = payload
    return reply


def chat_body(content='Complete transcript.', **extra):
    return {'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}], **extra}


def client():
    transport = ChatClient(api_base='http://local/v1/chat/completions', models=['bad', 'good'], timeout=1)
    transport._catalog_checked = True
    return transport


class ProviderContracts(unittest.TestCase):
    def test_malformed_content_blocks_advance_fallback(self):
        for text in (None, {'value': 'invented'}, 123):
            transport = client()
            bad = chat_body([{'type': 'text', 'text': text}])
            with self.subTest(text=text), mock.patch('requests.post', side_effect=[response(bad), response(chat_body())]):
                result = transport.chat([], max_tokens=100, timeout=1)
                self.assertEqual(result.content, 'Complete transcript.')
                self.assertEqual(len(transport.last_attempts), 2)

    def test_stt_error_text_never_becomes_a_source(self):
        cfg = config('--asr', 'openrouter-stt')
        source = parse_asr_source('openrouter-stt')
        for payload in ({'text': 'Partial.', 'error': {'message': 'upstream stopped'}}, [], None):
            transcriber = build_transcriber(cfg, source)
            with self.subTest(payload=payload), \
                 mock.patch.dict('os.environ', {'OPENROUTER_API_KEY': 'synthetic'}), \
                 mock.patch('speech_note.audio.encode_to_mp3', side_effect=lambda _p, target, **_k: target.write_bytes(b'audio')), \
                 mock.patch('requests.post', return_value=response(payload)):
                outcome = run_source(cfg, source, Path('unused.wav'), label='asr1', duration=1, transcriber=transcriber)
            self.assertIsNone(outcome.transcript)
            self.assertIsNotNone(outcome.error)
            self.assertIsNone(outcome.skip_reason)

    def test_evals_reject_the_same_embedded_errors(self):
        payload = chat_body('{"uncertain": []}', error={'message': 'upstream stopped'})
        with tempfile.TemporaryDirectory() as tmp, mock.patch('requests.post', return_value=response(payload)):
            root = Path(tmp)
            clip = root / 'clip.mp3'
            clip.write_bytes(b'audio')
            text, error, _ = cleanup_eval.ask('key', 'm', [], 'http://local', root / 'out.txt')
            self.assertFalse(text)
            self.assertIn('upstream stopped', error)
            notes, error, _ = annotation_eval.ask('key', 'm', [], {}, root / 'out.json', 'http://local')
            self.assertFalse(notes)
            self.assertIn('upstream stopped', error)
            text, error = asr_eval.transcribe_audio_llm('key', 'm', clip)
            self.assertFalse(text)
            self.assertIn('upstream stopped', error)
        with tempfile.TemporaryDirectory() as tmp, mock.patch('requests.post', return_value=response({'text': 'Partial.', 'error': 'stopped'})):
            clip = Path(tmp) / 'clip.mp3'
            clip.write_bytes(b'audio')
            text, error = asr_eval.transcribe_stt('key', 'm', clip)
            self.assertFalse(text)
            self.assertIn('stopped', error)


if __name__ == '__main__':
    unittest.main()
