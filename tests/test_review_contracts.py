"""Cross-layer failure contracts from the second full review; synthetic inputs only."""
import io
import json
import tempfile
import unittest
import requests
import dataclasses
import threading
import zipfile
import queue
import wave
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from speech_note import annotate, pipeline
from speech_note.asr import build_transcriber, run_source
from speech_note.chat import ChatClient
from speech_note.cli import parse_args, resolve_config, validate
from speech_note.config import parse_asr_source
from speech_note.model import Transcript, CleanupOutcome
from speech_note.organizer import Organizer, cleanup_request_plan
from speech_note.session import Session, ArtifactStore
from speech_note.capture import CaptureRunner
from speech_note.install import consent_to_download, download_prompts
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


class CleanupAndAuditContracts(unittest.TestCase):
    def test_malformed_audit_retries_without_applying_a_prefix(self):
        for malformed in ('{"uncertain": []} {"quote": "lost"}', '{"uncertain": [{"quote": "lost"}]}', 'not JSON'):
            transport = client()
            with self.subTest(malformed=malformed), mock.patch('requests.post', side_effect=[response(chat_body(malformed)), response(chat_body('{"uncertain": []}'))]):
                result = annotate.annotate(transport, cleaned_text='Complete.', sources=[], context_tokens=16384)
            self.assertIsNone(result.error)
            self.assertEqual(result.text, 'Complete.')
            self.assertEqual(len(transport.last_attempts), 2)
            self.assertIn('error', transport.last_attempts[0])

    def test_word_fragments_do_not_anchor_inside_other_words(self):
        note = annotate.UncertaintyNote('he', (annotate.Citation('she', 0, ''),))
        result = annotate.apply_notes('The weather improved.', [note])
        self.assertEqual(result.inline_count, 0)
        self.assertEqual(result.appended_count, 1)
        result = annotate.apply_notes('The weather improved; he left.', [note])
        self.assertIn('he[1] left', result.text)

    def test_exact_output_budget_is_usable_through_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config('-f', '-t', 'Complete transcript.', '-o', str(Path(tmp) / 'out.txt'),
                         '--organizer-mode', 'llama', '--organizer-max-output-tokens', '4096',
                         '--organizer-api-base', 'http://local/v1/chat/completions')
            with mock.patch('requests.get', side_effect=requests.ConnectionError('no catalog')), \
                 mock.patch('requests.post', return_value=response(chat_body())) as post, \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = pipeline.run_dry_text_pipeline(cfg)
            self.assertFalse(result.run_failed)
            self.assertEqual(post.call_count, 1)

    def test_skipped_cleanup_does_not_reuse_previous_attempt_history(self):
        transport = client()
        organizer = Organizer(mode='llama', client=transport, supervisor=None, context_tokens=16384, max_output_tokens=4096)
        with mock.patch('requests.post', return_value=response(chat_body())):
            first = organizer.cleanup([Transcript('s', 'm', 'user', 'Complete transcript.')])
            second = organizer.cleanup([])
        self.assertEqual(len(first.attempts), 1)
        self.assertEqual(second.attempts, [])


class ArtifactContracts(unittest.TestCase):
    def test_output_failure_keeps_computed_text_and_nonzero_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            blocked = root / 'blocked'
            blocked.mkdir()
            cfg = config('-f', '-t', 'Complete transcript.', '-o', str(blocked))
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = pipeline.run_dry_text_pipeline(cfg)
            self.assertTrue(result.run_failed)
            evidence = json.loads(Path(result.paths['diagnostics']).read_text())
            self.assertEqual(evidence['cleanup']['text'], 'Complete transcript.')
            self.assertTrue(evidence['run_failed'])
            self.assertTrue(evidence['errors'])

    def test_repeated_exports_do_not_mix_sources_or_keep_stale_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            export = root / 'sources'
            cfg = config('-t', 'New.', '--export-sources', str(export), '--artifacts-dir', str(root / 'artifacts'))
            first = Session(cfg)
            first.add_transcript(Transcript('s', 'old', 'external', 'Old source.'))
            first.cleanup = CleanupOutcome(text='Old clean.')
            pipeline.write_sources_export(cfg, first)
            second = Session(cfg)
            second.add_transcript(Transcript('s', 'new', 'external', 'New source.'))
            pipeline.write_sources_export(cfg, second)
            new_export = Path(second.paths['exported_sources'])
            self.assertNotEqual(new_export, export)
            self.assertEqual((export / 'clean.txt').read_text(), 'Old clean.\n')
            self.assertFalse((new_export / 'clean.txt').exists())
            self.assertEqual([p.name for p in new_export.iterdir()], ['01-new.txt'])

    def test_archive_diagnostics_keep_original_zip_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / 'input.zip'
            with zipfile.ZipFile(archive, 'w') as handle:
                handle.writestr('audio.wav', b'not decoded in no-ASR mode')
                handle.writestr('source.txt', 'Complete transcript.')
            cfg = config('-f', '--no-asr', '-i', str(archive), '-o', str(root / 'clean.txt'))
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = pipeline.run_archive_pipeline(cfg)
            evidence = json.loads(Path(result.paths['diagnostics']).read_text())
            self.assertEqual(evidence['config']['input_archive'], str(archive))
            self.assertFalse(Path(evidence['config']['input_file']).exists())

    def test_output_cannot_replace_source_input(self):
        for flag in ('-x', '-i', '--replay-input-file'):
            with self.subTest(flag=flag), self.assertRaisesRegex(SystemExit, 'overwrite an input'):
                validate(config(flag, '/tmp/synthetic-input.txt', '-o', '/tmp/synthetic-input.txt'))

    def test_concurrent_failed_runs_get_distinct_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = config('-f', '-t', 'source', '-o', str(root / 'same.txt'))
            barrier = threading.Barrier(2)
            def commit(index):
                worker = dataclasses.replace(cfg, artifacts_dir=root / str(index), archive_dir=root / str(index) / 'logs')
                session = Session(worker)
                session.add_transcript(Transcript('s', 'm', 'user', f'source {index}'))
                session.add_error(f'failure {index}')
                barrier.wait(timeout=2)
                pipeline.commit_artifacts(worker, session)
                return Path(session.paths['diagnostics'])
            with ThreadPoolExecutor(max_workers=2) as pool:
                paths = list(pool.map(commit, (1, 2)))
            self.assertEqual(len(set(paths)), 2)
            self.assertEqual({json.loads(p.read_text())['errors'][0] for p in paths}, {'failure 1', 'failure 2'})


class LifecycleContracts(unittest.TestCase):
    def test_unattended_and_worker_downloads_never_prompt(self):
        with mock.patch('builtins.input', side_effect=AssertionError('unexpected prompt')):
            with download_prompts(False):
                self.assertFalse(consent_to_download('model', '/tmp/cache', auto_yes=False, interactive=True))
                self.assertTrue(consent_to_download('model', '/tmp/cache', auto_yes=True, interactive=True))
            with ThreadPoolExecutor(max_workers=1) as pool:
                self.assertFalse(pool.submit(consent_to_download, 'model', '/tmp/cache', auto_yes=False, interactive=True).result())

    def test_replay_backpressure_preserves_every_frame_and_errors_finish(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config('--no-asr', '--replay-input-file', str(Path(tmp) / 'audio.wav'), '--replay-speed', '1000')
            runner = CaptureRunner.__new__(CaptureRunner)
            runner.config = cfg
            runner.session = Session(cfg)
            runner.sample_rate = 16000
            runner.audio_queue = queue.Queue(maxsize=1)
            runner.stop_event = threading.Event()
            runner.replay_finished = threading.Event()
            data = b'\x01\x00' * 16000
            with wave.open(str(cfg.replay_input_file), 'wb') as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(data)
            feeder = threading.Thread(target=runner._replay_feeder, args=(cfg.replay_input_file,))
            feeder.start()
            received = bytearray()
            while not runner.replay_finished.is_set() or not runner.audio_queue.empty():
                try:
                    received.extend(runner.audio_queue.get(timeout=0.1))
                except queue.Empty:
                    pass
            feeder.join(timeout=2)
            self.assertEqual(received, data)
            self.assertEqual(runner.session.audio_queue_full_count, 0)
            runner.replay_finished.clear()
            runner._replay_feeder(Path(tmp) / 'missing.wav')
            self.assertTrue(runner.replay_finished.is_set())
            self.assertTrue(runner.session.run_failed)

    def test_capture_failure_preserves_queued_pcm(self):
        cfg = config('--no-asr', '--replay-input-file', '/tmp/unused.wav')
        runner = CaptureRunner.__new__(CaptureRunner)
        runner.config = cfg
        runner.session = Session(cfg)
        runner.sample_rate = 16000
        runner.replay_mode = True
        runner.temp_dir = Path('/tmp')
        runner.stop_event = threading.Event()
        runner.audio_queue = queue.Queue()
        runner.audio_queue.put(b'\x01\x00' * 100)
        runner._threads = []
        with mock.patch('speech_note.capture.convert_to_pcm_wav'), \
             mock.patch.object(runner, '_start_capture_threads'), \
             mock.patch.object(runner, '_spawn'), \
             mock.patch.object(runner, '_consume_until_stopped', side_effect=RuntimeError('capture failed')):
            with self.assertRaisesRegex(RuntimeError, 'capture failed'):
                runner.run()
        self.assertEqual(runner.session.recorded_audio, b'\x01\x00' * 100)

    def test_audio_loss_marks_run_failed(self):
        session = Session(config())
        session.note_audio_queue_full()
        session.note_audio_queue_full()
        self.assertTrue(session.run_failed)
        self.assertEqual(len(session.errors), 1)

    def test_local_server_batch_has_one_owner_at_a_time(self):
        cfg = config('-f', '--parallel', '--organizer-mode', 'llama', '--organizer-provider', 'local')
        entries = [pipeline.BatchEntry(Path(f'/tmp/{i}.wav'), Path('/tmp')) for i in range(2)]
        threads = []
        def run(item):
            threads.append(threading.current_thread())
            return Session(item)
        with mock.patch.object(pipeline, 'run_file_pipeline', side_effect=run), redirect_stderr(io.StringIO()):
            pipeline._run_batch(cfg, entries)
        self.assertEqual(threads, [threading.main_thread()] * 2)


if __name__ == '__main__':
    unittest.main()
