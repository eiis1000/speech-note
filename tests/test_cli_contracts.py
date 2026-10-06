"""Real HTTP + subprocess CLI checks, using synthetic text and no hosted API.

Set SPEECH_NOTE_TEST_BINARY to test the packaged launcher outside the checkout.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class CliContracts(unittest.TestCase):
    def test_provider_fallback_failure_and_output_recovery(self):
        seen = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def reply(self, body):
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self.reply({'data': [{'id': name} for name in ('bad', 'good')]})

            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                model = request['model']
                seen.append(model)
                choice = {'message': {'content': 'Complete transcript.'}, 'finish_reason': 'stop'}
                if model == 'bad':
                    choice.update(message={'content': 'Partial'}, finish_reason='error',
                                  error={'code': 502, 'message': 'synthetic upstream failure'})
                self.reply({'model': model, 'choices': [choice]})

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                env = os.environ.copy()
                for name in ('PYTHONHOME', 'VIRTUAL_ENV', 'OPENROUTER_API_KEY'):
                    env.pop(name, None)
                env['XDG_CONFIG_HOME'] = str(root / 'config')
                env['PYTHONPATH'] = str(Path(__file__).resolve().parent.parent)
                binary = env.get('SPEECH_NOTE_TEST_BINARY')
                command = [binary] if binary else [sys.executable, '-m', 'speech_note']
                for model, blocked, expected in [('bad,good', False, 0), ('bad', False, 1), ('good', True, 1)]:
                    with self.subTest(model=model, blocked=blocked):
                        directory = root / str(len(list(root.iterdir())))
                        directory.mkdir()
                        env['XDG_STATE_HOME'] = str(directory / 'state')
                        output = directory / 'clean.txt'
                        if blocked:
                            output.mkdir()
                        seen.clear()
                        run = subprocess.run(command + [
                            '-f', '-t', 'Complete transcript.', '-o', str(output),
                            '--organizer-provider', 'local', '--organizer-model', model,
                            '--organizer-api-base', f'http://127.0.0.1:{server.server_port}/v1/chat/completions',
                        ], cwd=tmp, env=env, capture_output=True, text=True, timeout=30)
                        self.assertEqual(run.returncode, expected, run.stderr)
                        evidence = json.loads(next((directory / 'state' / 'speech-note' / 'diagnostics').glob('*.json')).read_text())
                        self.assertEqual(bool(list(directory.glob('*diagnostics.json'))), bool(expected))
                        self.assertEqual(evidence['run_failed'], bool(expected))
                        self.assertEqual(seen, model.split(','))
                        if expected == 0:
                            self.assertEqual(output.read_text().strip(), 'Complete transcript.')
                            self.assertEqual(len(evidence['cleanup']['attempts']), 2)
                        elif blocked:
                            self.assertEqual(evidence['cleanup']['text'], 'Complete transcript.')
                        else:
                            self.assertFalse(output.exists())
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
