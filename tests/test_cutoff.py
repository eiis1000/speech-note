"""Provider failure, recovery, and durable evidence contracts; synthetic data only."""
import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from speech_note import annotate
from speech_note.chat import ChatClient
from speech_note.cli import parse_args, resolve_config
from speech_note.model import Transcript
from speech_note.organizer import Organizer
from speech_note.pipeline import run_dry_text_pipeline


TEXT = "We discussed the project and reviewed the evidence before deciding."
SOURCE = Transcript("asr1", "synthetic", "asr-final", TEXT)


def body(text=TEXT, finish="stop", **fields):
    return {
        "id": "synthetic-generation", "model": "served", "provider": "test-provider",
        "usage": {"completion_tokens": 20},
        "choices": [{"message": {"content": text}, "finish_reason": finish}],
        **fields,
    }


def response(payload):
    result = mock.Mock(ok=True, status_code=200)
    result.json.return_value = payload
    result.text = json.dumps(payload)
    return result


def broken():
    result = body("We discussed the", "error")
    result["choices"][0]["error"] = {
        "code": 502, "message": "Upstream output-content filter stopped generation",
        "metadata": {"error_type": "provider_unavailable"},
    }
    return result


def client():
    result = ChatClient(api_base="https://openrouter.ai/api/v1/chat/completions",
                        models=["first", "second"], timeout=1)
    result._catalog_checked = True
    return result


def organizer(transport):
    return Organizer(mode="llama", client=transport, supervisor=None,
                     context_tokens=16384, max_output_tokens=4096)


class CutoffTests(unittest.TestCase):
    def test_http_200_choice_error_recovers_and_retains_provider_evidence(self):
        transport = client()
        notify = mock.Mock()
        with mock.patch("speech_note.chat.requests.post", side_effect=[response(broken()), response(body())]) as post:
            result = transport.chat([], max_tokens=4096, timeout=1, on_model_failure=notify)
        self.assertEqual(result.content, TEXT)
        self.assertEqual(post.call_count, 2)
        notify.assert_called_once_with("first", "second")
        self.assertEqual(result.request_id, "synthetic-generation")
        self.assertEqual(result.provider, "test-provider")
        self.assertEqual(result.usage, {"completion_tokens": 20})
        self.assertEqual(transport.last_attempts[0]["response"], broken())
        self.assertIn("output-content filter", transport.last_attempts[0]["error"])

    def test_error_and_finish_markers_override_even_complete_looking_text(self):
        cases = [body(finish=reason) for reason in ("error", "content_filter", "length", "tool_calls", "unknown", None)]
        cases.append(body(error={"message": "provider failed"}))
        nested = body()
        nested["choices"][0]["error"] = {"message": "provider failed"}
        cases.append(nested)
        refusal = body()
        refusal["choices"][0]["message"]["refusal"] = "cannot complete"
        cases.append(refusal)
        native = body()
        native["choices"][0]["native_finish_reason"] = "max_tokens"
        cases.append(native)
        for payload in cases:
            with self.subTest(payload=payload):
                transport = client()
                with mock.patch("speech_note.chat.requests.post", side_effect=[response(payload), response(body())]):
                    transport.chat([], max_tokens=100, timeout=1)
                self.assertEqual(len(transport.last_attempts), 2)
                self.assertIn("error", transport.last_attempts[0])

    def test_short_and_mid_sentence_stop_replies_trigger_model_fallback(self):
        for text in ("Brief.", "We discussed the project and reviewed the evidence before deciding to"):
            with self.subTest(text=text):
                transport = client()
                with mock.patch("speech_note.chat.requests.post", side_effect=[response(body(text)), response(body())]):
                    result = organizer(transport).cleanup([SOURCE])
                self.assertTrue(result.ok)
                self.assertEqual(result.text, TEXT)
                self.assertEqual(len(result.attempts), 2)

    def test_exhausted_cleanup_never_runs_annotation(self):
        transport = client()
        worker = organizer(transport)
        worker.annotate = True
        worker.annotation_client = mock.Mock()
        with mock.patch("speech_note.chat.requests.post", return_value=response(broken())):
            result = worker.cleanup([SOURCE, Transcript("asr2", "other", "asr-final", TEXT + " Extra.")])
        self.assertFalse(result.ok)
        self.assertEqual(result.text, "")
        self.assertEqual(len(result.attempts), 2)
        worker.annotation_client.chat.assert_not_called()

    def test_annotation_cannot_apply_partial_provider_error_json(self):
        transport = client()
        payload = broken()
        payload["choices"][0]["message"]["content"] = '{"uncertain": []}'
        with mock.patch("speech_note.chat.requests.post", return_value=response(payload)):
            result = annotate.annotate(transport, cleaned_text=TEXT, sources=[SOURCE], context_tokens=16384)
        self.assertEqual(result.text, TEXT)
        self.assertIn("output-content filter", result.error)

    def test_full_auto_recovery_keeps_sources_attempts_and_clean_text_without_secrets(self):
        self._full_auto(recover=True)

    def test_full_auto_exhaustion_keeps_evidence_but_no_clean_output(self):
        self._full_auto(recover=False)

    def _full_auto(self, *, recover):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "new-directory" / "note.txt"
            cfg = resolve_config(parse_args([
                "-f", "-t", TEXT, "-o", str(output), "--organizer-mode", "llama",
                "--organizer-provider", "openrouter",
                "--organizer-model", "first,second", "--organizer-api-base", "http://127.0.0.1:9/v1/chat/completions",
                "--no-annotate-uncertainty",
            ]))
            replies = [response(broken()), response(body() if recover else broken())]
            with mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "synthetic-test-secret"}), \
                 mock.patch("speech_note.chat.requests.get", side_effect=RuntimeError("offline")), \
                 mock.patch("speech_note.chat.requests.post", side_effect=replies), \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                session = run_dry_text_pipeline(cfg)
            self.assertEqual(session.run_failed, not recover)
            self.assertEqual(output.exists(), recover)
            saved = output.with_name("note-diagnostics.json")
            self.assertTrue(saved.exists())
            evidence = json.loads(saved.read_text())
            self.assertEqual(evidence["transcripts"][0]["text"], TEXT)
            attempts = evidence["cleanup"]["attempts"]
            self.assertEqual(len(attempts), 2)
            self.assertEqual(attempts[0]["response"]["choices"][0]["error"]["code"], 502)
            self.assertIn(TEXT, attempts[0]["request"]["messages"][-1]["content"])
            self.assertNotIn("Authorization", saved.read_text())
            self.assertNotIn("synthetic-test-secret", saved.read_text())
            self.assertEqual(evidence["cleanup"]["text"], TEXT if recover else "")

    def test_attempt_history_is_reset_between_calls(self):
        transport = client()
        with mock.patch("speech_note.chat.requests.post", return_value=response(body())):
            transport.chat([], max_tokens=100, timeout=1)
            transport.chat([], max_tokens=100, timeout=1)
        self.assertEqual(len(transport.last_attempts), 1)


if __name__ == "__main__":
    unittest.main()
