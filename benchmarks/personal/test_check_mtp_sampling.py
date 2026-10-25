import json
from types import SimpleNamespace
import unittest
import threading
from unittest.mock import patch

import check_mtp_sampling as smoke


CAPITAL = {'id': 'capital', 'prompt': 'Capital?', 'expected': r'\bParis\b'}


def native():
    return {'token_ids': [10, 11], 'pieces': ['Paris', ''], 'terminal_events': 1,
            'completion_events': 1, 'generated_tokens': 2, 'finish_reason': 1, 'utf8_errors': []}


class SamplingQualityTest(unittest.TestCase):
    def test_random_wording_is_allowed_without_id_or_cross_mode_equality(self):
        a, b = native(), native()
        b.update(token_ids=[30, 40, 50], generated_tokens=3, pieces=['The capital is ', 'Paris', '.'])
        smoke.validate_native(a, CAPITAL)
        smoke.validate_native(b, CAPITAL)
        self.assertNotEqual(a['token_ids'], b['token_ids'])

    def test_empty_reasoning_only_wrong_answer_or_unfinished_sentence_fail(self):
        for value in ['', '<think>Paris', '<think>Paris</think>', 'London', '\ufffdParis', '\ud800Paris']:
            with self.subTest(value=repr(value)), self.assertRaises((ValueError, UnicodeError)):
                smoke.validate_final(value, CAPITAL)
        case = dict(CAPITAL, expected=r'water', min_words=8, sentence=True)
        with self.assertRaises(ValueError):
            smoke.validate_final('Clean water matters', case)
        with self.assertRaises(ValueError):
            smoke.validate_final('Clean drinking water matters because it keeps people healthy', case)
        smoke.validate_final('Clean drinking water matters because it keeps people healthy.', case)

    def test_length_completion_missing_metadata_and_utf8_errors_fail(self):
        for changed in [{'finish_reason': 2}, {'terminal_events': 2}, {'completion_events': 0},
                        {'generated_tokens': 3}, {'utf8_errors': ['incomplete UTF-8']}]:
            row = native()
            row.update(changed)
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                smoke.validate_native(row, CAPITAL)

    def test_multibyte_sentence_is_preserved(self):
        case = {'expected': 'café', 'min_words': 5, 'sentence': True}
        self.assertEqual(smoke.validate_final('Un café chaud rend la matinée agréable.', case),
                         'Un café chaud rend la matinée agréable.')

    def test_unexercised_rejection_is_reported_not_waived(self):
        coverage = smoke.branch_coverage({'maximum_target_width': 3, 'accepted': 10,
                                          'rejected': 0, 'target_rollbacks': 5})
        self.assertFalse(coverage['rejection'])
        self.assertFalse(all(coverage.values()))


class HttpSamplingContractTest(unittest.TestCase):
    def test_positive_probability_fields_and_stop_are_in_wire_payload_not_minp(self):
        probability = {'temperature': 0.7, 'top_k': 40, 'top_p': 0.95, 'repetition_penalty': 1.05}
        case = dict(CAPITAL, stop=['<END>'])
        class Response:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def read(self):
                return json.dumps({'choices': [{'message': {'content': 'Paris'}, 'finish_reason': 'stop'}],
                                   'usage': {'completion_tokens': 1}}).encode()
        with patch.object(smoke.protocol.urllib.request, 'urlopen', return_value=Response()) as opened:
            row = smoke.http_request('http://local', case, False, 1024, 2, probability)
        self.assertNotIn('error', row)
        payload = json.loads(opened.call_args.args[0].data)
        for key, value in probability.items(): self.assertEqual(payload[key], value)
        self.assertEqual(payload['stop'], ['<END>'])
        self.assertNotIn('min_p', payload)
        self.assertEqual(row['usage']['completion_tokens'], 1)  # no invented nativeEOS/HTTP count equality

    def test_truncated_http_and_stop_text_leak_fail(self):
        for changed in [{'finish_reason': 'length'}, {'content': 'Paris<END>ignore'}]:
            row = dict(content='Paris', reasoning='', finish_reason='stop', usage={'completion_tokens': 1})
            row.update(changed)
            # Native protocol validator independently rejects length; wrapper rejects escaped stop text.
            if row['finish_reason'] == 'length':
                with self.assertRaises(ValueError):
                    from compare_servers import validate_response
                    validate_response(row['content'], '', row['finish_reason'], row['usage'], r'Paris')
            else:
                with patch.object(smoke.protocol, 'http_request', return_value=row):
                    result = smoke.http_request('http://local', dict(CAPITAL, stop=['<END>']), False, 1024, 1, {})
                    self.assertIn('stop text leaked', result['error'])

    def test_surrogate_in_reasoning_cannot_pass_valid_content(self):
        row = dict(content='Paris', reasoning='\ud800', finish_reason='stop')
        with patch.object(smoke.protocol, 'http_request', return_value=row):
            result = smoke.http_request('http://local', CAPITAL, True, 1024, 1, {})
        self.assertIn('error', result)



class StreamAndProvenanceTest(unittest.TestCase):
    def test_stream_requires_done_and_valid_final_stop_usage(self):
        probability = {'temperature': 0.7, 'top_k': 40, 'top_p': 0.95, 'repetition_penalty': 1.05}
        events = [b'data: {"choices":[{"delta":{"content":"Paris"},"finish_reason":null}]}\n',
                  b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":1}}\n']
        class Response:
            status = 200
            def __init__(self, lines): self.lines = lines
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def __iter__(self): return iter(self.lines)
        for done in [False, True]:
            lines = events + ([b'data: [DONE]\n'] if done else [])
            with patch.object(smoke.protocol.urllib.request, 'urlopen', return_value=Response(lines)):
                result = smoke.http_request('http://local', CAPITAL, True, 1024, 2, probability)
            self.assertEqual('error' not in result, done)
        malformed = [b'data: {"choices":[{"delta":{"content":"\xff"}}]}\n']
        with patch.object(smoke.protocol.urllib.request, 'urlopen', return_value=Response(malformed)):
            self.assertIn('error', smoke.http_request('http://local', CAPITAL, True, 1024, 2, probability))

    def test_canonical_and_model_bytes_must_match_before_execution(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library, server, model, manifest = [root / name for name in ['lib.so', 'server', 'model.gguf', 'artifact.json']]
            for path in [library, server, model]: path.write_bytes(path.name.encode())
            manifest.write_text(json.dumps({'revision': 'candidate', 'files': {
                str(path.resolve()): smoke.protocol.digest(path) for path in [library, server]}}))
            args = SimpleNamespace(library=library, server=server, model=model,
                                   artifact_manifest=manifest, expected_model_sha256=smoke.protocol.digest(model))
            self.assertEqual(smoke.verify_artifacts(args)['revision'], 'candidate')
            server.write_bytes(b'different')
            with self.assertRaisesRegex(ValueError, 'canonical artifact mismatch'):
                smoke.verify_artifacts(args)

    def test_invalid_unicode_diagnostics_remain_writable_without_hiding_failure(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.json'
            smoke.protocol.write(path, {'content': '\ud800', 'error': 'invalid UTF-8'})
            decoded = json.loads(path.read_text(encoding='utf-8', errors='strict'))
            self.assertEqual(decoded['error'], 'invalid UTF-8')
            self.assertEqual(decoded['content'], '\ud800')


class PromptRequirementConsistencyTest(unittest.TestCase):
    def test_word_minimum_is_explicit_in_each_sentence_prompt(self):
        from pathlib import Path
        cases = json.loads(Path(smoke.__file__).with_name('mtp_sampling_cases.json').read_text())
        for case in cases:
            if case.get('min_words'):
                self.assertIn(f"at least {case['min_words']} words", case['prompt'])

    def test_semantic_success_does_not_claim_strict_instruction_format(self):
        case = dict(CAPITAL, strict_format_pattern='Paris')
        self.assertEqual(smoke.validate_final('Paris, France.', case), 'Paris, France.')
        self.assertFalse(smoke.strict_format('Paris, France.', case)['passed'])
        self.assertTrue(smoke.strict_format('Paris', case)['passed'])
        self.assertFalse(smoke.strict_format('42.', {'strict_format_pattern': '42'})['passed'])


if __name__ == '__main__':
    unittest.main()


class NativeTextOnlyFlushTest(unittest.TestCase):
    def test_flush_preserves_unicode_text_without_counting_an_id(self):
        row = dict(token_ids=[], pieces=[], utf8_errors=[], first_token=threading.Event())
        smoke.protocol.record_native_piece(row, SimpleNamespace(token_id=7, text=b'Hi'), strict_utf8=True)
        smoke.protocol.record_native_piece(row, SimpleNamespace(token_id=8, text=b''), strict_utf8=True)
        smoke.protocol.record_native_piece(row, SimpleNamespace(token_id=-1, text='<é'.encode()), strict_utf8=True)
        self.assertEqual(row['token_ids'], [7, 8])
        self.assertEqual(''.join(row['pieces']), 'Hi<é')
        self.assertTrue(row['first_token'].is_set())
        self.assertEqual(row['utf8_errors'], [])

    def test_text_only_flush_does_not_signal_a_generated_token(self):
        row = dict(token_ids=[], pieces=[], utf8_errors=[], first_token=threading.Event())
        smoke.protocol.record_native_piece(row, SimpleNamespace(token_id=-1, text=b'prefix'), strict_utf8=True)
        self.assertFalse(row['first_token'].is_set())
        smoke.protocol.record_native_piece(row, SimpleNamespace(token_id=-1, text=b'\xff'), strict_utf8=True)
        self.assertEqual(len(row['utf8_errors']), 1)
