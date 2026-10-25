import ctypes as C
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import check_mtp_json as probe


def finished_row(text='{"answer":42}'):
    row = probe.new_record(probe.CASES[0], 0)
    probe.record_piece(row, text.encode(), 7, False)
    row.update(generated_tokens=2, finish_reason=1, completion_events=1)
    probe.record_piece(row, b'', -1, True)
    return probe.serializable(row)


class JsonValidatorTest(unittest.TestCase):
    def test_only_exact_semantic_object_and_empty_structural_object_pass(self):
        self.assertEqual(probe.parse_object(' \n{"answer":42}\t', probe.CASES[0]), {'answer': 42})
        self.assertEqual(probe.parse_object('{}', probe.CASES[1]), {})
        for text in ['{"answer":true}', '{"answer":"42"}', '{"answer":42.0}', '{"answer":42,"extra":1}',
                     '[42]', '{}', '{"answer":NaN}', '{"answer":Infinity}', '{"answer":42,"answer":42}',
                     'Here: {"answer":42}', '{"answer":42} done', '<think>42',
                     '<think>x</think>{"answer":42}', '{"answer":42', '\ufffd{"answer":42}', '\ud800']:
            with self.subTest(text=repr(text)), self.assertRaises((ValueError, UnicodeError)):
                probe.parse_object(text, probe.CASES[0])

    def test_length_or_missing_duplicate_terminal_metadata_cannot_pass(self):
        for change in [{'finish_reason': 2}, {'finish_reason': 3}, {'terminal_events': 2},
                       {'completion_events': 0}, {'completion_events': 2}, {'generated_tokens': 0},
                       {'generated_tokens': True}, {'generated_tokens': -1}, {'utf8_errors': ['broken']}]:
            row = finished_row()
            row.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                probe.validate_native(row, probe.CASES[0])
        row = finished_row()
        row['delivered_token_ids'] = [1, 2, 3]
        with self.assertRaises(ValueError): probe.validate_native(row, probe.CASES[0])

    def test_delivered_subset_and_negative_one_text_flush_never_invent_sample(self):
        row = probe.new_record(probe.CASES[0], 0)
        probe.record_piece(row, b'{"answer":', 7, False)
        probe.record_piece(row, b'42}', -1, False)
        row.update(generated_tokens=2, finish_reason=1, completion_events=1)
        probe.record_piece(row, b'', -1, True)
        clean = probe.serializable(row)
        probe.validate_native(clean, probe.CASES[0])
        self.assertEqual(clean['delivered_token_ids'], [7])
        self.assertEqual(clean['parsed_object'], {'answer': 42})
        self.assertTrue(row['first_token'].is_set())

    def test_utf8_fragments_are_owned_and_decoded_incrementally(self):
        row = probe.new_record(probe.CASES[0], 0)
        probe.record_piece(row, b'\xc3', 1, False)
        probe.record_piece(row, b'\xa9', -1, False)
        probe.record_piece(row, b'', -1, True)
        self.assertEqual(''.join(row['pieces']), 'é')
        self.assertEqual(row['utf8_errors'], [])
        row = probe.new_record(probe.CASES[0], 0)
        probe.record_piece(row, b'\xc3', 1, False)
        probe.record_piece(row, b'', -1, True)
        self.assertTrue(row['utf8_errors'])

    def test_nonterminal_whitespace_or_negative_one_flush_after_terminal_fails(self):
        for token_id in (-1, 7):
            row = probe.new_record(probe.CASES[0], 0)
            probe.record_piece(row, b'{"answer":42}', 7, False)
            probe.record_completion(row, 4, 1)
            probe.record_piece(row, b'', -1, True)
            probe.record_piece(row, b' ', token_id, False)
            with self.subTest(token_id=token_id), self.assertRaisesRegex(ValueError, 'callback order'):
                probe.validate_native(probe.serializable(row), probe.CASES[0])

    def test_one_metadata_arriving_after_one_terminal_still_fails(self):
        row = probe.new_record(probe.CASES[0], 0)
        probe.record_piece(row, b'{"answer":42}', 7, False)
        probe.record_piece(row, b'', -1, True)
        probe.record_completion(row, 2, 1)
        self.assertEqual(row['completion_events'], row['terminal_events'])
        with self.assertRaisesRegex(ValueError, 'callback order'):
            probe.validate_native(probe.serializable(row), probe.CASES[0])

    def test_cancellation_requires_explicit_cancel_error_not_arbitrary_failure(self):
        row = finished_row()
        row.update(finish_reason=3, terminal_text='Error: Engine shutdown')
        with self.assertRaisesRegex(ValueError, 'explicitly'): probe.validate_native(row, {'id': 'cancel'}, True)
        row['terminal_text'] = 'Error: Request canceled'
        probe.validate_native(row, {'id': 'cancel'}, True)

    def test_unsupported_public_range_seed_minp_and_stop_fields_rejected(self):
        for key in probe.UNSUPPORTED_FIELDS:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'unsupported'):
                probe.validate_public_fields({key: 1})


class Function:
    def __init__(self, call): self.call = call
    def __call__(self, *args): return self.call(*args)


class Library:
    def __init__(self, late=False):
        self.mode = 1
        self.next_id = 100
        self.callbacks = {}
        self.late = late
        self.free_calls = 0
        self.submitted = []
        self.InitEngineWithOptions = Function(self.init)
        self.FreeEngine = Function(self.free)
        self.DenseCoreGetLastError = Function(lambda: b'fake native error')
        self.DenseCoreGetRuntimePlan = Function(self.plan)
        self.DenseCoreSetGenerationCompletionCallback = Function(self.set_completed)
        self.CountTokens = Function(lambda *_: 1)
        self.DenseCoreTokenizeText = Function(self.tokenize)
        self.CancelRequest = Function(self.cancel)
        self.SubmitRequestWithSamplingConstraintsCallbackEx = Function(self.submit)
    def init(self, *args):
        self.mode = args[-1]._obj.spec_type
        return 17
    def plan(self, _handle, output):
        output._obj.effective_context = 2048
        output._obj.effective_concurrency = 1
        output._obj.effective_threads = 8
        output._obj.spec_selected = self.mode
        return 0
    def set_completed(self, _handle, callback):
        self.completed = callback
        return 0
    def tokenize(self, _handle, text, _bos, _eos, output, _capacity):
        output[0] = 21 if text == b'{' else 22
        return 1
    def emit(self, callback, data, token_id, finished, userdata):
        buffer = C.create_string_buffer(data)
        callback(C.cast(buffer, C.c_void_p), len(data), token_id, finished, userdata)
    def submit(self, _h, prompt, budget, _lora, _temp, _p, _k, _rep, stops, json_mode,
               allowed, allowed_count, strict, denied, denied_count, callback, userdata):
        self.next_id += 1
        rid = self.next_id
        self.submitted.append({'json_mode': json_mode, 'stops': stops, 'mask': list(allowed or [])[:allowed_count],
                               'strict': strict, 'budget': budget})
        self.callbacks[rid] = callback, userdata
        if b'10000' in prompt:
            self.emit(callback, b'{"numbers":[', 9, 0, userdata)
        else:
            if allowed_count:
                self.emit(callback, b'{}', 21, 0, userdata)
            else:
                self.emit(callback, b'{"answer":', 7, 0, userdata)
                self.emit(callback, b'42}', -1, 0, userdata)
            self.completed(2, 1, userdata)
            self.emit(callback, b'', -1, 1, userdata)
        return rid
    def cancel(self, _handle, rid):
        callback, userdata = self.callbacks[rid]
        self.completed(1, 3, userdata)
        self.emit(callback, b'Error: Request canceled', -1, 1, userdata)
        return 0
    def free(self, _handle):
        self.free_calls += 1
        if self.late:
            callback, userdata = next(iter(self.callbacks.values()))
            self.completed(2, 1, userdata)
            self.emit(callback, b'', -1, 1, userdata)


class NativeAdapterTest(unittest.TestCase):
    def args(self, directory, mode='none'):
        return SimpleNamespace(library=Path(directory)/'lib.so', model=Path(directory)/'model.gguf',
                               out=Path(directory)/'result.json', mode=mode, draft=2, threads=8, context=2048,
                               child_concurrency=1, repeats=1, max_tokens=512, timeout=1,
                               temperature=0, top_p=1, top_k=1, repetition_penalty=1)
    def test_exact_public_abi_json_constraints_metadata_cancel_and_recovery(self):
        for mode in ('none', 'mtp'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                lib = Library()
                with patch.object(probe.C, 'CDLL', return_value=lib): probe.native_child(self.args(directory, mode))
                report = json.loads((Path(directory)/'result.json').read_text())
                self.assertNotIn('error', report)
                self.assertEqual(lib.free_calls, 1)
                self.assertTrue(report['cancellation']['passed'])
                self.assertEqual(report['cancellation']['recovery']['parsed_object'], {'answer': 42})
                self.assertTrue(all(row['json_mode'] == 1 and row['stops'] is None for row in lib.submitted))
                self.assertTrue(any(row['mask'] == [21,22] and row['strict'] == 1 for row in lib.submitted))
                self.assertEqual(report['requests'][0]['delivered_token_ids'], [7])

    def test_callback_owners_survive_until_free_and_late_duplicate_is_not_waived(self):
        with tempfile.TemporaryDirectory() as directory:
            lib = Library(late=True)
            with patch.object(probe.C, 'CDLL', return_value=lib), self.assertRaisesRegex(ValueError, 'exactly once'):
                probe.native_child(self.args(directory))
            report = json.loads((Path(directory)/'result.json').read_text())
            self.assertIn('error', report)
            self.assertEqual(report['partial_requests'][0]['terminal_events'], 2)
            self.assertEqual(lib.free_calls, 1)

    def test_unsupported_brace_shape_is_explicit_not_expanded(self):
        lib = Library()
        lib.CountTokens.call = lambda *_: 2
        with self.assertRaisesRegex(ValueError, 'unsupported brace tokenizer shape'): probe.brace_ids(lib, 17)
        lib.CountTokens.call = lambda *_: 1
        lib.DenseCoreTokenizeText.call = lambda *_: 0
        with self.assertRaisesRegex(ValueError, 'unsupported brace tokenizer shape'): probe.brace_ids(lib, 17)


class HttpContractTest(unittest.TestCase):
    def test_json_and_strict_mask_are_sent_and_followup_not_plain_text(self):
        row = {'content': '{}', 'reasoning': '', 'finish_reason': 'stop', 'done': True,
               'usage': {'completion_tokens': 2}, 'events': [{'data': {'choices': [{'finish_reason': 'stop'}]}}]}
        with patch.object(probe.protocol, 'http_request', return_value=row) as call:
            result = probe.http_request('http://local', dict(probe.CASES[1], brace_token_ids=[21,22]), True, 512, 1,
                                        {'temperature': 0, 'top_k': 1, 'top_p': 1})
        self.assertNotIn('error', result)
        wire = call.call_args.args[-1]
        self.assertEqual(wire['response_format'], {'type': 'json_object'})
        self.assertEqual(wire['allowed_token_ids'], [21,22])
        self.assertTrue(wire['allowed_tokens_strict'])
        self.assertNotIn('stop', wire)

    def test_http_failure_is_persisted_with_runtime_snapshot(self):
        class Profile:
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def read(self): return b'{"native":{"effective_concurrency":4}}'
        with tempfile.TemporaryDirectory() as directory, patch.object(probe.protocol, 'http_request', return_value={'error': 'malformed JSON body', 'events': []}), patch.object(probe.protocol.urllib.request, 'urlopen', return_value=Profile()):
            row = probe.http_request('http://local', probe.CASES[0], True, 512, 1, {}, Path(directory))
            files = list(Path(directory).glob('*.json'))
            self.assertEqual(len(files), 1)
            self.assertEqual(json.loads(files[0].read_text()), row)
            self.assertEqual(row['error'], 'malformed JSON body')
            self.assertEqual(row['runtime_profile']['native']['effective_concurrency'], 4)

    def test_real_http_helper_serializes_public_fields_and_requires_done(self):
        class Response:
            status = 200
            def __init__(self, done): self.done = done
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def __iter__(self):
                events = [
                    {'choices': [{'delta': {'content': '{}'}, 'finish_reason': None}]},
                    {'choices': [{'delta': {}, 'finish_reason': 'stop'}]},
                    {'choices': [], 'usage': {'completion_tokens': 2}},
                ]
                for event in events: yield b'data: ' + json.dumps(event).encode() + b'\n'
                if self.done: yield b'data: [DONE]\n'
        for done in (False, True):
            with self.subTest(done=done), patch.object(probe.protocol.urllib.request, 'urlopen', return_value=Response(done)) as opened:
                row = probe.http_request('http://local', dict(probe.CASES[1], brace_token_ids=[21,22]), True, 512, 1, {})
                self.assertEqual('error' not in row, done)
                wire = json.loads(opened.call_args.args[0].data)
                self.assertEqual(wire['response_format'], {'type': 'json_object'})
                self.assertEqual(wire['allowed_token_ids'], [21,22])
                self.assertTrue(wire['allowed_tokens_strict'])
                self.assertNotIn('stop', wire)

    def test_length_reasoning_bad_usage_duplicate_finish_and_bad_json_fail(self):
        for changed in [{'finish_reason': 'length'}, {'done': False}, {'reasoning': '<think>42'},
                        {'usage': {'completion_tokens': 0}}, {'usage': {'completion_tokens': True}},
                        {'content': '{"answer":42,"answer":42}'},
                        {'events': [{'data': {'choices': [{'finish_reason': 'stop'}, {'finish_reason': 'stop'}]}}]}]:
            row = {'content': '{"answer":42}', 'reasoning': '', 'finish_reason': 'stop', 'done': True,
                   'usage': {'completion_tokens': 2}, 'events': [{'data': {'choices': [{'finish_reason': 'stop'}]}}]}
            row.update(changed)
            with self.subTest(changed=changed), patch.object(probe.protocol, 'http_request', return_value=row):
                self.assertIn('error', probe.http_request('http://local', probe.CASES[0], True, 512, 1, {}))


class ArtifactGateTest(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        lib, server, model, dep = [root/name for name in ('libdensecore.so', 'server', 'model.gguf', 'libggml.so')]
        for p in (lib, server, model, dep): p.write_text(p.name)
        source = 'a'*40
        report = root/'unit-report.json'
        report.write_text(json.dumps({'source': source, 'scope': 'mock fixture only'}))
        doc = {'revision': source, 'json_completion_unit_qualified': True,
               'json_completion_unit_source': source,
               'json_completion_unit_report': {'path': str(report), 'sha256': probe.protocol.digest(report)},
               'files': {str(p.resolve()): probe.protocol.digest(p) for p in (lib, server, dep)}}
        manifest = root/'manifest.json'
        manifest.write_text(json.dumps(doc))
        args = SimpleNamespace(library=lib, server=server, model=model, artifact_manifest=manifest,
                               expected_model_sha256=probe.protocol.digest(model), qualified_source=source)
        return args, doc
    def test_qualified_manifest_requires_exact_source_all_dependencies_and_model_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            args, document = self.fixture(directory)
            probe.verify_qualified(args)
            for change in [{'revision': 'b'*40}, {'json_completion_unit_qualified': False},
                           {'json_completion_unit_source': 'b'*40},
                           {'json_completion_unit_report': {'path': str(Path(directory)/'unit-report.json'), 'sha256': '0'*64}},
                           {'files': {str(args.library.resolve()): probe.protocol.digest(args.library),
                                      str(args.server.resolve()): probe.protocol.digest(args.server)}}]:
                doc = dict(document, **change)
                args.artifact_manifest.write_text(json.dumps(doc))
                with self.subTest(change=change), self.assertRaises(ValueError): probe.verify_qualified(args)
            args.artifact_manifest.write_text(json.dumps(document))
            args.model.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'model SHA'): probe.verify_qualified(args)
    def test_unit_report_bytes_and_its_source_are_bound_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            args, doc = self.fixture(directory)
            report = Path(doc['json_completion_unit_report']['path'])
            report.write_text(json.dumps({'source': 'b'*40}))
            with self.assertRaisesRegex(ValueError, 'report hash'): probe.verify_qualified(args)
            doc['json_completion_unit_report']['sha256'] = probe.protocol.digest(report)
            args.artifact_manifest.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, 'report source'): probe.verify_qualified(args)

    def test_old_manifest_is_rejected_before_any_native_library_load(self):
        with tempfile.TemporaryDirectory() as directory:
            args, document = self.fixture(directory)
            document['json_completion_unit_qualified'] = False
            args.artifact_manifest.write_text(json.dumps(document))
            argv = ['probe', '--native-child', '--mode', 'none', '--out', str(Path(directory)/'result.json')]
            for key in ('library', 'server', 'model', 'artifact_manifest', 'expected_model_sha256', 'qualified_source'):
                argv.extend(['--' + key.replace('_', '-'), str(getattr(args, key))])
            with patch('sys.argv', argv), patch.object(probe.C, 'CDLL') as loaded, patch.object(probe.traceback, 'print_exc'):
                self.assertEqual(probe.main(), 1)
                loaded.assert_not_called()
            self.assertIn('qualified', json.loads((Path(directory)/'result.json').read_text())['error'])

    def test_old4545_cannot_run_even_if_manifest_flag_is_forged(self):
        with tempfile.TemporaryDirectory() as directory:
            args, doc = self.fixture(directory)
            args.qualified_source = doc['revision'] = '4545e05' + '0'*33
            args.artifact_manifest.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, 'old4545'): probe.verify_qualified(args)


if __name__ == '__main__': unittest.main()
