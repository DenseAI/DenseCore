import copy
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from systemone_economics import digest, load_workload, summarize, validate_artifacts, pipeline_implementation_sha256, binding_keys, validate_server_identity, loaded_runtime_dependencies_sha256
from run_systemone_load import run, http_call, Native, require_unpadded


class EncoderAccounting(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        seq = {'token_ids': [10, 20, 0], 'token_keys': ['question', 'state', None], 'markers': [0], 'question_type': 0}
        self.obj = {'schema': 'systemone-workload-v1', 'provenance': {'model_sha256': 'b'*64, 'tokenizer_sha256': 'c'*64}, 'requests': [{'id': 'a', 'truncated': False,
            'sequences': [seq, copy.deepcopy(seq)], 'request': {'state': 'x', 'questions': {'q1': {}, 'q2': {}}}}]}
        self.workload_path = self.write('workload.json', self.obj)
        self.workload = load_workload(self.workload_path)
        self.manifest = {'_sha256': 'a'*64, 'boundary': 'full_model', 'model_sha256': 'b'*64,
            'tokenizer_sha256': 'c'*64, 'implementation_sha256': 'd'*64, 'precision': 'fp32',
            'workload_sha256': digest(self.workload_path), 'result_cache': False,
            'runtime_dependencies_sha256': {name: '9'*64 for name in ('libggml.so', 'libggml-cpu.so', 'libggml-base.so')}}
        self.evaluation = {k: self.manifest[k] for k in ('model_sha256', 'tokenizer_sha256', 'implementation_sha256', 'precision', 'workload_sha256', 'runtime_dependencies_sha256')}
        self.evaluation['status'] = 'pass'
        self.evaluation['runtime_dependencies_stable'] = True
        self.evidence = self.write('evidence.json', self.evaluation)
        self.quality = self.write('quality.json', {**{k: self.manifest[k] for k in ('model_sha256',
            'tokenizer_sha256', 'implementation_sha256', 'precision', 'workload_sha256', 'runtime_dependencies_sha256')},
            'status': 'pass', 'evaluation_artifact': self.evidence.name, 'evaluation_artifact_sha256': digest(self.evidence)})
        self.price = {'currency': 'USD', 'hourly_usd': 3.6, 'includes_all_instance_and_disk_costs': True,
            **{k: 'fixed' for k in ('snapshot_id', 'captured_at', 'purchase_basis', 'provider', 'region', 'machine_type', 'source_url')}}
        self.resource = {k: self.price[k] for k in ('provider', 'region', 'machine_type')}
        self.manifest['resource'] = self.resource
        self.price_path = self.write('price.json', self.price)
        self.manifest.update(quality_sha256=digest(self.quality), pricing_sha256=digest(self.price_path))
        self.meta = {'runtime_dependencies_sha256': copy.deepcopy(self.manifest['runtime_dependencies_sha256']), 'runtime_dependencies_stable': True, 'runtime_dependencies_origin': 'observed_process_maps', 'resource': dict(self.resource), 'resource_identity_status': 'operator_attested', 'run_id': 'fixture-run', 'manifest_sha256': 'a'*64, 'boundary': 'full_model', 'elapsed_s': 10., 'submission_stop_s': 5., 'record_count': 2}
        self.records = [{'run_id': 'fixture-run', 'attempt_id': i, 'workload_id': 'a', 'start_s': i, 'end_s': 8.+i,
            'useful_tokens': 2, 'compute_tokens': 6, 'success': i == 0,
            'error': None if i == 0 else 'HTTP 429', 'unresolved_backend_work': False,
            'response': [{'logits': [0.], 'act_logits': [0., 0.]}]*2} for i in range(2)]

    def write(self, name, obj):
        path = self.root/name
        path.write_text(json.dumps(obj))
        return path

    def summary(self):
        return summarize(self.manifest, self.workload, self.meta, self.records, self.price)

    def test_success_only_unique_useful_tokens_and_full_drain_cost(self):
        out = self.summary()
        self.assertEqual(out['useful_tokens'], 2)
        self.assertEqual(out['successful_compute_tokens'], 6)
        self.assertEqual(out['decisions'], 2)
        self.assertEqual(out['drain_s'], 5)
        self.assertAlmostEqual(out['tokens_per_dollar'], 200)
        self.assertEqual(out['error_rate'], .5)
        self.assertEqual(out['gpu_2x_verdict'], 'unavailable')

    def test_reject_padding_inflation(self):
        self.records[0]['useful_tokens'] = 6
        with self.assertRaisesRegex(ValueError, 'token count'):
            self.summary()

    def test_reject_duplicate_attempt(self):
        self.records[1]['attempt_id'] = 0
        with self.assertRaisesRegex(ValueError, 'duplicate attempt'):
            self.summary()

    def test_reject_omitted_failure(self):
        self.records.pop()
        with self.assertRaisesRegex(ValueError, 'missing request'):
            self.summary()

    def test_reject_omitted_drain(self):
        self.meta['elapsed_s'] = 5
        with self.assertRaisesRegex(ValueError, 'cost interval'):
            self.summary()

    def test_reject_unresolved_timeout_work(self):
        self.records[1]['unresolved_backend_work'] = True
        with self.assertRaisesRegex(ValueError, 'drain unverified'):
            self.summary()

    def test_reject_wrong_model_quality_and_price_artifacts(self):
        validate_artifacts(self.manifest, self.workload_path, self.quality, self.price_path)
        for key in ('model_sha256', 'tokenizer_sha256', 'implementation_sha256', 'precision'):
            bad = dict(self.manifest, **{key: 'e'*64})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'quality|provenance'):
                validate_artifacts(bad, self.workload_path, self.quality, self.price_path)
        self.price['hourly_usd'] = 1
        self.write('price.json', self.price)
        with self.assertRaisesRegex(ValueError, 'pricing_sha256'):
            validate_artifacts(self.manifest, self.workload_path, self.quality, self.price_path)

    def test_reject_rehashed_failed_or_mismatched_independent_evaluation(self):
        for field, value in [('status', 'fail'), ('model_sha256', 'f'*64), ('precision', 'int8')]:
            changed = dict(self.evaluation, **{field: value})
            self.write('evidence.json', changed)
            quality = json.loads(self.quality.read_text())
            quality['evaluation_artifact_sha256'] = digest(self.evidence)
            self.write('quality.json', quality)
            manifest = dict(self.manifest, quality_sha256=digest(self.quality))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'evaluation'):
                validate_artifacts(manifest, self.workload_path, self.quality, self.price_path)

    def test_reject_rehashed_contradictory_workload_provenance(self):
        self.obj['provenance']['model_sha256'] = 'f'*64
        self.write('workload.json', self.obj)
        manifest = dict(self.manifest, workload_sha256=digest(self.workload_path))
        with self.assertRaisesRegex(ValueError, 'workload provenance'):
            validate_artifacts(manifest, self.workload_path, self.quality, self.price_path)

    def test_incomplete_http_response_does_not_prove_backend_drain(self):
        import http.client
        def call(row):
            time.sleep(.3)
            raise http.client.IncompleteRead(b'partial', 100)
        meta, records = run(call, self.workload['requests'], .1, 1, self.root/'run', 'a'*64, 'http_e2e', resource=self.resource)
        self.assertEqual(len(records), 1)
        meta.update(runtime_dependencies_sha256=self.manifest['runtime_dependencies_sha256'], runtime_dependencies_stable=True)
        self.assertTrue(records[0]['unresolved_backend_work'])
        self.assertEqual(meta['memory']['status'], 'unavailable')
        with self.assertRaisesRegex(ValueError, 'drain unverified'):
            summarize(dict(self.manifest, boundary='http_e2e', endpoint='http://server/v1/systemone', server_identity_sha256='0'*64), self.workload,
                      dict(meta, runtime_dependencies_origin='operator_attested_server_deployment', server_identity_sha256='0'*64, endpoint='http://server/v1/systemone'), records, self.price)

    def test_reject_same_price_different_machine(self):
        self.manifest['resource'] = dict(self.resource, machine_type='different-machine')
        with self.assertRaisesRegex(ValueError, 'resource machine_type mismatch'):
            validate_artifacts(self.manifest, self.workload_path, self.quality, self.price_path)

    def test_reject_run_resource_different_from_priced_manifest(self):
        self.meta['resource']['machine_type'] = 'different-machine'
        with self.assertRaisesRegex(ValueError, 'run/manifest resource machine_type mismatch'):
            self.summary()

    def test_reject_masked_padding_counted_as_useful(self):
        self.obj['requests'][0]['sequences'][0]['attention_mask'] = [1, 0, 1]
        with self.assertRaisesRegex(ValueError, 'masked padding token counted as useful'):
            load_workload(self.write('padding.json', self.obj))

    def test_explicit_padding_counts_and_native_mask_rejection(self):
        row = self.obj['requests'][0]
        for seq in row['sequences']:
            seq['attention_mask'] = [1, 1, 0]
        row['padding_tokens'] = 2
        loaded = load_workload(self.write('padding.json', self.obj))
        self.assertEqual(loaded['requests'][0]['_useful_tokens'], 2)
        with self.assertRaisesRegex(ValueError, 'native ABI requires unpadded'):
            require_unpadded(row['sequences'][0])
        row['padding_tokens'] = 0
        with self.assertRaisesRegex(ValueError, 'padding count mismatch'):
            load_workload(self.write('padding.json', self.obj))

    def test_reject_malformed_attention_mask(self):
        for mask in ([1], [1, 2, 1], [True, True, True]):
            self.obj['requests'][0]['sequences'][0]['attention_mask'] = mask
            with self.subTest(mask=mask), self.assertRaisesRegex(ValueError, 'attention mask'):
                load_workload(self.write('padding.json', self.obj))

    def prepare_pipeline(self):
        self.manifest.update(boundary='encoder_pipeline', executable_sha256='1'*64, native_library_sha256='2'*64)
        self.manifest['implementation_sha256'] = pipeline_implementation_sha256('1'*64, '2'*64)
        self.evaluation = {k: self.manifest[k] for k in binding_keys(self.manifest)}
        self.evaluation['status'] = 'pass'
        self.evaluation['runtime_dependencies_stable'] = True
        self.write('evidence.json', self.evaluation)
        quality = dict(self.evaluation, evaluation_artifact='evidence.json', evaluation_artifact_sha256=digest(self.evidence))
        self.write('quality.json', quality)
        self.manifest['quality_sha256'] = digest(self.quality)
        self.meta.update({k: self.manifest[k] for k in binding_keys(self.manifest)})
        self.meta.update(boundary='encoder_pipeline', executable_sha256='1'*64, native_library_sha256='2'*64,
                         implementation_sha256=self.manifest['implementation_sha256'], native_library={'sha256': '2'*64})

    def test_pipeline_composite_matches_cross_language_canonical_digest(self):
        self.assertEqual(pipeline_implementation_sha256('1'*64, '2'*64),
                         'aaae6b3a041b174b3798064caf55bbb7d5f4629b2d50e277a5a8fe55ff62b179')
        self.prepare_pipeline()
        validate_artifacts(self.manifest, self.workload_path, self.quality, self.price_path)

    def test_reject_pipeline_changed_executable_with_same_native(self):
        self.prepare_pipeline()
        self.meta['executable_sha256'] = '3'*64
        with self.assertRaisesRegex(ValueError, 'actual pipeline executable_sha256 mismatch'):
            self.summary()
        bad_manifest = dict(self.manifest, executable_sha256='3'*64)
        with self.assertRaisesRegex(ValueError, 'pipeline implementation composite mismatch'):
            validate_artifacts(bad_manifest, self.workload_path, self.quality, self.price_path)

    def test_reject_rehashed_pipeline_quality_with_wrong_executable(self):
        self.prepare_pipeline()
        quality = json.loads(self.quality.read_text())
        quality['executable_sha256'] = '3'*64
        self.write('quality.json', quality)
        self.manifest['quality_sha256'] = digest(self.quality)
        with self.assertRaisesRegex(ValueError, 'quality executable_sha256 mismatch'):
            validate_artifacts(self.manifest, self.workload_path, self.quality, self.price_path)

    def test_reject_changed_ggml_dependency_with_same_main_native(self):
        native_before = self.manifest['implementation_sha256']
        self.meta['runtime_dependencies_sha256']['libggml-cpu.so'] = '8'*64
        self.assertEqual(self.manifest['implementation_sha256'], native_before)
        with self.assertRaisesRegex(ValueError, 'actual runtime dependencies mismatch'):
            self.summary()
        changed = copy.deepcopy(self.manifest)
        changed['runtime_dependencies_sha256']['libggml-cpu.so'] = '8'*64
        with self.assertRaisesRegex(ValueError, 'quality runtime_dependencies_sha256 mismatch'):
            validate_artifacts(changed, self.workload_path, self.quality, self.price_path)

    def prepare_http_identity(self):
        self.manifest.update(boundary='http_e2e', endpoint='http://server/v1/systemone')
        fields = ('model_sha256', 'tokenizer_sha256', 'implementation_sha256', 'precision', 'runtime_dependencies_sha256')
        identity = {k: self.manifest[k] for k in fields}
        identity.update(endpoint=self.manifest['endpoint'], origin='operator_attested_server_deployment')
        path = self.write('server-identity.json', identity)
        self.manifest['server_identity_sha256'] = digest(path)
        return path, identity

    def test_http_requires_bound_operator_server_identity(self):
        path, identity = self.prepare_http_identity()
        validate_artifacts(self.manifest, self.workload_path, self.quality, self.price_path, path)
        validate_server_identity(self.manifest, path, 'http://server/v1/systemone')
        with self.assertRaisesRegex(ValueError, 'server identity artifact required'):
            validate_artifacts(self.manifest, self.workload_path, self.quality, self.price_path)
        with self.assertRaisesRegex(ValueError, 'CLI endpoint mismatch'):
            validate_server_identity(self.manifest, path, 'http://another-server/v1/systemone')
        identity['model_sha256'] = '8'*64
        self.write(path.name, identity)
        self.manifest['server_identity_sha256'] = digest(path)
        with self.assertRaisesRegex(ValueError, 'HTTP server identity model_sha256 mismatch'):
            validate_artifacts(self.manifest, self.workload_path, self.quality, self.price_path, path)

    def test_reject_http_client_maps_claimed_as_server_identity(self):
        path, identity = self.prepare_http_identity()
        identity['origin'] = 'observed_process_maps'
        self.write(path.name, identity)
        self.manifest['server_identity_sha256'] = digest(path)
        with self.assertRaisesRegex(ValueError, 'HTTP server identity origin'):
            validate_server_identity(self.manifest, path)

    def test_collector_hashes_actual_dependency_bytes_without_changing_main(self):
        from unittest.mock import patch
        files = [self.root/name for name in ('libggml.so', 'libggml-cpu.so', 'libggml-base.so')]
        for path in files:
            path.write_bytes(b'original dependency')
        native = self.root/'libdensecore_decision.so'
        native.write_bytes(b'unchanged main native')
        before_main = digest(native)
        maps = ''.join(f'0000-1000 r-xp 0 00:00 0 {path}\n' for path in files)
        with patch.object(Path, 'read_text', return_value=maps):
            before = loaded_runtime_dependencies_sha256()
            files[1].write_bytes(b'changed kernel dependency')
            after = loaded_runtime_dependencies_sha256()
        self.assertEqual(digest(native), before_main)
        self.assertNotEqual(before['libggml-cpu.so'], after['libggml-cpu.so'])
        self.assertEqual(before['libggml-base.so'], after['libggml-base.so'])

    def test_runner_captures_monotonic_and_wall_finish_before_serialization(self):
        from unittest.mock import patch
        import datetime as dt
        origin = 1_000_000_000_000
        clock_values = [origin, origin+1_000_000, origin+2_000_000, origin+200_000_000, origin+201_000_000]
        first_wall = dt.datetime(2026, 10, 17, tzinfo=dt.timezone.utc)
        last_wall = first_wall-dt.timedelta(seconds=13.4)
        walls, events = [first_wall, last_wall], []
        class JumpingWallClock:
            @classmethod
            def now(cls, timezone):
                events.append('wall')
                return walls.pop(0)
        original_write = Path.write_text
        def write(path, contents, *args, **kwargs):
            events.append('serialize')
            return original_write(path, contents, *args, **kwargs)
        def call(row):
            return [{'logits': [0.], 'act_logits': [0., 0.]}]*2
        with patch('run_systemone_load.time.monotonic_ns', side_effect=clock_values), \
             patch('run_systemone_load.datetime.datetime', JumpingWallClock), \
             patch('run_systemone_load.Path.write_text', autospec=True, side_effect=write), \
             patch('run_systemone_load.NativeRSSSampler') as sampler:
            sampler.return_value.finish.return_value = {'status': 'unavailable', 'reason': 'test'}
            meta, records = run(call, self.workload['requests'], .1, 1, self.root/'clocks', 'a'*64, 'full_model')
        self.assertEqual(meta['schema'], 'systemone-run-v2')
        self.assertEqual(meta['monotonic_started_ns'], origin)
        self.assertEqual(meta['monotonic_finished_ns'], origin+201_000_000)
        self.assertEqual(meta['elapsed_s'], .201)
        self.assertEqual(meta['finished_at'], last_wall.isoformat())
        self.assertEqual(events, ['wall', 'wall', 'serialize', 'serialize'])
        self.assertEqual(records[0]['start_s'], .001)
        self.assertEqual(records[0]['end_s'], .002)
        self.assertTrue(meta['boot_id'])

    def test_reject_edited_quality_evidence(self):
        self.evidence.write_text('different evidence')
        with self.assertRaisesRegex(ValueError, 'evidence hash'):
            validate_artifacts(self.manifest, self.workload_path, self.quality, self.price_path)

    def test_reject_truncation_and_duplicate_logical_tokens(self):
        self.obj['requests'][0]['truncated'] = True
        with self.assertRaisesRegex(ValueError, 'truncated'):
            load_workload(self.write('bad.json', self.obj))
        self.obj['requests'][0]['truncated'] = False
        self.obj['requests'][0]['sequences'][0]['token_keys'] = ['same', 'same', None]
        with self.assertRaisesRegex(ValueError, 'duplicate token key'):
            load_workload(self.write('bad.json', self.obj))

    def test_native_batch_ctypes_preserves_request_outputs(self):
        import threading
        import types
        native = Native.__new__(Native)
        native.handle = 1
        native.lock = threading.Lock()
        def predict(handle, inputs, count, outputs, error, capacity):
            self.assertEqual(count, 4)
            for i in range(count):
                outputs[i].logits[0] = inputs[i].tokens[0] + i
                outputs[i].act_logits[0] = inputs[i].markers[0]
                outputs[i].act_logits[1] = inputs[i].token_count
            return 1
        native.lib = types.SimpleNamespace(DenseCoreDecisionPredictBatch=predict)
        rows = [copy.deepcopy(self.workload['requests'][0]) for _ in range(2)]
        rows[1]['sequences'][0]['token_ids'][0] = 99
        out = native.call_batch(rows)
        self.assertEqual(out[0][0]['logits'], [10.])
        self.assertEqual(out[1][0]['logits'], [101.])
        self.assertEqual(out[1][1]['act_logits'], [0., 3.])
        rows[0]['sequences'][0]['attention_mask'] = [1, 0, 1]
        with self.assertRaisesRegex(ValueError, 'native ABI requires unpadded'):
            native(rows[0])
        with self.assertRaisesRegex(ValueError, 'native ABI requires unpadded'):
            native.call_batch(rows)

    def test_batch_runner_records_each_request_and_shared_drain(self):
        class BatchCall:
            def call_batch(inner, rows):
                time.sleep(.3)
                return [[{'logits': [0.], 'act_logits': [0., 0.]}]*2 for row in rows]
        meta, records = run(BatchCall(), self.workload['requests'], .1, 1, self.root/'batch',
                            'a'*64, 'full_model', warmup=2, batch_size=4, resource=self.resource)
        self.assertEqual(len(records), 4)
        self.assertEqual(meta['batch_size'], 4)
        self.assertEqual(meta['warmup_rounds'], 2)
        self.assertEqual(meta['warmup_requests'], 8)
        self.assertEqual(meta['memory']['status'], 'measured')
        self.assertGreater(meta['memory']['peak_rss_bytes'], 0)
        self.assertIn('whole_native_client_process', meta['memory']['scope'])
        self.assertEqual({r['attempt_id'] for r in records}, {0, 1, 2, 3})
        self.assertEqual(len({r['start_s'] for r in records}), 1)
        self.assertEqual(len({r['end_s'] for r in records}), 1)
        meta.update(runtime_dependencies_sha256=self.manifest['runtime_dependencies_sha256'], runtime_dependencies_stable=True)
        result = summarize(self.manifest, self.workload, meta, records, self.price)
        self.assertEqual(result['successful_requests'], 4)
        self.assertEqual(result['useful_tokens'], 8)

    def test_run_keeps_failure_and_drains_last_request(self):
        def call(row):
            time.sleep(.3)
            raise ValueError('injected failure')
        meta, records = run(call, self.workload['requests'], .1, 1, self.root/'run', 'a'*64, 'full_model')
        self.assertEqual(len(records), 1)
        self.assertFalse(records[0]['success'])
        self.assertGreater(meta['elapsed_s'], .3)
        self.assertTrue((self.root/'run.records.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
