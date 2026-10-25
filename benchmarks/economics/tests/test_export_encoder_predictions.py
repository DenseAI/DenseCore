import importlib.util
from pathlib import Path
import sys
import unittest
import copy
import io
import json
import hashlib
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import redirect_stdout

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import export_encoder_predictions as exporter


class ExportPredictionTests(unittest.TestCase):
    def row(self):
        return {'id':'x','source_id':'source','gold_label':'a','sequences':[{}],
                'request':{'questions':{'q':{'type':'choice','criteria':['a','b','c','d']}}}}

    def test_mixed_length_batch_executes_exact_groups_and_restores_workload_order(self):
        rows = []
        for index, length in enumerate([3, 2, 3, 2, 3]):
            row = copy.deepcopy(self.row())
            row['id'] = str(index)
            row['sequences'] = [{'token_ids': [index]*length}]
            rows.append(row)
        class FakeNative:
            calls = []
            def __call__(self, row):
                self.calls.append(('single', [row['id']]))
                return [{'logits':[float(row['id']),0.,0.,0.], 'act_logits':[0.,0.]}]
            def call_batch(self, batch):
                self.calls.append(('batch', [r['id'] for r in batch]))
                if len({len(r['sequences'][0]['token_ids']) for r in batch}) != 1:
                    raise AssertionError('mixed lengths sent to native')
                return [[{'logits':[float(row['id']),0.,0.,0.], 'act_logits':[0.,0.]}] for row in batch]
        class FlushedRaw(io.StringIO):
            flushes = 0
            def flush(self): self.flushes += 1
        raw = FlushedRaw()
        native = FakeNative()
        result, sizes = exporter.run_predictions(native, rows, {}, 2, raw)
        self.assertEqual([r['id'] for r in result], ['0','1','2','3','4'])
        self.assertEqual([r['logits'][0] for r in result], [0.,1.,2.,3.,4.])
        self.assertEqual(native.calls, [('batch',['1','3']), ('batch',['0','2']), ('single',['4'])])
        self.assertEqual(sizes, [2,2,1])
        self.assertEqual(raw.flushes, 3)
        recorded = [json.loads(line) for line in raw.getvalue().splitlines()]
        self.assertEqual([r['id'] for r in recorded], ['1','3','0','2','4'])
        self.assertEqual([r['actual_batch_size'] for r in recorded], [2,2,2,2,1])
        self.assertEqual([r['workload_index'] for r in result], list(range(5)))

    def test_batch_limits_and_failed_batch_do_not_record_partial_results(self):
        row = self.row()
        row['sequences'] = [{'token_ids':[1,2]}]
        for size in (0,65,True):
            with self.assertRaises(ValueError): list(exporter.prediction_batches([row],size))
        class FailedNative:
            def call_batch(self, rows): raise RuntimeError('interrupted')
        raw = io.StringIO()
        with self.assertRaises(RuntimeError): exporter.run_predictions(FailedNative(),[row,row],{},2,raw)
        self.assertEqual(raw.getvalue(), '')

    def test_export_binds_mapped_dependencies_and_rejects_midrun_change(self):
        dependencies = {name:'a'*64 for name in ('libggml.so','libggml-cpu.so','libggml-base.so')}
        class Function:
            def __init__(self, value): self.value = value
            def __call__(self, handle): return self.value
        class FakeNative:
            def __init__(self, change):
                self.change, self.calls, self.closed = change, 0, False
                self.handle = object()
                self.lib = SimpleNamespace(DenseCoreDecisionMetadata=Function(b'{"general.architecture":"laya","densecore.execution_precision":"fp32"}'),
                                           DenseCoreDecisionTokenizerJSON=Function(b'tokenizer'))
            def __call__(self, row): return [{'logits':[1.,0.,0.,0.], 'act_logits':[0.,0.]}]
            def runtime_dependencies_sha256(self):
                self.calls += 1
                result = dict(dependencies)
                if self.change and self.calls > 1: result['libggml-cpu.so'] = 'b'*64
                return result
            def close(self): self.closed = True
        for change in (False,True):
            with tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                model, library, workload_path, output = [directory/name for name in ('model.gguf','lib.so','workload.json','output.json')]
                model.write_bytes(b'model');library.write_bytes(b'library')
                row = self.row()
                row['truncated'] = False
                row['sequences'] = [{'token_ids':[1,2,3,4,5,6], 'token_keys':['a','b','c','d','e','f'], 'markers':[1,2,3,4], 'question_type':0}]
                workload = {'schema':'systemone-workload-v1', 'requests':[row], 'provenance':{'model_sha256':hashlib.sha256(b'model').hexdigest(), 'tokenizer_sha256':hashlib.sha256(b'tokenizer').hexdigest()}}
                workload_path.write_text(json.dumps(workload))
                args = SimpleNamespace(model=model, library=library, workload=workload_path, output=output, threads=1, precision='fp32', batch_size=1)
                native = FakeNative(change)
                with patch.object(exporter, 'Native', return_value=native), redirect_stdout(io.StringIO()):
                    if change:
                        with self.assertRaisesRegex(ValueError, 'runtime dependencies changed'): exporter.export(args)
                        self.assertFalse(output.exists())
                        self.assertTrue(Path(str(output)+'.raw.jsonl').exists())
                    else:
                        result = exporter.export(args)
                        self.assertEqual(result['runtime_dependencies_sha256'], dependencies)
                        self.assertTrue(result['runtime_dependencies_stable'])
                        header = json.loads(Path(str(output)+'.raw.jsonl').read_text().splitlines()[0])
                        self.assertEqual(header['bindings']['runtime_dependencies_sha256'], dependencies)
                self.assertTrue(native.closed)

    def test_julia_uses_raw_softmax_and_rejects_nonunit_model_temperature(self):
        out = [{'logits':[2.,0.,0.,0.], 'act_logits':[0.,0.]}]
        metadata = {'general.architecture':'julia', 'julia.temperature.0':1.,
                    'laya.temperature_by_options': {'choice:3-5':5.}}
        pred = exporter.calibrated_prediction(self.row(), out, metadata)
        self.assertEqual(pred['temperature'], 1.)
        self.assertAlmostEqual(pred['probabilities'][0], .7112345942275938)
        metadata['julia.temperature.0'] = 2.
        with self.assertRaises(ValueError): exporter.calibrated_prediction(self.row(), out, metadata)

    def test_calibration_matches_metadata_and_restores_option_order(self):
        row = self.row()
        row['request']['questions']['q']['option_order'] = [3,2,1,0]
        out = [{'logits':[0.,0.,0.,4.], 'act_logits':[1.,2.]}]
        pred = exporter.calibrated_prediction(row, out, {'laya.temperature.0':4.,'laya.temperature_by_options':'{"choice:3-5":2.0}'})
        self.assertEqual(pred['predicted_label'], 'a')
        self.assertEqual(pred['logits'], [4.,0.,0.,0.])
        self.assertEqual(pred['temperature'], 2.)
        self.assertAlmostEqual(pred['probabilities'][0], 0.7112345942275938)
        self.assertTrue(pred['correct'])

    def test_exporter_uses_annotated_distribution_for_brier(self):
        row = self.row()
        row['source_gold_probabilities'] = dict(zip(['a','b','c','d'], [.25]*4))
        pred = exporter.calibrated_prediction(row, [{'logits':[0.]*4, 'act_logits':[0.,0.]}], {})
        self.assertEqual(pred['brier'], 0.)
        self.assertEqual(pred['gold_probabilities'], [.25]*4)

    def test_nonfinite_or_wrong_shape_never_produces_prediction(self):
        for logits in ([float('nan'),0.,0.,0.], [1.]):
            with self.assertRaises(ValueError):
                exporter.calibrated_prediction(self.row(),[{'logits':logits,'act_logits':[0.,0.]}],{})

    def test_stable_softmax_and_temperature_clamp(self):
        pred = exporter.calibrated_prediction(self.row(),[{'logits':[10000.,10000.,10000.,10000.],'act_logits':[0.,0.]}],{'laya.temperature.0':0.01})
        self.assertEqual(pred['probabilities'], [0.25]*4)
        self.assertEqual(pred['temperature'], .5)
        self.assertEqual(pred['brier'], .75)


if __name__=='__main__': unittest.main()
