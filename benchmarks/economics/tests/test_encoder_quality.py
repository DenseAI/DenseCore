import copy
import importlib.util
from pathlib import Path
import unittest
import tempfile
import json
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

quality = load('evaluate_encoder_quality')
prepare = load('prepare_encoder_workload')
normalize = load('normalize_encoder_dataset')


def fixture(n=1):
    provenance = {"dataset": "fancyzhx/ag_news", "revision": "a" * 40, "split": "test",
                  "source_sha256": "b" * 64, "model_sha256": "c" * 64, "tokenizer_sha256": "d" * 64}
    rows = [{"id": str(i), "token_ids": list(range(100)), "attention_mask": [1]*100, "token_keys": [str(j) for j in range(100)], "padding_tokens": 0, "truncated": False, "dropped_tokens": 0,
             "request": {"state": "news", "questions": {"topic": {"type": "choice", "criteria": ["world", "sports", "business", "tech"]}}},
             "gold_label": "world"} for i in range(n)]
    workload = prepare.prepare(rows, provenance)
    workload["artifact_sha256"] = "e" * 64
    run = {"precision": "fp32", "model_sha256": "c" * 64, "tokenizer_sha256": "d" * 64,
           "runtime_dependencies_sha256": {name: "2"*64 for name in ('libggml.so','libggml-cpu.so','libggml-base.so')},
           "runtime_dependencies_stable": True, "artifact_sha256": "f" * 64, "implementation_sha256": "1" * 64, "workload_sha256": "e" * 64,
           "predictions": [{"id": str(i), "labels": ["world", "sports", "business", "tech"],
                            "probabilities": [1., 0., 0., 0.], "logits": [1., 0., 0., 0.]} for i in range(n)]}
    candidate = copy.deepcopy(run)
    candidate["precision"] = "int8"
    return rows, provenance, workload, run, candidate


class EncoderEvaluationTests(unittest.TestCase):
    def test_quality_requires_dependency_evidence_but_allows_distinct_variants(self):
        _, _, workload, ref, cand = fixture()
        for malformed in (None, {}, {'libggml.so':'2'*64}, {name:'wrong' for name in ('libggml.so','libggml-cpu.so','libggml-base.so')}):
            changed = copy.deepcopy(cand)
            changed['runtime_dependencies_sha256'] = malformed
            with self.assertRaises(ValueError): quality.evaluate(workload, ref, changed, set())
        changed = copy.deepcopy(cand)
        changed.pop('runtime_dependencies_stable')
        with self.assertRaises(ValueError): quality.evaluate(workload, ref, changed, set())
        cand['runtime_dependencies_sha256']['libggml-cpu.so'] = '3'*64
        self.assertEqual(ref['implementation_sha256'], cand['implementation_sha256'])
        result = quality.evaluate(workload, ref, cand, set())
        self.assertEqual(result['runtime_dependencies_sha256'], cand['runtime_dependencies_sha256'])
        self.assertEqual(result['reference_runtime_dependencies_sha256'], ref['runtime_dependencies_sha256'])
        self.assertNotEqual(result['runtime_dependencies_sha256'], result['reference_runtime_dependencies_sha256'])

    def test_prediction_evidence_uses_exact_file_hash_not_declared_model_id(self):
        _, _, workload, ref, cand = fixture()
        with tempfile.TemporaryDirectory() as directory:
            reference_path = Path(directory) / 'reference.json'
            candidate_path = Path(directory) / 'candidate.json'
            reference_path.write_text(json.dumps(ref))
            candidate_path.write_text(json.dumps(cand))
            result = quality.evaluate(workload, quality.load_predictions(reference_path), quality.load_predictions(candidate_path), set())
            self.assertEqual(result['reference_artifact_sha256'], hashlib.sha256(reference_path.read_bytes()).hexdigest())
            self.assertNotEqual(result['reference_artifact_sha256'], result['candidate_artifact_sha256'])
            before = result['candidate_artifact_sha256']
            cand['predictions'][0]['probabilities'] = [.7,.1,.1,.1]
            candidate_path.write_text(json.dumps(cand))
            self.assertNotEqual(before, quality.load_predictions(candidate_path)['artifact_sha256'])

    def test_soft_distribution_brier_is_authoritative_and_validated(self):
        _, _, workload, ref, cand = fixture()
        workload['requests'][0]['source_gold_probabilities'] = dict(zip(['world','sports','business','tech'], [.7,.1,.1,.1]))
        for run in (ref, cand):
            run['predictions'][0]['probabilities'] = [.7,.1,.1,.1]
        result = quality.evaluate(workload, ref, cand, set())
        self.assertAlmostEqual(result['reference_brier'], 0.)
        self.assertAlmostEqual(result['candidate_brier'], 0.)
        self.assertEqual(result['reference_accuracy'], 1.)
        workload['requests'][0]['source_gold_probabilities'].pop('tech')
        with self.assertRaises(ValueError): quality.evaluate(workload, ref, cand, set())

    def test_final_quality_includes_short_examples_and_excludes_tuning_sources(self):
        rows, provenance, *_ = fixture(2)
        for row in rows:
            row['token_ids'] = list(range(40))
            row['attention_mask'] = [1]*40
            row['token_keys'] = [str(i) for i in range(40)]
        result = prepare.prepare(rows, provenance, quality_all_lengths=True, excluded_source_ids={'0'})
        self.assertEqual([r['id'] for r in result['requests']], ['1'])
        self.assertEqual(result['requests'][0]['compute_input_tokens'], 40)
        self.assertEqual(result['purpose'], 'final_quality_holdout')
        self.assertEqual(result['rejected'][0]['reason'], 'excluded_tuning_or_diagnostic_source')

    def test_typed_decisions_preserves_source_and_question_types(self):
        rows = [{"id": "source1", "state": '{"risk": "low"}', "questions": {
            "a": {"type": "choice", "criteria": {"a": "A", "b": "B", "c": "C", "d": "D"}},
            "b": {"type": "choice", "criteria": {"a": "A", "b": "B", "c": "C", "d": "D"}},
            "score": {"type": "score", "criteria": ["a", "b", "c", "d"]}},
            "gold": {name: {"label": "a", "probabilities": {"a": 1.}} for name in ('a', 'b')}}]
        normalized = list(normalize.normalize(rows, 'LocalLLaMA/typed-decisions', 'test'))
        self.assertEqual(len(normalized), 2)
        self.assertEqual(normalized[0]['source_id'], normalized[1]['source_id'])
        self.assertEqual(normalized[0]['request']['state'], rows[0]['state'])
        self.assertNotEqual(normalized[0]['id'], normalized[1]['id'])

    def test_unpadded_receipt_requires_explicit_all_one_attention_mask(self):
        rows, provenance, *_ = fixture()
        for mask in (None, [1]*99, [1]*99+[0], [True]*100):
            changed = copy.deepcopy(rows)
            changed[0]['attention_mask'] = mask
            with self.assertRaises(ValueError): prepare.prepare(changed, provenance)

    def test_natural_lengths_use_actual_tokens(self):
        rows, provenance, *_ = fixture()
        rows[0]['token_ids'] = list(range(251))
        rows[0]['attention_mask'] = [1]*251
        rows[0]['token_keys'] = [str(j) for j in range(251)]
        result = prepare.prepare(rows, provenance)
        self.assertEqual(result['requests'][0]['useful_input_tokens'], 251)
        self.assertEqual(result['requests'][0]['bucket_tokens'], 256)
        self.assertEqual(result['bucket_status']['128'], 'unavailable')

    def test_reject_padding_truncation_and_mutable_source(self):
        rows, provenance, *_ = fixture()
        for key, value in [('padding_tokens', 1), ('truncated', True), ('dropped_tokens', 1)]:
            changed = copy.deepcopy(rows)
            changed[0][key] = value
            self.assertEqual(prepare.prepare(changed, provenance)['rejected'][0]['reason'], 'padding_or_truncation')
        provenance['revision'] = 'main'
        with self.assertRaises(ValueError):
            prepare.prepare(rows, provenance)

    def test_small_identical_sample_is_unverified(self):
        _, _, workload, ref, cand = fixture()
        self.assertEqual(quality.evaluate(workload, ref, cand, set())['status'], 'unverified')

    def test_paired_degradation_fails(self):
        _, _, workload, ref, cand = fixture(200)
        for row in cand['predictions']:
            row['probabilities'] = [0., 1., 0., 0.]
        self.assertEqual(quality.evaluate(workload, ref, cand, set())['status'], 'fail')

    def test_large_paired_quality_can_pass(self):
        _, _, workload, ref, cand = fixture(20000)
        self.assertEqual(quality.evaluate(workload, ref, cand, set())['status'], 'pass')

    def test_no_missing_duplicate_reordered_predictions_or_tuning_leakage(self):
        _, _, workload, ref, cand = fixture(2)
        for change in ('missing', 'duplicate', 'labels', 'nan', 'model'):
            altered = copy.deepcopy(cand)
            if change == 'missing': altered['predictions'].pop()
            if change == 'duplicate': altered['predictions'].append(altered['predictions'][0])
            if change == 'labels': altered['predictions'][0]['labels'].reverse()
            if change == 'nan': altered['predictions'][0]['probabilities'][0] = float('nan')
            if change == 'model': altered['model_sha256'] = '0' * 64
            with self.assertRaises(ValueError): quality.evaluate(workload, ref, altered, set())
        with self.assertRaises(ValueError): quality.evaluate(workload, ref, cand, {'0'})

    def test_fp32_preserves_strict_option_and_independent_action_parity(self):
        _, _, workload, ref, cand = fixture()
        cand['precision'] = 'fp32'
        self.assertEqual(quality.evaluate(workload, ref, cand, set(), 'fp32')['status'], 'unverified')
        cand['independent_action_parity'] = 'pass'
        self.assertEqual(quality.evaluate(workload, ref, cand, set(), 'fp32')['status'], 'pass')
        cand['predictions'][0]['logits'][0] += .002
        self.assertEqual(quality.evaluate(workload, ref, cand, set(), 'fp32')['status'], 'fail')


if __name__ == '__main__': unittest.main()
