import unittest
import threading
import json
import tempfile
from pathlib import Path
from check_mtp_parity import compare_native, validate_native, wait_for_first_token, extract_mtp_counters, load_c1_baseline


def sample(ids=(10, 20)):
    return {'case': 'probe', 'repeat': 0, 'token_ids': list(ids), 'generated_tokens': len(ids),
            'terminal_events': 1, 'finish_reason': 1}


def group(mode, concurrency, ids=(10, 20)):
    return {'mode': mode, 'concurrency': concurrency,
            'runtime_plan': {'spec_selected': 1 if mode == 'none' else 2},
            'requests': [sample(ids)], 'cancellation': {'passed': True}}


class ExactParityGateTest(unittest.TestCase):
    def test_every_mode_and_concurrency_must_match_c1_ids(self):
        rows = [group('none', 1), group('none', 4), group('mtp', 1), group('mtp', 4)]
        self.assertEqual(compare_native(rows), [])
        rows[-1]['requests'][0]['token_ids'] = [10, 99]
        self.assertIn('index 1', compare_native(rows)[0])

    def test_cancel_probe_fails_immediately_after_terminal_error(self):
        row = {'first_token': threading.Event(), 'done': threading.Event()}
        row['done'].set()
        with self.assertRaisesRegex(RuntimeError, 'terminated before'): wait_for_first_token(row, 60)

    def test_partial_matrix_cannot_pass(self):
        failures = compare_native([group('none', 1), group('mtp', 1), group('mtp', 4)])
        self.assertTrue(any('missing for none/c4' in failure for failure in failures))

    def test_callback_loss_or_duplicate_terminal_cannot_pass_parity(self):
        row = sample()
        row['generated_tokens'] = 3
        with self.assertRaisesRegex(ValueError, 'token count'): validate_native(row)
        row = sample()
        row['terminal_events'] = 2
        with self.assertRaisesRegex(ValueError, 'exactly once'): validate_native(row)

    def test_empty_or_failed_requests_cannot_pass(self):
        with self.assertRaises(ValueError): validate_native(sample(()))
        row = sample()
        row['finish_reason'] = 3
        with self.assertRaises(ValueError): validate_native(row)

    def test_requested_mode_and_cancel_evidence_are_required(self):
        rows = [group('none', 1), group('mtp', 1)]
        rows[1]['runtime_plan']['spec_selected'] = 1
        rows[1]['cancellation']['passed'] = False
        failures = compare_native(rows)
        self.assertEqual(len(failures), 2)


if __name__ == '__main__': unittest.main()

class MtpCounterEvidenceTest(unittest.TestCase):
    def test_cumulative_counters_are_counted_once_per_request_and_width_is_retained(self):
        rows = extract_mtp_counters("""
[MTPVerify] request_id=1 target_width=4 proposed=9 verified=3 accepted=2 rejected=1 target_rollbacks=1
[MTPVerify] request_id=1 target_width=1 proposed=12 verified=4 accepted=3 rejected=1 target_rollbacks=2
[MTPVerify] request_id=2 target_width=2 proposed=2 verified=1 accepted=0 rejected=1 target_rollbacks=1
""")
        self.assertEqual(rows['maximum_target_width'], 4)
        self.assertEqual(rows['proposed'], 14)
        self.assertEqual(rows['accepted'], 3)
        self.assertEqual(rows['rejected'], 2)
        self.assertEqual(rows['target_rollbacks'], 3)

class BaselineReuseGateTest(unittest.TestCase):
    def test_reuse_requires_exact_artifacts_and_valid_c1_pair(self):
        provenance = {key: value for key, value in (
            ('model_sha256', 'm'), ('library_sha256', 'l'), ('dependencies', {'l':'sha'}),
            ('sampling', {'temperature':0}), ('draft_n_max',2), ('cpu_affinity',[0]), ('cpu_profile',['cpu']))}
        baseline = dict(provenance, passed=True, native=[group('none',1), group('mtp',1)])
        for row in baseline['native']:
            row['runtime_plan'].update(effective_context=2048, effective_threads=8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'baseline.json'
            path.write_text(json.dumps(baseline))
            self.assertEqual(len(load_c1_baseline(path, provenance, 2048, 8)), 2)
            with self.assertRaisesRegex(ValueError, 'thread mismatch'):
                load_c1_baseline(path, provenance, 2048, 4)
            with self.assertRaisesRegex(ValueError, 'model_sha256'):
                load_c1_baseline(path, dict(provenance, model_sha256='different'), 2048, 8)
            baseline['native'][1]['requests'][0]['token_ids'] = [99,20]
            path.write_text(json.dumps(baseline))
            with self.assertRaisesRegex(ValueError, 'native evidence invalid'):
                load_c1_baseline(path, provenance, 2048, 8)

class BaselineCommandLineTest(unittest.TestCase):
    def test_new_cells_with_baseline_reach_parent(self):
        from unittest.mock import patch
        import check_mtp_parity as probe
        argv = ['probe', '--library', 'lib.so', '--model', 'model.gguf', '--out', 'out',
                '--concurrency', '2', '4', '--baseline-summary', 'baseline.json']
        with patch('sys.argv', argv), patch.object(probe, 'parent', return_value=0) as run:
            self.assertEqual(probe.main(), 0)
            self.assertEqual(run.call_args.args[0].concurrency, [2,4])
        with patch('sys.argv', argv[:-2]), patch.object(probe, 'parent') as run:
            with self.assertRaises(SystemExit): probe.main()
            run.assert_not_called()
