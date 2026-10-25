import unittest
from check_mtp_parity import compare_native, validate_native


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
