"""Pure gates only; these tests are not real c8/runtime qualification.

Run with: python3 -m unittest discover -s benchmarks/personal -p 'test_*.py'
"""
import threading
import time
import unittest

from check_concurrency_budget import (DetailedMetrics, parity_errors, plan_errors,
                                      pressure_errors, rejection_valid, timed_daemon)


def negative():
    return dict(label='negative', request_id=-3, callback_events=0, completion_events=0,
                submit_error='native request admission required_bytes=134217729 available_bytes=134217728')


class ConcurrencyBudgetGateTests(unittest.TestCase):
    def test_explicit_runtime_cap_context_threads_and_none_required(self):
        plan = dict(effective_context=512, effective_concurrency=8, effective_threads=8, spec_selected=1)
        self.assertFalse(plan_errors(plan, 512))
        for key in plan:
            with self.subTest(field=key):
                self.assertEqual(len(plan_errors(dict(plan, **{key: 0}), 512)), 1)

    def test_sampled_ids_not_text_or_native_count_alone(self):
        row = dict(label='c8', terminal_events=1, completion_events=1, generated_tokens=128,
                   finish_reason=2, token_ids=list(range(128)))
        self.assertFalse(parity_errors([row], list(range(128))))
        changed = dict(row, token_ids=list(range(127))+[999])
        self.assertIn('sampled IDs', parity_errors([changed], row['token_ids'])[0])

    def test_early_eos_or_duplicate_terminal_cannot_pass_length_parity(self):
        for field, value in [('terminal_events', 2), ('completion_events', 0),
                             ('generated_tokens', 127), ('finish_reason', 1)]:
            with self.subTest(field=field, value=value):
                row = dict(label='c8', terminal_events=1, completion_events=1, generated_tokens=128,
                           finish_reason=2, token_ids=list(range(128)))
                row[field] = value
                self.assertTrue(parity_errors([row], row['token_ids']))

    def test_stop_can_fit_but_prompt_rejection_proves_bounded_admission(self):
        accepted = dict(label='stop', request_id=11, callback_events=1, completion_events=1)
        self.assertFalse(pressure_errors([accepted, negative()]))
        self.assertIn('not established', pressure_errors([accepted, accepted])[0])

    def test_negative_requires_synchronous_sized_error_and_no_callbacks(self):
        for field, value in [('request_id', 1), ('callback_events', 1), ('completion_events', 1),
                             ('submit_error', 'required_bytes=10 available_bytes=10'),
                             ('submit_error', 'out of memory')]:
            with self.subTest(field=field, value=value):
                row = negative()
                row[field] = value
                self.assertFalse(rejection_valid(row))

    def test_late_callback_invalidates_previously_valid_rejection(self):
        row = negative()
        self.assertTrue(rejection_valid(row))
        row['callback_events'] += 1
        self.assertTrue(pressure_errors([row]))

    def test_daemon_timeout_does_not_wait_for_native_return(self):
        release = threading.Event()
        started = time.monotonic()
        try:
            with self.assertRaises(TimeoutError):
                timed_daemon(lambda: release.wait(10), .02)
            self.assertLess(time.monotonic()-started, 1)
        finally:
            release.set()

    def test_daemon_error_preserves_original_failure(self):
        def fail():
            raise ValueError('producer failure')
        with self.assertRaisesRegex(ValueError, 'producer failure'):
            timed_daemon(fail, 1)

    def test_native_metrics_uses_platform_long_and_preserves_field_order(self):
        # Full header ABI order; actual runtime will provide the measurements.
        fields = [key for key, _ in DetailedMetrics._fields_]
        self.assertEqual(fields[:5], ['active_requests', 'total_requests', 'completed_requests',
                                     'failed_requests', 'pending_requests'])
        self.assertEqual(fields[-5:], ['kv_cache_usage_percent', 'avg_batch_size', 'current_batch_size',
                                      'oom_errors', 'timeout_errors'])

    def test_truncated_callback_baseline_cannot_validate_itself(self):
        row = dict(label='baseline', terminal_events=1, completion_events=1, generated_tokens=128,
                   finish_reason=2, token_ids=list(range(127)))
        self.assertIn('count', parity_errors([row], row['token_ids'])[0])


if __name__ == '__main__':
    unittest.main()
