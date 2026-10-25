import copy
import unittest

from check_prefix_parity import attach_log_evidence, compare, prefix_evidence


def valid_groups():
    labels = [*(f'repeat{i}' for i in range(1, 5)), *(f'turn{i}' for i in range(1, 6)),
              'after-cancel', 'after-churn']
    rows = [{'label': label, 'request_id': i + 1, 'prompt': label, 'token_ids': [4, 2],
             'generated_tokens': 2, 'finish_reason': 1, 'terminal_events': 1} for i, label in enumerate(labels)]
    group = {'requests': rows, 'cancellation': {'passed': True},
             'prefix_evidence': {'hits': 2, 'restores': 2, 'skipped_tokens': [64, 64]},
             'eviction': {'observed_miss_after_hit': True}}
    return {'off': copy.deepcopy(group), 'on': copy.deepcopy(group)}


class PrefixEvidenceTest(unittest.TestCase):
    def test_exact_ids_include_stop_token(self):
        groups = valid_groups()
        self.assertEqual(compare(groups, True), [])
        groups['on']['requests'][4]['token_ids'][-1] = 99
        self.assertTrue(any('off/on exact' in x for x in compare(groups)))

    def test_missing_or_truncated_callback_records_fail(self):
        groups = valid_groups()
        groups['on']['requests'][0]['generated_tokens'] += 1
        self.assertTrue(any('token count' in x for x in compare(groups)))
        groups = valid_groups()
        for mode in groups: groups[mode]['requests'] = groups[mode]['requests'][:4]
        self.assertTrue(any('incomplete' in x for x in compare(groups)))

    def test_pressure_alone_does_not_prove_eviction(self):
        groups = valid_groups()
        groups['on']['eviction']['observed_miss_after_hit'] = False
        self.assertEqual(compare(groups), [])
        self.assertTrue(any('eviction not observed' in x for x in compare(groups, True)))

    def test_missing_concurrency_and_cancel_evidence_fail(self):
        groups = valid_groups()
        groups['on']['requested_concurrency'] = [1, 2, 4]
        groups['on']['cancellation']['passed'] = False
        failures = compare(groups)
        self.assertTrue(any('c4 request matrix incomplete' in x for x in failures))
        self.assertTrue(any('cancellation/recovery' in x for x in failures))

    def test_cache_pressure_errors_cannot_pass_on_early_parity(self):
        groups = valid_groups()
        groups['on']['churn_requests'] = [{'label': 'churn42', 'terminal_events': 1,
            'token_ids': [], 'generated_tokens': 0, 'finish_reason': 3}]
        self.assertTrue(any('cache pressure request failed' in x for x in compare(groups)))

    def test_restore_hit_and_skipped_tokens_are_each_required(self):
        for field in ('hits', 'restores', 'skipped_tokens'):
            groups = valid_groups()
            groups['on']['prefix_evidence'][field] = 0
            self.assertTrue(any('actual snapshot' in x for x in compare(groups)))

    def test_log_evidence_uses_actual_request_id(self):
        group = valid_groups()['on']
        attach_log_evidence(group, '[Qwen35DecodeSummary] req=4 prefix_cache_hit=1 '
            'prefix_cache_skipped_tokens=64 hybrid_ssm_snapshot_restore_applied=1\n'
            '[Qwen35DecodeSummary] req=11 prefix_cache_hit=0 prefix_cache_skipped_tokens=0 '
            'hybrid_ssm_snapshot_restore_applied=0\n')
        self.assertTrue(group['eviction']['observed_miss_after_hit'])
        self.assertEqual(group['prefix_evidence'], {'hits': 1, 'restores': 1, 'skipped_tokens': [64]})
        self.assertEqual(prefix_evidence('prefix_cache_hit=10'), {'hits': 0, 'restores': 0, 'skipped_tokens': []})

    def test_same_prompt_across_concurrency_must_match(self):
        groups = valid_groups()
        for group in groups.values():
            group['requests'][4]['prompt'] = group['requests'][0]['prompt']
            group['requests'][4]['token_ids'] = [4, 99]
        self.assertTrue(any('scheduling/concurrency' in x for x in compare(groups)))


if __name__ == '__main__': unittest.main()
