import argparse
import copy
from pathlib import Path
import tempfile
import unittest

from check_mtp_parity import digest
from qualify_prefix import combined_gate, commands, verify_files
from test_check_prefix_parity import valid_groups


def phases():
    result = {}
    for name, widths in (('main', [1, 2, 4]), ('pressure', [1])):
        groups = valid_groups()
        for group in groups.values():
            group['requested_concurrency'] = widths
            group['runtime_plan'] = {'effective_context': 2048, 'effective_concurrency': max(widths)}
            row = group['requests'][0]
            group['requests'].extend(dict(row, label=f'c{width}-{i}', prompt=f'c{width}-{i}')
                                     for width in widths for i in range(width))
            group['churn_requests'] = [dict(row, label=f'churn{i}') for i in range(8)]
            if name == 'pressure':
                group['near_full_pressure'] = {'prompt_tokens': 2027, 'requested_max_tokens': 1,
                                              'effective_context': 2048}
                group['requests'].append(dict(row, label='near-full-pressure', prompt='cold',
                                              token_ids=[4], generated_tokens=1, finish_reason=2))
        result[name] = {'passed': True, 'model_sha256': 'model', 'library_sha256': 'library',
                        'native': groups, 'configuration': {'context': 2048, 'concurrency': widths,
                            'churn': 8, 'near_full_pressure': name == 'pressure'}}
    result['main'].update(server_sha256='server', http_final_answer_verified=True)
    rows = [{'turn': turn, 'response': {'choices': [{'message': {'content': city}, 'finish_reason': 'stop'}],
             'usage': {'completion_tokens': 2}}} for turn, city in enumerate(('Paris', 'Tokyo', 'Berlin', 'Rome', 'Madrid'), 1)]
    result['main']['http'] = {'off': copy.deepcopy(rows), 'on': copy.deepcopy(rows)}
    return result


def gate(data):
    return combined_gate(data, 'model', 'library', 'server', 2048)


class CombinedPrefixGateTest(unittest.TestCase):
    def test_requires_both_phases(self):
        good = phases()
        self.assertEqual(gate(good), [])
        for missing in ('main', 'pressure'):
            data = copy.deepcopy(good)
            del data[missing]
            self.assertTrue(gate(data))

    def test_failed_phase_never_becomes_combined_pass(self):
        data = phases()
        data['pressure']['passed'] = False
        self.assertTrue(gate(data))

    def test_rechecks_actual_snapshot_eviction(self):
        data = phases()
        data['pressure']['native']['on']['eviction']['hybrid_snapshot_entries_removed_delta'] = 0
        self.assertTrue(any('eviction not observed' in failure for failure in gate(data)))

    def test_main_need_not_evict_when_pressure_does(self):
        data = phases()
        for group in data['main']['native'].values():
            group['eviction'] = {}
        self.assertEqual(gate(data), [])

    def test_rechecks_near_full_context_and_one_output(self):
        for field, value in (('prompt_tokens', 500), ('requested_max_tokens', 96)):
            data = phases()
            data['pressure']['native']['on']['near_full_pressure'][field] = value
            self.assertTrue(any('bounded cold context' in failure for failure in gate(data)))

    def test_rechecks_actual_cold_output_ids(self):
        data = phases()
        data['pressure']['native']['on']['requests'][-1]['token_ids'] = [99]
        self.assertTrue(any('off/on exact' in failure for failure in gate(data)))

    def test_rechecks_capacity_and_all_concurrency_rows(self):
        data = phases()
        data['main']['native']['on']['runtime_plan']['effective_concurrency'] = 1
        self.assertTrue(gate(data))
        data = phases()
        data['main']['native']['on']['requests'] = [row for row in data['main']['native']['on']['requests']
                                                   if row['label'] != 'c4-3']
        self.assertTrue(gate(data))

    def test_incomplete_churn_cannot_claim_success(self):
        data = phases()
        data['main']['native']['on']['churn_requests'].pop()
        self.assertTrue(gate(data))

    def test_reasoning_only_and_truncated_http_fail(self):
        for change in ({'message': {'content': '', 'reasoning_content': 'Paris'}}, {'finish_reason': 'length'}):
            data = phases()
            data['main']['http']['on'][0]['response']['choices'][0].update(change)
            self.assertTrue(any('HTTP on/turn1' in failure for failure in gate(data)))

    def test_mismatched_model_or_library_is_rejected(self):
        for field in ('model_sha256', 'library_sha256'):
            data = phases()
            data['pressure'][field] = 'other'
            self.assertTrue(gate(data))

    def test_commands_keep_separate_pressure_gate(self):
        args = argparse.Namespace(library=Path('/lib'), server=Path('/server'), model=Path('/model'),
                                  out=Path('/out'), threads=4, context=2048, port=18090)
        plan = commands(args)
        self.assertIn('--server', plan['main'])
        self.assertNotIn('--require-eviction', plan['main'])
        self.assertNotIn('--server', plan['pressure'])
        self.assertIn('--near-full-pressure', plan['pressure'])
        self.assertIn('--require-eviction', plan['pressure'])
        for command in plan.values():
            self.assertEqual(command[command.index('--churn') + 1], '8')
            self.assertNotIn('--build-lock', command)

    def test_candidate_dependency_changes_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'dependency'
            path.write_bytes(b'original')
            artifact = {'files': {str(path): digest(path)}}
            verify_files(artifact)
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                verify_files(artifact)


if __name__ == '__main__':
    unittest.main()
