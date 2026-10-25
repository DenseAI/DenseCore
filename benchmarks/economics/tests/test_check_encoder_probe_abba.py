import datetime as dt
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_encoder_probe_abba import check
from systemone_economics import digest


class ProbeABBA(unittest.TestCase):
    def write(self, path, value):
        path = self.root/path
        path.write_text(json.dumps(value))
        return path

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.workload = self.write('workload.json', {'schema': 'systemone-workload-v1', 'requests': [
            {'id': 'case', 'truncated': False, 'request': {'questions': {'q': {}}},
             'sequences': [{'token_ids': [3, 5], 'token_keys': [None, 'state:0'],
                            'markers': [0], 'question_type': 0}]}]})
        self.runs = []
        origin = dt.datetime(2026, 10, 16, tzinfo=dt.timezone.utc)
        for cycle in range(5):
            for leg, variant in enumerate('ABBA'):
                name = f'c{cycle}-l{leg}-{variant}'
                dependencies = {name: '9'*64 for name in ('libggml.so', 'libggml-cpu.so', 'libggml-base.so')}
                config = {'runtime_dependencies_sha256': dependencies, 'schema': 'encoder-probe-v1', 'status': 'diagnostic_only',
                    'variant': variant, 'model_sha256': 'a'*64, 'workload_sha256': digest(self.workload),
                    'implementation_sha256': ('b' if variant == 'A' else 'c')*64,
                    'threads': 4, 'batch_size': 1, 'cycle': cycle, 'leg': leg, 'legs': 'ABBA',
                    'duration_s': 120, 'precision': 'fp32', 'selected_cases': ['case']}
                cp = self.write(name+'.config.json', config)
                count = 1 if variant == 'A' else 2
                records = [{'run_id': name, 'attempt_id': i, 'batch_id': i, 'batch_size': 1,
                    'workload_id': 'case', 'start_s': i, 'end_s': 120.5, 'useful_tokens': 1,
                    'compute_tokens': 2, 'success': True, 'error': None, 'unresolved_backend_work': False,
                    'response': [{'logits': [0.], 'act_logits': [0., 0.]}]} for i in range(count)]
                rp = self.root/(name+'.records.jsonl')
                rp.write_text(''.join(json.dumps(row)+'\n' for row in records))
                start = origin + dt.timedelta(seconds=(cycle*4+leg)*130)
                self.write(name+'.metadata.json', {'runtime_dependencies_sha256': dependencies, 'runtime_dependencies_stable': True, 'runtime_dependencies_origin': 'observed_process_maps', 'run_id': name, 'records_sha256': digest(rp),
                    'manifest_sha256': digest(cp), 'boundary': 'full_model', 'threads': 4,
                    'batch_size': 1, 'concurrency': 1, 'warmup_requests': 2, 'submission_stop_s': 120,
                    'elapsed_s': 121, 'record_count': count, 'host_id': 'same-host',
                    'started_at': start.isoformat(), 'finished_at': (start+dt.timedelta(seconds=121)).isoformat()})
                self.runs.append({**config, 'prefix': f'old/relative/output/{name}'})
        self.summary = self.write('probe-summary.json', {'status': 'diagnostic_only', 'runs': self.runs})

    def run_check(self):
        return check(self.summary, self.workload)

    def edit_metadata(self, index, **changes):
        p = self.root/(Path(self.runs[index]['prefix']).name+'.metadata.json')
        self.write(p.name, {**json.loads(p.read_text()), **changes})

    def test_recomputes_raw_rates_and_full_drain_without_quality_or_price(self):
        out = self.run_check()
        self.assertEqual(out['quality_verdict'], 'unverified')
        self.assertEqual(out['economics_verdict'], 'unavailable')
        result = out['configurations'][0]
        self.assertEqual(result['cycles'], 5)
        self.assertEqual(result['metrics']['useful_tokens_per_s']['candidate_over_baseline_lower95'], 2)
        self.assertAlmostEqual(result['metrics']['compute_tokens_per_s']['candidate_rate_lower95'], 4/121)
        self.assertAlmostEqual(result['metrics']['decisions_per_s']['candidate_rate_lower95'], 2/121)
        self.assertEqual(result['runs'][0]['drain_s'], 1)

    def test_reject_short_warmup_overlap_and_changed_controls(self):
        for changes, message in [({'warmup_requests': 0}, 'warm-up'),
                                  ({'submission_stop_s': 119}, '120-second'),
                                  ({'threads': 8}, 'control'),
                                  ({'started_at': '2026-10-16T00:00:00+00:00'}, 'wall/measured'),
                                  ({'host_id': 'other'}, 'provenance')]:
            with self.subTest(changes=changes):
                path = self.root/(Path(self.runs[1]['prefix']).name+'.metadata.json')
                original = path.read_text()
                self.edit_metadata(1, **changes)
                with self.assertRaisesRegex(ValueError, message): self.run_check()
                path.write_text(original)

    def test_reject_reused_run_id_and_edited_raw_bytes(self):
        self.edit_metadata(1, run_id=self.runs[0]['prefix'].split('/')[-1])
        with self.assertRaisesRegex(ValueError, 'reused'): self.run_check()
        self.edit_metadata(1, run_id=self.runs[1]['prefix'].split('/')[-1])
        p = self.root/(Path(self.runs[0]['prefix']).name+'.records.jsonl')
        p.write_text(p.read_text()+'\n')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'): self.run_check()

    def test_reject_copied_records_even_with_updated_digest(self):
        first, second = (self.root/(Path(self.runs[i]['prefix']).name+'.records.jsonl') for i in (0, 4))
        second.write_bytes(first.read_bytes())
        self.edit_metadata(4, records_sha256=digest(second))
        with self.assertRaisesRegex(ValueError, 'reused raw content'): self.run_check()

    def test_reject_changed_dependency_same_main_even_with_rehashed_config(self):
        index = 1
        listed = self.runs[index]
        prefix = self.root/Path(listed['prefix']).name
        cp, mp = Path(str(prefix)+'.config.json'), Path(str(prefix)+'.metadata.json')
        changed = dict(listed['runtime_dependencies_sha256'], **{'libggml-cpu.so': '8'*64})
        listed['runtime_dependencies_sha256'] = changed
        self.write(cp.name, {**json.loads(cp.read_text()), 'runtime_dependencies_sha256': changed})
        self.edit_metadata(index, runtime_dependencies_sha256=changed, manifest_sha256=digest(cp))
        self.write(self.summary.name, {'status': 'diagnostic_only', 'runs': self.runs})
        with self.assertRaisesRegex(ValueError, 'variant implementation/precision/runtime dependencies changed'):
            self.run_check()

    def test_reject_missing_or_mismatched_measured_dependency_map(self):
        self.edit_metadata(0, runtime_dependencies_sha256=None)
        with self.assertRaisesRegex(ValueError, 'measured runtime dependencies mismatch'):
            self.run_check()

    def upgrade_to_monotonic_v2(self):
        for index, listed in enumerate(self.runs):
            begin = 100_000_000_000_000 + index*130_000_000_000
            self.edit_metadata(index, schema='systemone-run-v2',
                monotonic_started_ns=begin, monotonic_finished_ns=begin+121_000_000_000,
                boot_id='11111111-2222-3333-4444-555555555555',
                monotonic_clock={'name': 'time.monotonic_ns', 'implementation': 'clock_gettime(CLOCK_MONOTONIC)',
                                 'monotonic': True, 'adjustable': False})

    def test_v2_monotonic_accepts_observed_wall_jumps_and_reports_them(self):
        self.upgrade_to_monotonic_v2()
        for index, listed in enumerate(self.runs):
            path = self.root/(Path(listed['prefix']).name+'.metadata.json')
            metadata = json.loads(path.read_text())
            start_shift = sum(step for at, step in ((5, -13.612807), (15, -13.651839)) if index > at)
            end_shift = sum(step for at, step in ((5, -13.612807), (15, -13.651839)) if index >= at)
            self.edit_metadata(index,
                started_at=(dt.datetime.fromisoformat(metadata['started_at'])+dt.timedelta(seconds=start_shift)).isoformat(),
                finished_at=(dt.datetime.fromisoformat(metadata['finished_at'])+dt.timedelta(seconds=end_shift)).isoformat())
        result = self.run_check()['configurations'][0]
        clock = result['clock_evidence']
        self.assertEqual(clock['protocol'], 'observed_monotonic_v2')
        self.assertEqual(clock['wall_clock_status'], 'discontinuity_observed')
        warnings = [w for w in clock['warnings'] if w['kind'] == 'wall_clock_duration_discontinuity']
        self.assertEqual(len(warnings), 2)
        self.assertAlmostEqual(warnings[0]['wall_minus_monotonic_s'], -13.612807)
        self.assertEqual(result['metrics']['useful_tokens_per_s']['candidate_over_baseline_lower95'], 2)

    def test_v2_uses_monotonic_order_and_reports_between_run_wall_step(self):
        self.upgrade_to_monotonic_v2()
        for index in range(1, len(self.runs)):
            path = self.root/(Path(self.runs[index]['prefix']).name+'.metadata.json')
            metadata = json.loads(path.read_text())
            self.edit_metadata(index,
                started_at=(dt.datetime.fromisoformat(metadata['started_at'])-dt.timedelta(seconds=150)).isoformat(),
                finished_at=(dt.datetime.fromisoformat(metadata['finished_at'])-dt.timedelta(seconds=150)).isoformat())
        warnings = self.run_check()['configurations'][0]['clock_evidence']['warnings']
        self.assertTrue(any(w['kind'] == 'wall_clock_order_discontinuity' for w in warnings))
        # Realtime order can look valid while the boot-bound measured intervals overlap.
        begin = 100_000_000_000_000+120_000_000_000
        self.edit_metadata(1, monotonic_started_ns=begin, monotonic_finished_ns=begin+121_000_000_000)
        with self.assertRaisesRegex(ValueError, 'absolute monotonic runs'):
            self.run_check()

    def test_v2_reports_small_between_run_clock_step_without_wall_overlap(self):
        self.upgrade_to_monotonic_v2()
        for index in range(1, len(self.runs)):
            path = self.root/(Path(self.runs[index]['prefix']).name+'.metadata.json')
            metadata = json.loads(path.read_text())
            self.edit_metadata(index,
                started_at=(dt.datetime.fromisoformat(metadata['started_at'])-dt.timedelta(seconds=5)).isoformat(),
                finished_at=(dt.datetime.fromisoformat(metadata['finished_at'])-dt.timedelta(seconds=5)).isoformat())
        clock = self.run_check()['configurations'][0]['clock_evidence']
        self.assertEqual(clock['wall_clock_status'], 'discontinuity_observed')
        gaps = [warning for warning in clock['warnings'] if warning['kind'] == 'wall_clock_gap_discontinuity']
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]['wall_gap_s'], 4)
        self.assertEqual(gaps[0]['monotonic_gap_s'], 9)
        self.assertEqual(gaps[0]['wall_minus_monotonic_gap_s'], -5)

    def test_v2_rejects_boot_change_duration_tamper_and_partial_evidence(self):
        self.upgrade_to_monotonic_v2()
        path = self.root/(Path(self.runs[1]['prefix']).name+'.metadata.json')
        original = path.read_text()
        for changes, message in [
                ({'boot_id': '99999999-2222-3333-4444-555555555555'}, 'boot identity changed'),
                ({'monotonic_finished_ns': 100_000_000_000_000+130_000_000_000+122_000_000_000}, 'monotonic/measured'),
                ({'monotonic_clock': {'name': 'time.monotonic_ns', 'implementation': 'other', 'monotonic': True, 'adjustable': True}}, 'nonadjustable')]:
            with self.subTest(changes=changes):
                self.edit_metadata(1, **changes)
                with self.assertRaisesRegex(ValueError, message): self.run_check()
                path.write_text(original)
        metadata = json.loads(original)
        del metadata['monotonic_finished_ns']
        self.write(path.name, metadata)
        with self.assertRaisesRegex(ValueError, 'complete v2 monotonic'):
            self.run_check()

    def test_legacy_wall_jump_still_rejects_without_monotonic_evidence(self):
        path = self.root/(Path(self.runs[1]['prefix']).name+'.metadata.json')
        metadata = json.loads(path.read_text())
        self.edit_metadata(1, finished_at=(dt.datetime.fromisoformat(metadata['finished_at'])-dt.timedelta(seconds=13.612807)).isoformat())
        with self.assertRaisesRegex(ValueError, 'wall/measured interval mismatch'):
            self.run_check()

    def test_reject_missing_cycle(self):
        self.write(self.summary.name, {'status': 'diagnostic_only', 'runs': self.runs[:-4]})
        with self.assertRaisesRegex(ValueError, 'five ABBA'): self.run_check()

    def test_failures_do_not_enter_rate_numerator(self):
        for listed in self.runs:
            if listed['variant'] != 'B': continue
            prefix = self.root/Path(listed['prefix']).name
            p = Path(str(prefix)+'.records.jsonl')
            records = [json.loads(line) for line in p.read_text().splitlines()]
            records[1].update(success=False, error='native error')
            p.write_text(''.join(json.dumps(row)+'\n' for row in records))
            metadata = Path(str(prefix)+'.metadata.json')
            self.write(metadata.name, {**json.loads(metadata.read_text()), 'records_sha256': digest(p)})
        out = self.run_check()['configurations'][0]
        self.assertEqual(out['metrics']['decisions_per_s']['candidate_over_baseline_lower95'], 1)
        self.assertEqual(out['max_error_rate'], .5)

    def test_legacy_fixture_never_manufactures_useful_tokens(self):
        self.write(self.workload.name, {'cases': [{'name': 'case', 'token_ids': [3, 5],
                                                  'markers': [0], 'question_type': 0}]})
        for listed in self.runs:
            listed['workload_sha256'] = digest(self.workload)
            prefix = self.root/Path(listed['prefix']).name
            cp, mp, rp = (Path(str(prefix)+suffix) for suffix in
                          ('.config.json', '.metadata.json', '.records.jsonl'))
            self.write(cp.name, {**json.loads(cp.read_text()), 'workload_sha256': digest(self.workload)})
            rows = [json.loads(line) for line in rp.read_text().splitlines()]
            for row in rows: row['useful_tokens'] = 2
            rp.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            self.write(mp.name, {**json.loads(mp.read_text()), 'records_sha256': digest(rp),
                                 'manifest_sha256': digest(cp)})
        self.write(self.summary.name, {'status': 'diagnostic_only', 'runs': self.runs})
        metrics = self.run_check()['configurations'][0]['metrics']
        self.assertEqual(metrics['useful_tokens_per_s']['status'], 'unavailable')
        self.assertEqual(metrics['compute_tokens_per_s']['candidate_over_baseline_lower95'], 2)


if __name__ == '__main__': unittest.main()
