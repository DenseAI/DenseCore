import copy
import datetime as dt
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_encoder_abba import check, lower95
from systemone_economics import digest
import test_systemone_economics as accounting_fixtures


class ABBAEvidence(unittest.TestCase):
    def setUp(self):
        f = accounting_fixtures.EncoderAccounting()
        f.setUp()
        self.addCleanup(f.doCleanups)
        self.f = f
        self.cycles = []
        origin = dt.datetime(2026, 10, 16, tzinfo=dt.timezone.utc)
        for cycle in range(5):
            refs = []
            for col, variant in enumerate('ABBA'):
                name = f'{cycle}-{col}'
                manifest = {k: v for k, v in f.manifest.items() if k != '_sha256'}
                manifest['implementation_sha256'] = ('a' if variant == 'A' else 'b')*64
                quality = json.loads(f.quality.read_text())
                quality['implementation_sha256'] = manifest['implementation_sha256']
                evaluation = json.loads(f.evidence.read_text())
                evaluation['implementation_sha256'] = manifest['implementation_sha256']
                ep = f.write(name+'-evidence.json', evaluation)
                quality['evaluation_artifact'] = ep.name
                quality['evaluation_artifact_sha256'] = digest(ep)
                qp = f.write(name+'-quality.json', quality)
                manifest['quality_sha256'] = digest(qp)
                mp = f.write(name+'-manifest.json', manifest)
                start = origin + dt.timedelta(seconds=(cycle*4+col)*130)
                records = copy.deepcopy(f.records)
                for rec in records:
                    rec['end_s'] += (cycle*4+col)*.001
                    rec['run_id'] = name
                    rec['response'] = [{'logits': [0.], 'act_logits': [0., 0.]}]*2
                records[1]['success'], records[1]['error'] = True, None
                # B performs the same successful work in half the time, but
                # both variants still use >=120s of real submission window.
                if variant == 'B':
                    records += [{**r, 'attempt_id': r['attempt_id']+2} for r in records]
                meta = {**f.meta, 'manifest_sha256': digest(mp), 'elapsed_s': 121.,
                        'submission_stop_s': 120., 'record_count': len(records),
                        'run_id': name, 'warmup_requests': 2, 'threads': 4, 'concurrency': 1, 'batch_size': 1, 'host_id': 'host',
                        'started_at': start.isoformat(), 'finished_at': (start+dt.timedelta(seconds=121)).isoformat()}
                rp = f.root/(name+'.jsonl')
                rp.write_text(''.join(json.dumps(r)+'\n' for r in records))
                meta['records_sha256'] = digest(rp)
                refs.append({'manifest': mp.name, 'quality': qp.name, 'workload': f.workload_path.name,
                             'pricing': f.price_path.name, 'records': rp.name,
                             'metadata': f.write(name+'-metadata.json', meta).name})
            self.cycles.append(refs)
        self.plan = f.write('abba.json', {'cycles': self.cycles})

    def test_recomputes_raw_cycles_and_keeps_gpu_unavailable(self):
        out = check(self.plan)
        self.assertEqual(out['baseline_improvement'], 'pass')
        self.assertEqual(out['candidate_over_baseline_lower95'], 2)
        self.assertEqual(out['public_gpu_2x'], 'unavailable')

    def test_reject_reused_runs(self):
        self.cycles[1][0] = self.cycles[0][0]
        self.f.write('abba.json', {'cycles': self.cycles})
        with self.assertRaisesRegex(ValueError, 'reused'):
            check(self.plan)

    def test_reject_copied_raw_content_at_new_path(self):
        first = self.f.root/self.cycles[0][0]['records']
        later = self.f.root/self.cycles[1][0]['records']
        later.write_bytes(first.read_bytes())
        meta_path = self.f.root/self.cycles[1][0]['metadata']
        meta = json.loads(meta_path.read_text())
        meta['records_sha256'] = digest(later)
        self.f.write(meta_path.name, meta)
        with self.assertRaisesRegex(ValueError, 'reused raw content'):
            check(self.plan)

    def test_reject_edited_raw_records(self):
        first = self.f.root/self.cycles[0][0]['records']
        first.write_text(first.read_text()+'\n')
        with self.assertRaisesRegex(ValueError, 'raw records hash'):
            check(self.plan)

    def test_reject_short_or_overlapping_runs_and_changed_host(self):
        path = self.f.root/self.cycles[0][1]['metadata']
        original = json.loads(path.read_text())
        for field, value, error in [('submission_stop_s', 119, '120 seconds'),
                                    ('host_id', 'different', 'provenance'),
                                    ('started_at', '2026-10-16T00:00:00+00:00', 'overlapping')]:
            self.f.write(path.name, {**original, field: value})
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, error):
                check(self.plan)
        self.f.write(path.name, original)

    def test_interval_does_not_promote_noisy_mean(self):
        self.assertLess(lower95([.8, .9, 1, 1.2, 1.6]), 1)

    def test_reject_changed_dependency_with_same_main_library(self):
        ref = self.cycles[1][0]
        manifest_path = self.f.root/ref['manifest']
        manifest = json.loads(manifest_path.read_text())
        dependencies = {**manifest['runtime_dependencies_sha256']}
        dependencies[next(iter(dependencies))] = 'd'*64
        manifest['runtime_dependencies_sha256'] = dependencies
        quality_path = self.f.root/ref['quality']
        quality = json.loads(quality_path.read_text())
        evidence_path = quality_path.parent/quality['evaluation_artifact']
        evidence = json.loads(evidence_path.read_text())
        evidence['runtime_dependencies_sha256'] = dependencies
        self.f.write(evidence_path.name, evidence)
        quality.update(runtime_dependencies_sha256=dependencies,
                       evaluation_artifact_sha256=digest(evidence_path))
        self.f.write(quality_path.name, quality)
        manifest['quality_sha256'] = digest(quality_path)
        self.f.write(manifest_path.name, manifest)
        metadata_path = self.f.root/ref['metadata']
        metadata = json.loads(metadata_path.read_text())
        metadata.update(runtime_dependencies_sha256=dependencies,
                        manifest_sha256=digest(manifest_path))
        self.f.write(metadata_path.name, metadata)
        with self.assertRaisesRegex(ValueError, 'build changed'):
            check(self.plan)


if __name__ == '__main__':
    unittest.main()
