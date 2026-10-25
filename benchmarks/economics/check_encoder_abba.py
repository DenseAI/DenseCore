#!/usr/bin/env python3
"""Recompute five paired ABBA cycles from bound encoder raw artifacts.

Input: {"cycles": [[A, B, B, A], ...]} with five cycles. Each run is an
object of paths: manifest, workload, quality, pricing, records, metadata.
Paths are relative to this file's input manifest. No external GPU uncertainty
is inferred. A GPU comparison needs a separately qualified comparable baseline.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import statistics
from pathlib import Path

from systemone_economics import digest, load_workload, require, summarize, validate_artifacts


def lower95(values):
    """Lower endpoint of a two-sided Student t interval, n=5, df=4."""
    require(len(values) == 5, 'five independent cycle estimates required')
    return statistics.mean(values) - 2.7764451051977987 * statistics.stdev(values) / math.sqrt(5)


def check(path):
    path = Path(path)
    plan = json.loads(path.read_text())
    cycles = plan['cycles']
    require(len(cycles) == 5 and all(len(c) == 4 for c in cycles), 'five ABBA cycles required')
    seen, seen_content, identities, provenance = set(), set(), {}, None
    previous_end = None
    precisions = {}
    ratios, candidate_rates = [], []
    for cycle in cycles:
        scores = []
        for variant, ref in zip('ABBA', cycle):
            paths = {k: (path.parent / ref[k]).resolve() for k in
                     ('manifest', 'workload', 'quality', 'pricing', 'records', 'metadata')}
            if ref.get('server_identity'):
                paths['server_identity'] = (path.parent / ref['server_identity']).resolve()
            require(paths['records'] not in seen, 'reused raw run')
            seen.add(paths['records'])
            raw_hash = digest(paths['records'])
            require(raw_hash not in seen_content, 'reused raw content')
            seen_content.add(raw_hash)
            manifest = json.loads(paths['manifest'].read_text())
            _, price = validate_artifacts(manifest, paths['workload'], paths['quality'], paths['pricing'],
                                          paths.get('server_identity'))
            manifest['_sha256'] = digest(paths['manifest'])
            meta = json.loads(paths['metadata'].read_text())
            require(meta.get('records_sha256') == raw_hash, 'raw records hash mismatch')
            records = [json.loads(r) for r in paths['records'].read_text().splitlines() if r.strip()]
            score = summarize(manifest, load_workload(paths['workload']), meta, records, price)
            require(meta['submission_stop_s'] >= 120 and meta['warmup_requests'] > 0,
                    'final runs require warm-up and at least 120 seconds before drain')
            require(score['successful_requests'] > 0, 'no successful work')
            start, end = (dt.datetime.fromisoformat(meta[k]) for k in ('started_at', 'finished_at'))
            require(start.tzinfo is not None and end.tzinfo is not None and end > start,
                    'timezone-aware increasing wall timestamps required')
            require(previous_end is None or start >= previous_end, 'overlapping or out-of-order runs')
            require(abs((end-start).total_seconds()-meta['elapsed_s']) < 1,
                    'wall and measured duration mismatch')
            previous_end = end
            identity = {k: manifest[k] for k in ('implementation_sha256', 'runtime_dependencies_sha256')}
            if manifest.get('server_identity_sha256'):
                identity['server_identity_sha256'] = manifest['server_identity_sha256']
            require(precisions.setdefault(variant, manifest['precision']) == manifest['precision'],
                    'precision changed within variant')
            require(identities.setdefault(variant, identity) == identity, 'build changed within variant')
            common = {k: manifest[k] for k in ('model_sha256', 'tokenizer_sha256', 'workload_sha256',
                                               'boundary', 'pricing_sha256')}
            common.update({k: meta[k] for k in ('host_id', 'threads', 'concurrency', 'batch_size')})
            require(bool(common['host_id']), 'missing host')
            provenance = common if provenance is None else provenance
            require(common == provenance, 'ABBA provenance mismatch')
            scores.append(score['tokens_per_s'])
        require(identities['A'] != identities['B'], 'baseline and candidate builds must differ')
        a, b = statistics.mean((scores[0], scores[3])), statistics.mean(scores[1:3])
        require(a > 0, 'zero baseline throughput')
        ratios.append(b/a)
        candidate_rates.append(b)
    lo = lower95(ratios)
    return {'status': 'measured', 'cycles': 5, 'provenance': provenance, 'variant_precisions': precisions,
            'variant_implementations': identities,
            'cycle_candidate_over_baseline': ratios,
            'candidate_over_baseline_mean': statistics.mean(ratios),
            'candidate_over_baseline_lower95': lo,
            'candidate_tokens_per_s_lower95': max(0, lower95(candidate_rates)),
            'baseline_improvement': 'pass' if lo > 1 else 'not_demonstrated',
            'public_gpu_2x': 'unavailable',
            'limitations': ['Student t interval assumes independent cycle estimates.',
                            'No comparable full-instance GPU price and quality artifact supplied.']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.write_text(json.dumps(check(args.manifest), indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
