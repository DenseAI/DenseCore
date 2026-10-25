#!/usr/bin/env python3
"""Run local native A/B diagnostics without asserting quality or economic success."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from run_systemone_load import Native, run
from systemone_economics import digest, load_workload, require


class SerialBatch:
    """Explicit baseline fallback: independent single calls, never native batching."""
    def __init__(self, native):
        self.native = native

    def __call__(self, row):
        return self.native(row)

    def call_batch(self, rows):
        return [self.native(row) for row in rows]


def load_probe_workload(path, case_names=None):
    obj = json.loads(Path(path).read_text())
    if obj.get('schema') == 'systemone-workload-v1':
        rows = load_workload(path)['requests']
    else:
        rows = []
        for case in obj['cases']:
            seq = {k: case[k] for k in ('token_ids', 'markers', 'question_type')}
            seq.update({k: case[k] for k in ('reference_logits', 'reference_act_logits', 'reference_probabilities', 'temperature') if k in case})
            rows.append({'id': case['name'], 'sequences': [seq], '_useful_tokens': len(case['token_ids']),
                         '_compute_tokens': len(case['token_ids']), '_useful_accounting': 'unavailable_fixture_has_no_logical_token_keys'})
    if case_names:
        rows = [row for row in rows if row['id'] in case_names]
        require(set(case_names) == {row['id'] for row in rows}, 'unknown probe case')
    require(bool(rows), 'empty probe workload')
    return rows


def parity_diagnostic(rows, records):
    by_id = {row['id']: row for row in rows}
    option_error, probability_error, action_error = [], [], []
    checked, missing = set(), set()
    for rec in records:
        if not rec['success'] or rec['workload_id'] in checked:
            continue
        row = by_id[rec['workload_id']]
        checked.add(row['id'])
        for seq, out in zip(row['sequences'], rec['response']):
            if 'reference_logits' not in seq:
                missing.add(row['id'])
                continue
            require(len(out['logits']) == len(seq['reference_logits']), 'reference logit shape')
            option_error += [abs(a-b) for a, b in zip(out['logits'], seq['reference_logits'])]
            if 'reference_act_logits' in seq:
                action_error += [abs(a-b) for a, b in zip(out['act_logits'], seq['reference_act_logits'])]
            if 'reference_probabilities' in seq and 'temperature' in seq:
                values = [v/seq['temperature'] for v in out['logits']]
                weights = [math.exp(v-max(values)) for v in values]
                probabilities = [v/sum(weights) for v in weights]
                probability_error += [abs(a-b) for a, b in zip(probabilities, seq['reference_probabilities'])]
    return {'status': 'diagnostic_only', 'checked_cases': sorted(checked),
            'missing_reference_cases': sorted(missing | (set(by_id)-checked)),
            'option_logit_max_absolute_error': max(option_error, default=None),
            'probability_max_absolute_error': max(probability_error, default=None),
            'action_logit_max_absolute_error': max(action_error, default=None),
            'action_gate': 'unverified_separate_reference_criteria_required',
            'quality_verdict': 'unverified'}


def csv_ints(value):
    out = [int(v) for v in value.split(',')]
    require(bool(out) and all(v > 0 for v in out), 'positive integer list required')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-library', type=Path, required=True)
    p.add_argument('--candidate-library', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--workload', type=Path, required=True, help='System One workload or independent parity fixture JSON')
    p.add_argument('--output', type=Path, required=True, help='Artifact directory')
    p.add_argument('--threads', type=csv_ints, default=[1])
    p.add_argument('--batches', type=csv_ints, default=[1])
    p.add_argument('--cases', help='Comma-separated case IDs (default all)')
    p.add_argument('--duration', type=float, default=1)
    p.add_argument('--cycles', type=int, default=1)
    p.add_argument('--legs', choices=['ABBA', 'B'], default='ABBA', help='B is candidate-only diagnostic exploration, never an A/B estimate')
    p.add_argument('--precision', choices=['fp32', 'q8_0'], default='fp32', help='Candidate precision; baseline remains fp32')
    p.add_argument('--warmup', type=int, default=1, help='Warmup rounds; each round submits batch-size requests')
    args = p.parse_args()
    require(math.isfinite(args.duration) and args.duration > 0 and args.cycles > 0 and args.warmup >= 0, 'invalid probe controls')
    require(all(v <= 64 for v in args.batches), 'batch sizes 1..64')
    args.output.mkdir(parents=True, exist_ok=True)
    rows = load_probe_workload(args.workload, args.cases.split(',') if args.cases else None)
    model_hash, workload_hash = digest(args.model), digest(args.workload)
    libraries = {'A': args.baseline_library, 'B': args.candidate_library}
    hashes = {k: digest(v) for k, v in libraries.items()}
    results = []
    for threads in args.threads:
        for batch in args.batches:
            require(max(len(row['sequences']) for row in rows)*batch <= 64, 'batch exceeds 64 encoder sequences')
            for cycle in range(args.cycles):
                for leg, variant in enumerate(args.legs):
                    prefix = args.output/f't{threads}-b{batch}-c{cycle}-l{leg}-{variant}'
                    native = Native(libraries[variant], args.model, threads, args.precision if variant == 'B' else 'fp32')
                    dependencies = native.runtime_dependencies_sha256()
                    config = {'runtime_dependencies_sha256': dependencies, 'schema': 'encoder-probe-v1', 'status': 'diagnostic_only', 'variant': variant,
                              'model_sha256': model_hash, 'workload_sha256': workload_hash,
                              'implementation_sha256': hashes[variant], 'threads': threads, 'batch_size': batch,
                              'cycle': cycle, 'leg': leg, 'duration_s': args.duration, 'legs': args.legs,
                              'precision': args.precision if variant == 'B' else 'fp32',
                              'quality_verdict': 'unverified', 'economics_verdict': 'unavailable',
                              'selected_cases': [row['id'] for row in rows]}
                    config_path = Path(str(prefix)+'.config.json')
                    config_path.write_text(json.dumps(config, indent=2)+'\n')
                    native_batch = batch > 1 and hasattr(native.lib, 'DenseCoreDecisionPredictBatch')
                    call = native if native_batch or batch == 1 else SerialBatch(native)
                    try:
                        for i in range(args.warmup):
                            warm_rows = [rows[(i*batch+j) % len(rows)] for j in range(batch)]
                            call(warm_rows[0]) if batch == 1 else call.call_batch(warm_rows)
                        metadata, records = run(call, rows, args.duration, 1, prefix, digest(config_path),
                                                'full_model', threads, args.warmup, batch, runtime_dependencies_sha256=dependencies)
                    finally:
                        native.close()
                    success = [rec for rec in records if rec['success']]
                    latencies = sorted(rec['end_s']-rec['start_s'] for rec in success)
                    percentile = lambda q: latencies[min(len(latencies)-1, math.ceil(q*len(latencies))-1)] if latencies else None
                    useful_known = all('_useful_accounting' not in row for row in rows)
                    useful_tokens = sum(rec['useful_tokens'] for rec in success) if useful_known else None
                    result = {**config, 'run_id': metadata['run_id'], 'elapsed_s': metadata['elapsed_s'],
                              'records_sha256': metadata['records_sha256'], 'requests': len(records),
                              'successful_requests': len(success), 'batch_execution': 'native_batch' if native_batch else 'serial_single',
                              'useful_input_tokens': useful_tokens, 'useful_tokens': useful_tokens,
                              'useful_tokens_per_s': useful_tokens/metadata['elapsed_s'] if useful_known else None,
                              'useful_accounting': 'logical_token_keys' if useful_known else 'unavailable_fixture_has_no_logical_token_keys',
                              'latency_s': {'p50': percentile(.5), 'p95': percentile(.95), 'p99': percentile(.99)},
                              'memory': metadata['memory'],
                              'compute_tokens_per_s': sum(rec['compute_tokens'] for rec in success)/metadata['elapsed_s'],
                              'requests_per_s': len(success)/metadata['elapsed_s'],
                              'parity': parity_diagnostic(rows, records), 'prefix': str(prefix)}
                    Path(str(prefix)+'.probe.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
                    results.append(result)
                    print(json.dumps({k: result[k] for k in ('variant', 'threads', 'batch_size', 'cycle', 'leg', 'requests_per_s', 'batch_execution')}), flush=True)
    (args.output/'probe-summary.json').write_text(json.dumps({'status': 'diagnostic_only', 'runs': results}, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
