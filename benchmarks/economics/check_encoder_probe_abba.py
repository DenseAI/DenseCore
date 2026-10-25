#!/usr/bin/env python3
"""Validate five local probe ABBA cycles without prices or quality claims.

Read probe-summary.json and its adjacent .config/.metadata/.records artifacts.
The original workload is required; optional model/library paths verify binaries.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import re
import statistics

from check_encoder_abba import lower95
from run_encoder_probe import load_probe_workload
from systemone_economics import digest, number, require, validate_native_response, validate_runtime_dependencies


def read(path):
    return json.loads(Path(path).read_text())


def sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def clock_evidence(metadata, duration):
    fields = ('monotonic_started_ns', 'monotonic_finished_ns', 'boot_id', 'monotonic_clock')
    modern = metadata.get('schema') == 'systemone-run-v2' or any(key in metadata for key in fields)
    start, end = (dt.datetime.fromisoformat(metadata[key]) for key in ('started_at', 'finished_at'))
    require(start.tzinfo is not None and end.tzinfo is not None, 'timezone-aware wall interval required')
    wall_seconds = (end-start).total_seconds()
    if not modern:
        # Historical evidence retains the original strict checks. A wall jump
        # cannot be forgiven without independently captured monotonic evidence.
        require(end > start, 'timezone-aware wall interval required')
        require(abs(wall_seconds-duration) < 1, 'wall/measured interval mismatch')
        return {'mode': 'legacy_strict_wall', 'start': start, 'end': end,
                'monotonic_start_ns': None, 'monotonic_end_ns': None,
                'boot_id': None, 'clock': None, 'warnings': []}
    require(metadata.get('schema') == 'systemone-run-v2' and all(key in metadata for key in fields),
            'complete v2 monotonic clock evidence required')
    begin, finish = metadata['monotonic_started_ns'], metadata['monotonic_finished_ns']
    require(type(begin) is int and type(finish) is int and 0 < begin < finish, 'invalid absolute monotonic interval')
    require(abs((finish-begin)/1_000_000_000-duration) < 1e-6, 'monotonic/measured interval mismatch')
    boot = metadata['boot_id']
    require(isinstance(boot, str) and re.fullmatch(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', boot), 'invalid observed boot identity')
    clock = metadata['monotonic_clock']
    require(isinstance(clock, dict) and clock.get('name') == 'time.monotonic_ns' and
            clock.get('monotonic') is True and clock.get('adjustable') is False and
            isinstance(clock.get('implementation'), str) and bool(clock['implementation']),
            'nonadjustable observed monotonic clock required')
    warnings = []
    if abs(wall_seconds-duration) >= 1:
        warnings.append({'kind': 'wall_clock_duration_discontinuity', 'wall_elapsed_s': wall_seconds,
                         'monotonic_elapsed_s': duration, 'wall_minus_monotonic_s': wall_seconds-duration})
    return {'mode': 'observed_monotonic_v2', 'start': start, 'end': end,
            'monotonic_start_ns': begin, 'monotonic_end_ns': finish,
            'boot_id': boot, 'clock': clock, 'warnings': warnings}


def check(summary_path, workload_path, model_path=None, baseline_library=None, candidate_library=None):
    summary_path, workload_path = Path(summary_path), Path(workload_path)
    summary = read(summary_path)
    require(summary.get('status') == 'diagnostic_only', 'probe summary status')
    workload_hash = digest(workload_path)
    source_rows = load_probe_workload(workload_path)
    all_rows = {row['id']: row for row in source_rows}
    require(len(source_rows) == len(all_rows), 'duplicate workload cases')
    groups, seen_paths, seen_hashes, seen_ids = {}, set(), set(), set()
    for listed in summary['runs']:
        # The runner writes every artifact beside its summary. Its recorded
        # prefix may be relative to the invocation directory, not the summary.
        prefix = summary_path.parent / Path(listed['prefix']).name
        require(prefix not in seen_paths, 'reused probe run')
        seen_paths.add(prefix)
        config_path = Path(str(prefix)+'.config.json')
        config = read(config_path)
        metadata = read(str(prefix)+'.metadata.json')
        records_path = Path(str(prefix)+'.records.jsonl')
        raw_hash = digest(records_path)
        require(raw_hash not in seen_hashes, 'reused raw content')
        seen_hashes.add(raw_hash)
        require(metadata['records_sha256'] == raw_hash, 'raw records hash mismatch')
        require(metadata['manifest_sha256'] == digest(config_path), 'probe config hash mismatch')
        require(metadata.get('run_id') and metadata['run_id'] not in seen_ids, 'reused or missing run identity')
        seen_ids.add(metadata['run_id'])
        for key in ('model_sha256', 'workload_sha256', 'implementation_sha256', 'precision',
                    'threads', 'batch_size', 'cycle', 'leg', 'variant', 'selected_cases', 'runtime_dependencies_sha256'):
            require(listed[key] == config[key], f'summary/config {key} mismatch')
        validate_runtime_dependencies(config.get('runtime_dependencies_sha256'))
        require(metadata.get('runtime_dependencies_sha256') == config['runtime_dependencies_sha256'], 'measured runtime dependencies mismatch')
        require(metadata.get('runtime_dependencies_stable') is True, 'runtime dependencies stability unverified')
        require(metadata.get('runtime_dependencies_origin') == 'observed_process_maps', 'native runtime dependency origin mismatch')
        require(config['schema'] == 'encoder-probe-v1' and config['status'] == 'diagnostic_only', 'probe config schema')
        require(config['workload_sha256'] == workload_hash, 'workload hash mismatch')
        for key in ('model_sha256', 'implementation_sha256'):
            require(sha(config[key]), f'invalid {key}')
        require(config.get('legs') == 'ABBA', 'ABBA legs required')
        require(type(config['cycle']) is int and 0 <= config['cycle'] < 5, 'cycle must be 0..4')
        require(type(config['leg']) is int and 0 <= config['leg'] < 4 and
                config['variant'] == 'ABBA'[config['leg']], 'ABBA variant order')
        require(config['precision'] in ('fp32', 'q8_0') and
                (config['variant'] != 'A' or config['precision'] == 'fp32'), 'invalid variant precision')
        for key in ('threads', 'batch_size'):
            require(type(config[key]) is int and config[key] > 0 and metadata[key] == config[key], f'{key} control mismatch')
        require(metadata['boundary'] == 'full_model' and metadata['concurrency'] == 1, 'native probe boundary/concurrency')
        require(metadata['warmup_requests'] > 0 and config['duration_s'] >= 120 and
                metadata['submission_stop_s'] == config['duration_s'], 'warm-up and 120-second submission required')
        duration = number(metadata['elapsed_s'], 'elapsed interval', positive=True)
        require(duration >= metadata['submission_stop_s'], 'drain omitted')
        timing = clock_evidence(metadata, duration)
        start, end = timing['start'], timing['end']
        selected = config['selected_cases']
        require(selected and len(selected) == len(set(selected)) and all(rid in all_rows for rid in selected), 'invalid selected cases')
        rows = [all_rows[rid] for rid in selected]
        known_useful = all('_useful_accounting' not in row for row in rows)
        records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
        require(records and metadata['record_count'] == len(records), 'missing raw requests')
        attempts, useful, computed, decisions, failures = set(), 0, 0, 0, 0
        for record in records:
            attempt = record['attempt_id']
            require(type(attempt) is int and attempt >= 0 and attempt not in attempts, 'duplicate/invalid attempt')
            attempts.add(attempt)
            require(record.get('run_id') == metadata['run_id'], 'record run identity mismatch')
            row = rows[attempt % len(rows)]
            require(record['workload_id'] == row['id'], 'workload request order mismatch')
            require(record['batch_size'] == config['batch_size'] and
                    record['batch_id'] == attempt//config['batch_size']*config['batch_size'], 'raw batch control mismatch')
            begin, finish = number(record['start_s'], 'request start'), number(record['end_s'], 'request end')
            require(begin <= metadata['submission_stop_s'] and begin <= finish <= duration, 'request outside full interval')
            require(record.get('unresolved_backend_work') is False, 'unresolved backend work')
            require(record['useful_tokens'] == row['_useful_tokens'] and record['compute_tokens'] == row['_compute_tokens'], 'raw token count mismatch')
            require(type(record['success']) is bool, 'success flag')
            if record['success']:
                require(record.get('error') is None, 'successful record has error')
                validate_native_response(record['response'], row)
                useful += row['_useful_tokens']
                computed += row['_compute_tokens']
                decisions += len(row['sequences'])
            else:
                failures += 1
        require(attempts == set(range(len(records))) and len(records) % config['batch_size'] == 0, 'missing attempts or partial batch')
        require(decisions > 0, 'no successful decisions')
        if model_path is not None:
            require(digest(model_path) == config['model_sha256'], 'actual model hash mismatch')
        library = baseline_library if config['variant'] == 'A' else candidate_library
        if library is not None:
            require(digest(library) == config['implementation_sha256'], 'actual library hash mismatch')
        key = (config['threads'], config['batch_size'], tuple(selected))
        groups.setdefault(key, []).append({'config': config, 'metadata': metadata, 'start': start, 'end': end, 'timing': timing,
            'raw_sha256': raw_hash, 'error_rate': failures/len(records), 'elapsed_s': duration,
            'drain_s': duration-metadata['submission_stop_s'],
            'useful_tokens_per_s': useful/duration if known_useful else None,
            'compute_tokens_per_s': computed/duration, 'decisions_per_s': decisions/duration})
    require(groups, 'empty probe summary')
    results = []
    for (threads, batch, selected), runs in groups.items():
        require(len(runs) == 20, 'exactly five ABBA cycles required per configuration')
        runs.sort(key=lambda run: (run['config']['cycle'], run['config']['leg']))
        require([(r['config']['cycle'], r['config']['leg']) for r in runs] ==
                [(cycle, leg) for cycle in range(5) for leg in range(4)], 'missing or duplicate ABBA legs')
        identities, previous_run, model_hash, host, controls = {}, None, None, None, None
        clock_mode, boot_id, monotonic_clock, clock_warnings = None, None, None, []
        for run in runs:
            c, m = run['config'], run['metadata']
            timing = run['timing']
            clock_mode = clock_mode or timing['mode']
            require(timing['mode'] == clock_mode, 'clock evidence protocol changed within ABBA group')
            if clock_mode == 'observed_monotonic_v2':
                boot_id = boot_id or timing['boot_id']
                monotonic_clock = monotonic_clock or timing['clock']
                require(timing['boot_id'] == boot_id, 'observed boot identity changed across runs')
                require(timing['clock'] == monotonic_clock, 'observed monotonic clock changed across runs')
                require(previous_run is None or timing['monotonic_start_ns'] >= previous_run['timing']['monotonic_end_ns'],
                        'overlapping/out-of-order absolute monotonic runs')
                if previous_run is not None:
                    wall_gap = (run['start']-previous_run['end']).total_seconds()
                    monotonic_gap = (timing['monotonic_start_ns']-previous_run['timing']['monotonic_end_ns'])/1_000_000_000
                    if abs(wall_gap-monotonic_gap) >= 1:
                        clock_warnings.append({'run_id': m['run_id'],
                            'kind': 'wall_clock_order_discontinuity' if wall_gap < 0 else 'wall_clock_gap_discontinuity',
                            'wall_gap_s': wall_gap, 'monotonic_gap_s': monotonic_gap,
                            'wall_minus_monotonic_gap_s': wall_gap-monotonic_gap})
            else:
                require(previous_run is None or run['start'] >= previous_run['end'], 'overlapping/out-of-order runs')
            clock_warnings.extend(dict(warning, run_id=m['run_id']) for warning in timing['warnings'])
            previous_run = run
            model_hash = model_hash or c['model_sha256']
            host = host or m['host_id']
            require(c['model_sha256'] == model_hash and m['host_id'] == host and bool(host), 'model/host provenance mismatch')
            current_controls = (c['duration_s'], m['warmup_requests'], m['concurrency'])
            controls = controls or current_controls
            require(controls == current_controls, 'measurement controls changed')
            identity = (c['implementation_sha256'], c['precision'], c['runtime_dependencies_sha256'])
            require(identities.setdefault(c['variant'], identity) == identity, 'variant implementation/precision/runtime dependencies changed')
        require(identities['A'] != identities['B'], 'baseline and candidate must differ')
        metrics = {}
        for metric in ('useful_tokens_per_s', 'compute_tokens_per_s', 'decisions_per_s'):
            if any(run[metric] is None for run in runs):
                metrics[metric] = {'status': 'unavailable', 'reason': 'workload lacks logical useful-token accounting'}
                continue
            ratios, candidates = [], []
            for cycle in range(5):
                values = [run[metric] for run in runs[cycle*4:cycle*4+4]]
                a, b = statistics.mean((values[0], values[3])), statistics.mean(values[1:3])
                require(a > 0, 'zero baseline rate')
                ratios.append(b/a); candidates.append(b)
            lo = lower95(ratios)
            metrics[metric] = {'status': 'measured', 'cycle_candidate_over_baseline': ratios,
                'candidate_over_baseline_mean': statistics.mean(ratios),
                'candidate_over_baseline_lower95': lo,
                'candidate_rate_mean': statistics.mean(candidates),
                'candidate_rate_lower95': max(0, lower95(candidates)),
                'performance_improvement': 'demonstrated' if lo > 1 else 'not_demonstrated'}
        results.append({'threads': threads, 'batch_size': batch, 'selected_cases': list(selected),
            'model_sha256': model_hash, 'workload_sha256': workload_hash, 'host_id': host,
            'variant_identities': {variant: {'implementation_sha256': value[0], 'precision': value[1], 'runtime_dependencies_sha256': value[2]}
                                   for variant, value in identities.items()},
            'clock_evidence': {'protocol': clock_mode, 'boot_id': boot_id, 'monotonic_clock': monotonic_clock,
                               'wall_clock_status': 'discontinuity_observed' if clock_warnings else 'consistent_with_measured_intervals',
                               'warnings': clock_warnings},
            'cycles': 5, 'metrics': metrics, 'max_error_rate': max(run['error_rate'] for run in runs),
            'runs': [{key: run[key] for key in ('raw_sha256', 'elapsed_s', 'drain_s', 'error_rate')} for run in runs]})
    return {'status': 'measured_local_performance', 'quality_verdict': 'unverified',
            'economics_verdict': 'unavailable', 'public_gpu_2x': 'unavailable', 'configurations': results,
            'file_verification': {'workload': 'verified',
                'model': 'verified' if model_path is not None else 'declared_hash_only',
                'baseline_library': 'verified' if baseline_library is not None else 'declared_hash_only',
                'candidate_library': 'verified' if candidate_library is not None else 'declared_hash_only'},
            'limitations': ['Quality requires separate independent evaluation.',
                            'Student t intervals assume five independent cycle estimates.',
                            'No prices or GPU economics are inferred.',
                            'V2 timing uses boot-bound absolute monotonic evidence; wall-clock discontinuities are reported.',
                            'Legacy artifacts without monotonic evidence retain strict realtime validation.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    parser.add_argument('--workload', type=Path, required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--baseline-library', type=Path)
    parser.add_argument('--candidate-library', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = check(args.summary, args.workload, args.model, args.baseline_library, args.candidate_library)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
