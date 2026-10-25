#!/usr/bin/env python3
"""Run predeclared sticky OFF/ON and best-llama comparisons with five ABBA blocks.

No cloud provisioning. Commands, model bytes, affinity and raw API output are recorded.
A separate completed-answer and kernel/fallback audit is required before promotion.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

spec = importlib.util.spec_from_file_location('measure', Path(__file__).parents[1] / 'personal' / 'compare_servers.py')
measure = importlib.util.module_from_spec(spec); spec.loader.exec_module(measure)


def dense_controls(case):
    if case not in ('off', 'on'):
        raise ValueError('dense NUMA case must be off or on')
    # Native graph policy requires the operational opt-in. Both the native
    # graph and small-decode backend honor the diagnostic disable in a fresh
    # debug-enabled process; the latter has a separate maintained default ON.
    return {'DENSECORE_EXPERIMENTAL_MOE_NUMA_STICKY': '1' if case == 'on' else '0',
        'DENSECORE_DEBUG_DISABLE_MOE_NUMA_STICKY': '0' if case == 'on' else '1'}


def build_environment(config, case, threads):
    env = os.environ.copy()
    env.update(config.get('env', {}))
    if case == 'llama':
        # Never load DenseCore's different ggml ABI into the baseline process.
        env.pop('LD_PRELOAD', None)
        env['LD_LIBRARY_PATH'] = str(Path(config['llama_command'][0]).resolve().parent)
        env.update(config.get('llama_env', {}))
    else:
        env.update(config.get('dense_env', {}))
        env.update(dense_controls(case))
    env.update({'THREADS': str(threads), 'OMP_NUM_THREADS': str(threads),
        'DENSECORE_MAX_NUM_SEQS': '1', 'DENSECORE_MAX_SEQ_LEN': str(config['context']),
        'MAIN_MODEL_PATH': config['model'], 'DENSECORE_PREFIX_CACHE_REUSE': '0',
        'DENSECORE_NUMA_WEIGHTS': 'round_robin', 'HOST': '127.0.0.1',
        'PORT': str(config.get('port', 18082))})
    return env


def require_debug_build(config):
    path = Path(config['dense_build_cache'])
    text = path.read_text()
    if not re.search(r'^DENSECORE_ENABLE_DEBUG_ENV:BOOL=(ON|TRUE|1)$', text, re.MULTILINE):
        raise ValueError('NUMA OFF control requires a server configured with -DDENSECORE_ENABLE_DEBUG_ENV=ON')
    with path.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    return {'path': str(path.resolve()), 'sha256': digest, 'diagnostic_env_enabled': True}


def verify_dispatch_policy(text, case, minimum_records=1):
    rows = []
    for line in text.splitlines():
        state = re.search(r'\bnative_moe_numa_sticky_state=(\w+)', line)
        sticky = re.search(r'\bnative_moe_numa_sticky_dispatch_ops=(\d+)', line)
        legacy = re.search(r'\bnative_moe_numa_legacy_dispatch_ops=(\d+)', line)
        if not (state and sticky and legacy):
            continue
        histogram = re.search(r'\bnative_moe_numa_node_dispatch_ops=(\S+)', line)
        rows.append({'state': state[1], 'sticky_dispatch_ops': int(sticky[1]),
            'legacy_dispatch_ops': int(legacy[1]),
            'node_dispatch_ops': histogram[1] if histogram else None})
    if len(rows) < minimum_records:
        raise ValueError('missing native NUMA state/dispatch evidence; no speed claim is valid')
    if case == 'off':
        if any(row['state'] != 'disabled_by_debug' or row['sticky_dispatch_ops'] != 0 or
                row['legacy_dispatch_ops'] <= 0 for row in rows):
            raise ValueError('NUMA OFF was not proven disabled by the live diagnostic control')
        if 'MoE sticky decode dispatch active' in text:
            raise ValueError('small-decode sticky path remained active in the OFF process')
    elif case == 'on':
        if any(row['state'] != 'enabled' or row['sticky_dispatch_ops'] <= 0 for row in rows):
            raise ValueError('NUMA ON not proven: unsupported/degenerate/unqualified model or zero dispatch')
    else:
        raise ValueError('NUMA proof applies only to dense off/on cases')
    return {'verified': True, 'case': case, 'observations': rows}


def physical_cpus(threads):
    rows = subprocess.check_output(['lscpu', '-p=CPU,CORE,SOCKET,NODE'], text=True)
    nodes, seen = {}, set()
    allowed = os.sched_getaffinity(0)
    for row in rows.splitlines():
        if row.startswith('#'): continue
        cpu, core, socket, node = map(int, row.split(','))
        if cpu not in allowed or (socket, core) in seen: continue
        seen.add((socket, core)); nodes.setdefault(node, []).append(cpu)
    selected = []
    while len(selected) < threads:
        progress = False
        for node in sorted(nodes):
            if nodes[node] and len(selected) < threads:
                selected.append(nodes[node].pop(0)); progress = True
        if not progress: raise ValueError(f'need {threads} physical cores inside process affinity')
    return selected


def sample(config, case, threads, cpus, out):
    port = config.get('port', 18082)
    command = config['dense_command'] if case != 'llama' else config['llama_command']
    env = build_environment(config, case, threads)
    if case == 'llama':
        command = command + ['-m', config['model'], '-t', str(threads), '-tb', str(threads),
            '-c', str(config['context']), '--host', '127.0.0.1', '--port', str(port), '--jinja']
        mode = config['llama_numa']
        if mode != 'none': command += ['--numa', mode]
        if mode == 'numactl': command = ['numactl', '--interleave=all'] + command
        if config.get('llama_mtp', False):
            command += ['--spec-type', 'draft-mtp', '--spec-draft-n-max', str(config.get('draft_n_max', 2))]
    command = ['taskset', '-c', ','.join(map(str, cpus))] + command
    metadata = {'case': case, 'threads': threads, 'cpus': cpus, 'command': command,
        'environment_overrides': {key: env[key] for key in ('THREADS', 'DENSECORE_MAX_NUM_SEQS',
            'DENSECORE_MAX_SEQ_LEN', 'OMP_NUM_THREADS', 'DENSECORE_PREFIX_CACHE_REUSE',
            'DENSECORE_NUMA_WEIGHTS', 'LD_LIBRARY_PATH', 'LD_PRELOAD',
            'DENSECORE_EXPERIMENTAL_MOE_NUMA_STICKY', 'DENSECORE_DEBUG_DISABLE_MOE_NUMA_STICKY') if key in env},
        'llama_numa': config['llama_numa'], 'llama_mtp': config.get('llama_mtp', False)}
    with out.with_suffix('.server.log').open('wb') as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=log, start_new_session=True)
        try:
            deadline = time.monotonic() + 600
            while True:
                if process.poll() is not None: raise RuntimeError('server exited during startup')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=1): break
                except OSError:
                    if time.monotonic() >= deadline: raise RuntimeError('server readiness timeout')
                    time.sleep(.5)
            maps_path = Path(f'/proc/{process.pid}/maps')
            maps = maps_path.read_text()
            out.with_suffix('.proc-maps.txt').write_text(maps)
            loaded = {}
            for line in maps.splitlines():
                parts = line.split(maxsplit=5)
                if len(parts) < 6 or not parts[5].startswith('/'):
                    continue
                path = Path(parts[5])
                if not any(name in path.name for name in ('libdensecore', 'libggml', 'libllama')):
                    continue
                if str(path) not in loaded and path.is_file():
                    with path.open('rb') as handle:
                        loaded[str(path)] = hashlib.file_digest(handle, 'sha256').hexdigest()
            metadata['loaded_library_sha256'] = loaded
            result = measure.group(f'http://127.0.0.1:{port}', config['prompts'], 1,
                config.get('max_tokens', 1024), config['expected_pattern'])
            result['provenance'] = metadata
            if case != 'llama':
                # Native summaries may flush just after the HTTP terminal event.
                proof_deadline = time.monotonic() + 2
                while True:
                    log.flush()
                    try:
                        result['dispatch_policy_proof'] = verify_dispatch_policy(
                            out.with_suffix('.server.log').read_text(errors='replace'), case, len(config['prompts']))
                        break
                    except ValueError as error:
                        if time.monotonic() >= proof_deadline:
                            result['dispatch_policy_error'] = str(error)
                            break
                        time.sleep(.05)
            result['comparison_qualified'] = bool(result['quality_pass'] and
                (case == 'llama' or result.get('dispatch_policy_proof', {}).get('verified')))
            if not result['comparison_qualified']:
                result['unqualified_observed_aggregate_output_tps'] = result['aggregate_completed_output_tps']
                result['aggregate_completed_output_tps'] = None
            out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
            if not result['quality_pass']: raise RuntimeError(f'quality gate failed; raw sample: {out}')
            if result.get('dispatch_policy_error'):
                raise RuntimeError(f"{result['dispatch_policy_error']}; raw sample: {out}")
            return result
        finally:
            os.killpg(process.pid, signal.SIGTERM) if process.poll() is None else None
            try: process.wait(timeout=40)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
                raise RuntimeError('server exceeded shutdown deadline')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--threads', type=int, choices=(16, 48), required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config['llama_numa'] not in ('none', 'distribute', 'isolate', 'numactl'):
        parser.error('llama_numa must be selected before the scored comparison')
    config['dense_build_evidence'] = require_debug_build(config)
    args.out.mkdir(parents=True, exist_ok=False)
    # Verify model bytes once per invocation; a supplied digest alone is not verification.
    with Path(config['model']).open('rb') as handle:
        actual = hashlib.file_digest(handle, 'sha256').hexdigest()
    if actual != config['sha256']: raise ValueError('model SHA-256 mismatch')
    for family in ('dense', 'llama'):
        command = config[family + '_command']
        executable = Path(command[0])
        with executable.open('rb') as handle: config[family + '_binary_sha256'] = hashlib.file_digest(handle, 'sha256').hexdigest()
    config['physical_cpu_ids'] = physical_cpus(args.threads)
    (args.out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    env = os.environ.copy(); env['OUTDIR'] = str(args.out / 'physical-preflight')
    subprocess.run(['bash', str(Path(__file__).with_name('physical_preflight.sh'))], env=env, check=True)
    for pair in ('off', 'llama'):
        for block in range(5):
            for index, case in enumerate((pair, 'on', 'on', pair)):
                sample(config, case, args.threads, config['physical_cpu_ids'],
                    args.out / f'{pair}-block{block + 1}-{index + 1}-{case}.json')
    print(f'Raw five-block ABBA samples saved to {args.out}; no promotion inferred.')

if __name__ == '__main__': main()
