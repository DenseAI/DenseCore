#!/usr/bin/env python3
"""Same-file native prefix token parity and optional HTTP final-answer validation.

Runs fresh off/on processes; exact callback IDs include sampled stop tokens.
Cache pressure must produce actual allocator KV/snapshot eviction counters to
pass its strict gate. No speed, batching, support, or default-promotion claim.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes as C
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import urllib.request

from check_mtp_parity import (COMPLETE, TOKEN, InitOptions, RuntimePlan, digest,
                              validate_native, wait_for_first_token, write)
from compare_servers import validate_response


def prefix_evidence(text):
    fields = ('blocks', 'hybrid_snapshots', 'hybrid_payload_bytes')
    counts = {field: [int(x) for x in re.findall(rf'prefix_cache_evicted_{field}_total=(\d+)', text)]
              for field in fields}
    return {'hits': len(re.findall(r'prefix_cache_hit=1\b', text)),
            'restores': len(re.findall(r'hybrid_ssm_snapshot_restore_applied=1\b', text)),
            'skipped_tokens': [int(x) for x in re.findall(r'prefix_cache_skipped_tokens=(\d+)', text) if int(x)],
            'eviction_counters_present': all(counts.values()),
            'eviction_counters': {field: {'start': min(values), 'end': max(values), 'delta': max(values)-min(values)}
                                  for field, values in counts.items() if values}}


def compare(groups, require_eviction=False):
    failures = []
    if set(groups) != {'off', 'on'}:
        return ['fresh off/on process evidence missing']
    baseline = {row['label']: row for row in groups['off']['requests']}
    enabled = {row['label']: row for row in groups['on']['requests']}
    if baseline.keys() != enabled.keys() or not baseline:
        failures.append('off/on request matrix differs or is empty')
    required = {*(f'repeat{i}' for i in range(1, 5)), *(f'turn{i}' for i in range(1, 6)), 'after-cancel', 'after-churn'}
    if not required <= baseline.keys(): failures.append('repeat/five-turn/cancel/churn matrix incomplete')
    for mode, group in groups.items():
        concurrency = group.get('requested_concurrency', [])
        labels = {row['label'] for row in group['requests']}
        for width in concurrency:
            if not {f'c{width}-{i}' for i in range(width)} <= labels:
                failures.append(f'{mode}: c{width} request matrix incomplete')
        prompts = {}
        for row in group['requests']:
            try:
                validate_native(row)
            except ValueError as error:
                failures.append(f"{mode}/{row['label']}: {error}")
            if 'prompt' in row and prompts.setdefault(row['prompt'], row['token_ids']) != row['token_ids']:
                failures.append(f"{mode}/{row['label']}: same prompt differs across scheduling/concurrency")
        if not group.get('cancellation', {}).get('passed'):
            failures.append(f'{mode}: cancellation/recovery evidence missing')
        for row in group.get('churn_requests', []):
            try:
                validate_native(row)
            except ValueError as error:
                failures.append(f"{mode}/{row['label']}: cache pressure request failed: {error}")
        repeat = [row for row in group['requests'] if row['label'] in ('repeat1', 'repeat4')]
        if len(repeat) != 2 or repeat[0]['token_ids'] != repeat[1]['token_ids']:
            failures.append(f'{mode}: fresh process #1/#4 exact IDs differ')
    for label in baseline.keys() & enabled.keys():
        if baseline[label]['token_ids'] != enabled[label]['token_ids']:
            failures.append(f'{label}: off/on exact sampled token IDs differ')
    evidence = groups['on'].get('prefix_evidence', {})
    if not evidence.get('hits') or not evidence.get('restores') or not evidence.get('skipped_tokens'):
        failures.append('on: actual snapshot restore/hit/skipped-prefill evidence missing')
    eviction = groups['on'].get('eviction', {})
    if require_eviction and (not eviction.get('counters_present') or
            any(eviction.get(field, 0) <= 0 for field in ('kv_blocks_reclaimed_delta',
                'hybrid_snapshot_entries_removed_delta', 'hybrid_logical_payload_bytes_removed_delta'))):
        failures.append('eviction not observed: actual KV reclamation and hybrid snapshot removal counters required')
    return failures


def child(args):
    lib = C.CDLL(str(args.library))
    lib.DenseCoreGetLastError.restype = C.c_char_p
    lib.InitEngineWithOptions.argtypes = [C.c_char_p, C.c_char_p, C.c_int, C.c_int, C.c_int, C.c_int, C.POINTER(InitOptions)]
    lib.InitEngineWithOptions.restype = C.c_void_p
    lib.FreeEngine.argtypes = [C.c_void_p]
    lib.DenseCoreGetRuntimePlan.argtypes = [C.c_void_p, C.POINTER(RuntimePlan)]
    lib.DenseCoreSetGenerationCompletionCallback.argtypes = [C.c_void_p, COMPLETE]
    lib.SubmitRequestWithTokenResults.argtypes = [C.c_void_p, C.c_char_p, C.c_int, C.c_float, C.c_float,
                                                C.c_int, C.c_float, TOKEN, C.c_void_p]
    lib.SubmitRequestWithTokenResults.restype = C.c_int
    lib.CancelRequest.argtypes = [C.c_void_p, C.c_int]
    options = InitOptions(C.sizeof(InitOptions), 1, 1, 2)
    handle = lib.InitEngineWithOptions(os.fsencode(args.model), None, args.threads, -1, 0, 0, C.byref(options))
    if not handle:
        raise RuntimeError((lib.DenseCoreGetLastError() or b'initialization failed').decode(errors='replace'))
    records, mutex = {}, threading.Lock()

    @TOKEN
    def token(item, userdata):
        with mutex:
            row = records[int(userdata)]
            value = item.contents
            if value.is_finished:
                row['terminal_events'] += 1
                row['done'].set()
            elif value.token_id >= 0:
                row['token_ids'].append(value.token_id)
                row['pieces'].append((value.text or b'').decode(errors='replace'))
                row['first_token'].set()

    @COMPLETE
    def complete(count, reason, userdata):
        with mutex:
            records[int(userdata)].update(generated_tokens=count, finish_reason=reason)

    def submit(label, prompt, cancel=False):
        with mutex:
            key = len(records) + 1
            row = {'label': label, 'prompt': prompt, 'token_ids': [], 'pieces': [], 'terminal_events': 0,
                   'done': threading.Event(), 'first_token': threading.Event()}
            records[key] = row
        started = time.monotonic()
        request = lib.SubmitRequestWithTokenResults(handle, prompt.encode(), 1024 if cancel else args.max_tokens,
                                                    0.0, 1.0, 0, 1.0, token, C.c_void_p(key))
        row['request_id'] = request
        if request < 0:
            raise RuntimeError((lib.DenseCoreGetLastError() or b'submit failed').decode(errors='replace'))
        if cancel:
            wait_for_first_token(row, args.timeout)
            row['cancel_status'] = lib.CancelRequest(handle, request)
        if not row['done'].wait(args.timeout):
            lib.CancelRequest(handle, request)
            raise TimeoutError(f'{label} failed to terminate')
        row['wall_seconds'] = time.monotonic() - started
        exported = {k: v for k, v in row.items() if k not in ('done', 'first_token')}
        if not cancel: validate_native(exported)
        return exported

    # Fixed history avoids mode-specific answers changing later prompt bytes.
    system = 'You are a concise geography assistant. Give only the requested city name. ' * 8
    countries = [('France', 'Paris'), ('Japan', 'Tokyo'), ('Germany', 'Berlin'), ('Italy', 'Rome'), ('Spain', 'Madrid')]
    def question(country):
        return f'User: What is the capital of {country}? Answer with the city name only.\nAssistant:'
    result = {'mode': args.child, 'requests': [], 'churn_requests': [],
              'requested_concurrency': args.concurrency,
              'scope': 'raw native prompts; final answer QA uses HTTP chat separately'}
    try:
        plan = RuntimePlan(); plan.struct_size, plan.version = C.sizeof(plan), 1
        if lib.DenseCoreGetRuntimePlan(handle, C.byref(plan)) != 0:
            raise RuntimeError('runtime plan unavailable')
        result['runtime_plan'] = {name: getattr(plan, name) for name, _ in plan._fields_ if name != 'spec_reason'}
        if plan.spec_selected != 1:
            raise RuntimeError('requested ordinary decode was not selected')
        if lib.DenseCoreSetGenerationCompletionCallback(handle, complete) != 0:
            raise RuntimeError('completion callback unavailable')
        probe = system + question('France')
        for i in range(1, 5): result['requests'].append(submit(f'repeat{i}', probe))
        for concurrency in args.concurrency:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                jobs = [(f'c{concurrency}-{i}', system + question(country))
                        for i, (country, _) in enumerate(countries[:concurrency])]
                result['requests'].extend(pool.map(lambda job: submit(*job), jobs))
        history = system
        for turn, (country, city) in enumerate(countries, 1):
            result['requests'].append(submit(f'turn{turn}', history + question(country)))
            history += question(country) + f' {city}\n'
        cancellation = submit('cancel', system + 'List integers from 1 through 10000, one per line. Start now.', True)
        result['cancellation'] = {'request': cancellation, 'passed': cancellation.get('cancel_status') == 0 and
            cancellation.get('terminal_events') == 1 and cancellation.get('finish_reason') == 3}
        result['requests'].append(submit('after-cancel', probe))
        # Distinct early prefixes create pressure; logs distinguish a proved
        # eviction from a cache that simply retained every probe.
        for i in range(args.churn):
            result['churn_requests'].append(submit(f'churn{i}', f'Cache pressure document {i}. ' * 24 + question('France')))
        result['requests'].append(submit('after-churn', probe))
    except BaseException as error:
        result['error'] = str(error)
        raise
    finally:
        lib.FreeEngine(handle)
        # Preserve the failing/in-flight row too, even when validation stops a
        # pressure wave or an operator interrupts a repeated scheduling stall.
        result['all_requests'] = [{k: v for k, v in row.items() if k not in ('done', 'first_token')}
                                  for row in records.values()]
        write(args.out, result)


def attach_log_evidence(result, text):
    result['prefix_evidence'] = prefix_evidence(text)
    # Keep the particular probe's residency as information. Cache policy can
    # legitimately retain this hot probe while reclaiming different entries.
    profiles = {}
    for line in text.splitlines():
        if 'prefix_cache_hit=' not in line: continue
        rid = re.search(r'(?:req(?:uest)?(?:_id)?[= :]+)(\d+)', line)
        hit = re.search(r'prefix_cache_hit=(\d+)', line)
        if rid and hit: profiles[int(rid[1])] = int(hit[1])
    rows = {row['label']: row for row in result.get('requests', [])}
    before = profiles.get(rows.get('repeat4', {}).get('request_id'))
    after = profiles.get(rows.get('after-churn', {}).get('request_id'))
    result['eviction'] = {'probe_before_hit': before, 'probe_after_hit': after,
                          'observed_miss_after_hit': before == 1 and after == 0,
                          'scope': 'process-cumulative allocator counters; fresh single-model child',
                          'counters_present': result['prefix_evidence']['eviction_counters_present']}
    counters = result['prefix_evidence']['eviction_counters']
    for source, target in (('blocks', 'kv_blocks_reclaimed_delta'),
            ('hybrid_snapshots', 'hybrid_snapshot_entries_removed_delta'),
            ('hybrid_payload_bytes', 'hybrid_logical_payload_bytes_removed_delta')):
        result['eviction'][target] = counters.get(source, {}).get('delta', 0)


def http_final_answers(args, mode, environment):
    """Actual chat-template final answers with cache_prompt explicitly enabled."""
    base = f'http://127.0.0.1:{args.port}'
    log = args.out / f'http-{mode}.log'
    rows = []
    with log.open('w') as output:
        process = subprocess.Popen([str(args.server.resolve()), 'serve', '-m', str(args.model),
            '-t', str(args.threads), '--host', '127.0.0.1', '--port', str(args.port), '--spec-type', 'none'],
            env=environment, stdout=output, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + args.timeout
            while True:
                if process.poll() is not None: raise RuntimeError(f'HTTP {mode} exited; see {log}')
                try:
                    with urllib.request.urlopen(base + '/health/ready', timeout=2) as response:
                        if response.status == 200: break
                except OSError: pass
                if time.monotonic() >= deadline: raise TimeoutError('HTTP readiness timeout')
                time.sleep(.2)
            system = {'role': 'system', 'content': 'Give only the requested city name. ' * 16}
            countries = [('France', 'Paris'), ('Japan', 'Tokyo'), ('Germany', 'Berlin'), ('Italy', 'Rome'), ('Spain', 'Madrid')]
            history = [system]
            for turn, (country, city) in enumerate(countries, 1):
                history.append({'role': 'user', 'content': f'What is the capital of {country}? Answer with the city name only.'})
                body = {'model': 'local', 'messages': list(history), 'temperature': 0, 'max_tokens': args.http_max_tokens,
                        'stream': False, 'cache_prompt': True, 'chat_template_kwargs': {'enable_thinking': False}}
                row = {'turn': turn, 'request': body}
                request = urllib.request.Request(base + '/v1/chat/completions', json.dumps(body).encode(),
                                                 {'Content-Type': 'application/json'})
                try:
                    with urllib.request.urlopen(request, timeout=args.timeout) as response: raw = json.load(response)
                    row['response'] = raw
                    choice = raw['choices'][0]
                    message = choice['message']
                    row['final_answer'] = validate_response(message.get('content') or '', message.get('reasoning_content') or '',
                                                            choice.get('finish_reason'), raw.get('usage', {}), rf'\b{city}\b')
                except Exception as error:
                    row['error'] = str(error)
                rows.append(row)
                write(args.out / f'http-{mode}.json', rows)
                history.append({'role': 'assistant', 'content': city})
        finally:
            process.terminate()
            try: process.wait(timeout=20)
            except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=10)
    return rows


def parent(args):
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {'library_sha256': digest(args.library), 'model_sha256': digest(args.model),
                'harness_sha256': digest(__file__), 'native': {}, 'http': {},
                'qualification': 'correctness evidence only; no default promotion or performance claim'}
    if args.server: manifest['server_sha256'] = digest(args.server)
    manifest['configuration'] = {name: getattr(args, name) for name in
        ('context', 'threads', 'concurrency', 'max_tokens', 'http_max_tokens', 'churn')}
    manifest['dependencies'] = {str(path): digest(path) for path in
        (Path(__file__).with_name('check_mtp_parity.py'), Path(__file__).with_name('compare_servers.py'))}
    manifest['host'] = os.uname()._asdict() if hasattr(os.uname(), '_asdict') else list(os.uname())
    manifest['source_checkout_head'] = subprocess.check_output(
        ['git', '-C', str(args.library.parent), 'rev-parse', 'HEAD'], text=True).strip()
    write(args.out / 'summary.json', manifest)
    for mode in ('off', 'on'):
        env = dict(os.environ, DENSECORE_PREFIX_CACHE_REUSE=str(int(mode == 'on')),
                   DENSECORE_QWEN35_PREFIX_CACHE_REUSE=str(int(mode == 'on')),
                   DENSECORE_MAX_SEQ_LEN=str(args.context), DENSECORE_MAX_NUM_SEQS=str(max(args.concurrency)),
                   LD_LIBRARY_PATH=str(args.library.parent) + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
        # Same startup preload as SDK/native MTP harness for initial-exec TLS.
        env['LD_PRELOAD'] = ' '.join(dict.fromkeys([str(args.library), *env.get('LD_PRELOAD', '').split()]))
        raw = args.out / f'native-{mode}.json'
        log = args.out / f'native-{mode}.log'
        command = [sys.executable, str(Path(__file__).resolve()), '--child', mode, '--library', str(args.library),
                   '--model', str(args.model), '--out', str(raw), '--threads', str(args.threads),
                   '--max-tokens', str(args.max_tokens), '--timeout', str(args.timeout), '--churn', str(args.churn),
                   '--concurrency', *map(str, args.concurrency)]
        with log.open('w') as output:
            run = subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT,
                                 timeout=args.timeout * (20 + args.churn + sum(args.concurrency)))
        if raw.exists():
            result = json.loads(raw.read_text())
            attach_log_evidence(result, log.read_text(errors='replace'))
            manifest['native'][mode] = result
            write(args.out / 'summary.json', manifest)
        if run.returncode:
            raise RuntimeError(f'native {mode} failed; see {log}')
        if args.server:
            manifest['http'][mode] = http_final_answers(args, mode, env)
            write(args.out / 'summary.json', manifest)
    failures = compare(manifest['native'], args.require_eviction)
    for mode, result in manifest['native'].items():
        plan = result.get('runtime_plan', {})
        if plan.get('effective_context') != args.context or plan.get('effective_concurrency') != max(args.concurrency):
            failures.append(f'{mode}: effective context/concurrency differs from requested workload')
    for mode, result in manifest['http'].items():
        if len(result) != 5: failures.append(f'HTTP {mode}: five-turn evidence incomplete')
        for row in result:
            if row.get('error'): failures.append(f"HTTP {mode}/turn{row['turn']}: {row['error']}")
    if digest(args.library) != manifest['library_sha256'] or digest(args.model) != manifest['model_sha256']:
        failures.append('library/model bytes changed during run')
    if args.server and digest(args.server) != manifest['server_sha256']:
        failures.append('server bytes changed during run')
    for path, expected in manifest['dependencies'].items():
        if digest(path) != expected: failures.append(f'harness dependency changed: {path}')
    manifest.update(passed=not failures, failures=failures, http_final_answer_verified=bool(args.server),
                    eviction_required=args.require_eviction)
    write(args.out / 'summary.json', manifest)
    print(f"{'FAIL' if failures else 'PASS'}: {args.out / 'summary.json'}")
    return int(bool(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--server', type=Path)
    parser.add_argument('--build-lock', type=Path)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--context', type=int, default=2048)
    parser.add_argument('--concurrency', type=int, nargs='+', choices=(1, 2, 4), default=[1, 2, 4])
    parser.add_argument('--max-tokens', type=int, default=96)
    parser.add_argument('--http-max-tokens', type=int, default=1024)
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--churn', type=int, default=32)
    parser.add_argument('--require-eviction', action='store_true')
    parser.add_argument('--port', type=int, default=18090)
    parser.add_argument('--child', choices=('off', 'on'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.library, args.model, args.out = args.library.resolve(), args.model.resolve(), args.out.resolve()
    args.concurrency = sorted(set(args.concurrency))
    if min(args.threads, args.context, args.max_tokens) < 1 or args.churn < 0 or args.timeout <= 0:
        parser.error('threads/context/tokens/timeout must be positive; churn must be nonnegative')
    if args.child:
        child(args)
        return 0
    try:
        if args.build_lock:
            import fcntl
            with args.build_lock.open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                return parent(args)
        return parent(args)
    except Exception as error:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / 'summary.json'
        partial = json.loads(path.read_text()) if path.exists() else {}
        partial.update(passed=False, error=str(error))
        write(path, partial)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
