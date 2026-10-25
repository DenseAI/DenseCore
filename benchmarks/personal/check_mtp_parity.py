#!/usr/bin/env python3
"""Exact native token parity/isolation plus optional real HTTP final-answer checks.

Native sampling is greedy. The current public native API has no seed argument;
this harness does not pretend that a seed was applied. This is correctness
validation, not an ABBA performance qualification or a support promotion.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes as C
import hashlib
import http.client
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import threading
import traceback
import time
import urllib.request

CASES = [
    {"id": "capital", "prompt": "What is the capital of France? Reply with only the city name.", "expected": r"\bParis\b"},
    {"id": "arithmetic", "prompt": "What is 17 plus 25? Reply with only the integer.", "expected": r"\b42\b"},
]


NATIVE_CASES = CASES + [{"id": "long_parity", "prompt":
    "Write every integer from 1 through 80, one integer per line. Do not skip any integers. Start with 1."}]


def extract_mtp_counters(log):
    requests = {}
    for line in log.splitlines():
        if '[MTPVerify]' not in line:
            continue
        fields = {key: int(value) for key, value in re.findall(r'(\w+)=(\d+)', line)}
        if 'request_id' not in fields:
            continue
        current = requests.setdefault(fields.pop('request_id'), {})
        for key, value in fields.items():
            current[key] = max(current.get(key, 0), value)
    return {'requests': requests,
            'maximum_target_width': max((row.get('target_width', 0) for row in requests.values()), default=0),
            **{key: sum(row.get(key, 0) for row in requests.values()) for key in
               ('proposed', 'verified', 'accepted', 'rejected', 'target_rollbacks')}}


def digest(path):
    result = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')


def validate_native(row):
    if row.get('terminal_events') != 1:
        raise ValueError('native request did not terminate exactly once')
    if row.get('finish_reason') not in (1, 2):
        raise ValueError('native request failed instead of stop/length completion')
    if not row.get('token_ids') or len(row['token_ids']) != row.get('generated_tokens'):
        raise ValueError('structured callback token count differs from authoritative native count')


def load_c1_baseline(path, manifest, context, threads):
    baseline = json.loads(Path(path).read_text())
    if baseline.get('passed') is not True:
        raise ValueError('baseline summary did not pass its original gate')
    for key in ('model_sha256', 'library_sha256', 'dependencies', 'sampling', 'draft_n_max',
                'cpu_affinity', 'cpu_profile'):
        if key not in baseline or baseline[key] != manifest.get(key):
            raise ValueError(f'baseline provenance mismatch: {key}')
    if manifest.get('cmake_cache_sha256') != baseline.get('cmake_cache_sha256'):
        raise ValueError('baseline provenance mismatch: cmake_cache_sha256')
    if manifest.get('server_sha256') != baseline.get('server_sha256'):
        raise ValueError('baseline provenance mismatch: server_sha256')
    groups = [row for row in baseline.get('native', []) if row.get('concurrency') == 1]
    if {(row.get('mode'), row.get('concurrency')) for row in groups} != {('none', 1), ('mtp', 1)} or len(groups) != 2:
        raise ValueError('baseline must contain exactly one none/mtp c1 pair')
    for row in groups:
        plan = row.get('runtime_plan', {})
        if plan.get('effective_context') != context or plan.get('effective_threads') != threads:
            raise ValueError('baseline context/thread mismatch')
        row['evidence_source'] = str(Path(path).resolve())
    failures = compare_native(groups)
    if failures:
        raise ValueError('baseline native evidence invalid: ' + '; '.join(failures))
    return groups


def compare_native(rows):
    baseline = {}
    failures = []
    present = {(row['mode'], row['concurrency']) for row in rows}
    expected = {(mode, concurrency) for mode in ('none', 'mtp')
                for concurrency in ({1} | {row['concurrency'] for row in rows})}
    for mode, concurrency in sorted(expected - present):
        failures.append(f'native evidence missing for {mode}/c{concurrency}')
    for group in rows:
        expected = 1 if group['mode'] == 'none' else 2
        if group.get('runtime_plan', {}).get('spec_selected') != expected:
            failures.append(f"{group['mode']}/c{group['concurrency']}: requested mode was not selected")
        for row in group.get('requests', []):
            label = f"{group['mode']}/c{group['concurrency']}/{row['case']}/{row['repeat']}"
            try:
                validate_native(row)
            except ValueError as error:
                failures.append(f'{label}: {error}')
                continue
            if group['mode'] == 'none' and group['concurrency'] == 1:
                previous = baseline.setdefault(row['case'], row['token_ids'])
            else:
                previous = baseline.get(row['case'])
            if previous is None or row['token_ids'] != previous:
                mismatch = next((i for i, pair in enumerate(zip(previous or [], row['token_ids']))
                                 if pair[0] != pair[1]), min(len(previous or []), len(row['token_ids'])))
                failures.append(f'{label}: exact sampled token IDs differ from none/c1 at index {mismatch}')
        if not group.get('requests') or not group.get('cancellation', {}).get('passed'):
            failures.append(f"{group['mode']}/c{group['concurrency']}: requests/cancellation verification missing")
    return failures


def wait_for_first_token(row, timeout):
    deadline = time.monotonic() + timeout
    while not row['first_token'].is_set():
        if row['done'].is_set():
            raise RuntimeError('cancel probe terminated before producing a token')
        remaining = deadline - time.monotonic()
        if remaining <= 0: raise TimeoutError('cancel probe produced no token')
        row['first_token'].wait(min(0.05, remaining))


class InitOptions(C.Structure):
    _fields_ = [('struct_size', C.c_uint32), ('version', C.c_uint32), ('spec_type', C.c_int32), ('draft', C.c_int32)]


class RuntimePlan(C.Structure):
    _fields_ = [
        ('struct_size', C.c_uint32), ('version', C.c_uint32),
        *[(name, C.c_int32) for name in ('effective_threads', 'effective_context', 'effective_concurrency',
                                       'spec_requested', 'spec_selected', 'spec_draft_n_max')],
        *[(name, C.c_uint64) for name in ('available_bytes', 'required_bytes', 'kv_bytes', 'graph_bytes',
            'repack_bytes', 'recurrent_bytes', 'prefix_snapshot_bytes', 'mtp_bytes', 'allocator_bytes')],
        ('spec_reason', C.c_char * 192)]


class TokenResult(C.Structure):
    _fields_ = [('token_id', C.c_int), ('text', C.c_char_p), ('is_finished', C.c_int)]


TOKEN = C.CFUNCTYPE(None, C.POINTER(TokenResult), C.c_void_p)
COMPLETE = C.CFUNCTYPE(None, C.c_int, C.c_int, C.c_void_p)


def native_child(args):
    lib = C.CDLL(str(args.library.resolve()))
    lib.DenseCoreGetLastError.restype = C.c_char_p
    lib.InitEngineWithOptions.argtypes = [C.c_char_p, C.c_char_p, C.c_int, C.c_int, C.c_int, C.c_int, C.POINTER(InitOptions)]
    lib.InitEngineWithOptions.restype = C.c_void_p
    lib.FreeEngine.argtypes = [C.c_void_p]
    lib.FreeEngine.restype = None
    lib.DenseCoreGetRuntimePlan.argtypes = [C.c_void_p, C.POINTER(RuntimePlan)]
    lib.DenseCoreSetGenerationCompletionCallback.argtypes = [C.c_void_p, COMPLETE]
    lib.SubmitRequestWithTokenResults.argtypes = [C.c_void_p, C.c_char_p, C.c_int, C.c_float, C.c_float,
                                                  C.c_int, C.c_float, TOKEN, C.c_void_p]
    lib.SubmitRequestWithTokenResults.restype = C.c_int
    lib.CancelRequest.argtypes = [C.c_void_p, C.c_int]
    mode = 1 if args.mode == 'none' else 2
    options = InitOptions(C.sizeof(InitOptions), 1, mode, args.draft)
    handle = lib.InitEngineWithOptions(os.fsencode(args.model.resolve()), None, args.threads, -1, 0, 0, C.byref(options))
    if not handle:
        raise RuntimeError((lib.DenseCoreGetLastError() or b'initialization failed').decode(errors='replace'))
    records, guard = {}, threading.Lock()

    @TOKEN
    def token_callback(result, data):
        with guard:
            row = records[int(data)]
            item = result.contents
            if item.is_finished:
                row['terminal_events'] += 1
                row['done'].set()
            elif item.token_id >= 0:
                row['token_ids'].append(item.token_id)
                row['pieces'].append((item.text or b'').decode('utf-8', errors='replace'))
                row['first_token'].set()

    @COMPLETE
    def completed(count, reason, data):
        with guard:
            records[int(data)].update(generated_tokens=count, finish_reason=reason)

    def submit(case, repeat, cancel=False):
        with guard:
            key = len(records) + 1
            row = {'case': case['id'], 'repeat': repeat, 'token_ids': [], 'pieces': [], 'terminal_events': 0,
                   'done': threading.Event(), 'first_token': threading.Event()}
            records[key] = row
        started = time.monotonic()
        request = lib.SubmitRequestWithTokenResults(handle, case['prompt'].encode(),
                    max(512, args.max_tokens) if cancel else args.max_tokens, 0.0, 1.0, 0, 1.0,
                    token_callback, C.c_void_p(key))
        row['request_id'] = request
        if request < 0:
            raise RuntimeError((lib.DenseCoreGetLastError() or b'submit failed').decode(errors='replace'))
        if cancel:
            wait_for_first_token(row, args.timeout)
            row['cancel_status'] = lib.CancelRequest(handle, request)
        if not row['done'].wait(args.timeout):
            lib.CancelRequest(handle, request)
            raise TimeoutError('native request failed to terminate')
        row['wall_seconds'] = time.monotonic() - started
        return {key: value for key, value in row.items() if key not in ('done', 'first_token')}

    result = {'mode': args.mode, 'concurrency': args.child_concurrency, 'greedy': True,
              'native_seed_api': False, 'requests': []}
    try:
        plan = RuntimePlan()
        plan.struct_size, plan.version = C.sizeof(plan), 1
        if lib.DenseCoreGetRuntimePlan(handle, C.byref(plan)) != 0:
            raise RuntimeError('native runtime plan unavailable')
        result['runtime_plan'] = {name: getattr(plan, name) for name, _ in plan._fields_}
        result['runtime_plan']['spec_reason'] = plan.spec_reason.decode(errors='replace')
        if lib.DenseCoreSetGenerationCompletionCallback(handle, completed) != 0:
            raise RuntimeError('native completion metadata callback unavailable')
        jobs = [(case, repeat) for repeat in range(args.repeats) for case in NATIVE_CASES]
        with ThreadPoolExecutor(max_workers=args.child_concurrency) as pool:
            result['requests'] = list(pool.map(lambda job: submit(*job), jobs))
        for row in result['requests']: validate_native(row)
        probe = submit({'id': 'cancel', 'prompt': 'List every integer from 1 through 10000, one per line. Start now.'}, 0, True)
        followup = submit(CASES[0], args.repeats)
        result['requests'].append(followup)
        result['cancellation'] = {'request': probe, 'passed': probe.get('cancel_status') == 0 and
                probe.get('terminal_events') == 1 and probe.get('finish_reason') == 3 and
                probe.get('generated_tokens', 0) < max(512, args.max_tokens)}
    except Exception as error:
        result['error'] = str(error)
        write(args.out, result)
        raise
    finally:
        lib.FreeEngine(handle)
    write(args.out, result)


def http_request(base, case, stream, max_tokens, timeout):
    from compare_servers import validate_response
    body = {'model': 'local', 'messages': [{'role': 'user', 'content': case['prompt']}],
        'temperature': 0, 'max_tokens': max_tokens, 'stream': stream, 'cache_prompt': False,
        'chat_template_kwargs': {'enable_thinking': False}}
    if stream: body['stream_options'] = {'include_usage': True}
    raw = {'case': case['id'], 'stream': stream, 'body': body, 'events': []}
    request = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
    content, reasoning, finish, usage, done = [], [], None, {}, not stream
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw['status'] = response.status
            if stream:
                for line in response:
                    if not line.startswith(b'data:'): continue
                    payload = line[5:].strip()
                    if payload == b'[DONE]': done = True; break
                    event = json.loads(payload)
                    raw['events'].append({'seconds': time.monotonic() - started, 'data': event})
                    if event.get('error'): raise ValueError(str(event['error']))
                    if event.get('usage'): usage = event['usage']
                    for choice in event.get('choices', []):
                        delta = choice.get('delta', {})
                        content.append(delta.get('content') or '')
                        reasoning.append(delta.get('reasoning_content') or '')
                        finish = choice.get('finish_reason') or finish
            else:
                event = json.load(response)
                raw['response'] = event
                choice = event['choices'][0]
                content = [choice['message'].get('content') or '']
                reasoning = [choice['message'].get('reasoning_content') or '']
                finish, usage = choice.get('finish_reason'), event.get('usage', {})
        raw.update(done=done, content=''.join(content), reasoning=''.join(reasoning), finish_reason=finish, usage=usage)
        if not done: raise ValueError('stream missing [DONE]')
        raw['final_answer'] = validate_response(raw['content'], raw['reasoning'], finish, usage, case['expected'])
    except Exception as error:
        raw['error'] = str(error)
    raw['wall_seconds'] = time.monotonic() - started
    return raw


def http_mode(args, mode, out, environment):
    base = f'http://127.0.0.1:{args.port}'
    log = out / f'http-{mode}.log'
    with log.open('w') as output:
        process = subprocess.Popen([str(args.server.resolve()), 'serve', '-m', str(args.model.resolve()), '-t', str(args.threads),
            '--host', '127.0.0.1', '--port', str(args.port), '--spec-type', mode,
            '--spec-draft-n-max', str(args.draft)], env=environment, stdout=output, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + args.timeout
            while True:
                if process.poll() is not None: raise RuntimeError(f'{mode} server exited before ready; see {log}')
                try:
                    with urllib.request.urlopen(base + '/health/ready', timeout=2) as ready:
                        if ready.status == 200: break
                except OSError: pass
                if time.monotonic() >= deadline: raise TimeoutError(f'{mode} server readiness deadline exceeded')
                time.sleep(0.2)
            rows = []
            for concurrency in args.concurrency:
                for stream in (False, True):
                    jobs = [case for _ in range(args.repeats) for case in CASES]
                    with ThreadPoolExecutor(max_workers=concurrency) as pool:
                        requests = list(pool.map(lambda case: http_request(base, case, stream, args.http_max_tokens, args.timeout), jobs))
                    rows.append({'mode': mode, 'concurrency': concurrency, 'stream': stream, 'requests': requests})
            # Disconnect a live stream, then prove readiness and a completed follow-up.
            connection = http.client.HTTPConnection('127.0.0.1', args.port, timeout=args.timeout)
            probe = {'model': 'local', 'messages': [{'role': 'user', 'content': 'List the integers from 1 through 10000.'}],
                     'stream': True, 'max_tokens': args.http_max_tokens, 'temperature': 0}
            connection.request('POST', '/v1/chat/completions', json.dumps(probe), {'Content-Type': 'application/json'})
            response = connection.getresponse()
            saw_event, probe_events = False, []
            while line := response.readline():
                if not line.startswith(b'data:'): continue
                payload = line[5:].strip()
                if payload == b'[DONE]': break
                event = json.loads(payload)
                probe_events.append(event)
                if any((choice.get('delta', {}).get('content') or choice.get('delta', {}).get('reasoning_content'))
                       for choice in event.get('choices', [])):
                    saw_event = True; break
            response.close(); connection.close()
            with urllib.request.urlopen(base + '/health/ready', timeout=5) as ready:
                ready_status = ready.status
            followup = http_request(base, CASES[0], True, args.http_max_tokens, args.timeout)
            result = {'groups': rows, 'disconnect_probe': {'saw_event': saw_event, 'events': probe_events,
                'ready_after_disconnect': ready_status, 'followup': followup,
                'scope': 'disconnect after visible/reasoning token and recovery only; native cancellation checked separately'}}
            write(out / f'http-{mode}.json', result)
            return result
        finally:
            process.terminate()
            try: process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=10)


def parent(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    library, model = args.library.resolve(), args.model.resolve()
    manifest = {'sampling': {'temperature': 0, 'top_p': 1, 'top_k': 0, 'repetition_penalty': 1, 'kv_type': 'FP16'},
        'model': str(model), 'model_sha256': digest(model), 'library': str(library),
        'library_sha256': digest(library), 'host': platform.uname()._asdict(),
        'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
        'greedy': True, 'native_seed_api': False, 'draft_n_max': args.draft, 'native': [], 'http': []}
    manifest['harness_sha256'] = digest(Path(__file__).resolve())
    manifest['dependencies'] = {str(path): digest(path) for path in sorted({
        entry.resolve() for entry in library.parent.glob('lib*.so*') if entry.is_file()})}
    cache = library.parent / 'CMakeCache.txt'
    if cache.exists(): manifest['cmake_cache_sha256'] = digest(cache)
    cpuinfo = Path('/proc/cpuinfo')
    if cpuinfo.exists():
        manifest['cpu_profile'] = [line for line in cpuinfo.read_text().split('\n\n')[0].splitlines()
                                   if line.startswith(('model name', 'flags', 'CPU implementer', 'CPU part', 'Features'))]
    if args.server: manifest['server_sha256'] = digest(args.server)
    if args.baseline_summary:
        if 1 in args.concurrency:
            raise ValueError('--baseline-summary requires only new concurrency cells, excluding c1')
        manifest['native'] = load_c1_baseline(args.baseline_summary, manifest, args.context, args.threads)
        manifest['baseline_summary'] = {'path': str(args.baseline_summary.resolve()),
                                        'sha256': digest(args.baseline_summary)}

    write(out / 'summary.json', manifest)
    environment = dict(os.environ, DENSECORE_MAX_SEQ_LEN=str(args.context),
        DENSECORE_MAX_NUM_SEQS=str(max(args.concurrency)), DENSECORE_PREFIX_CACHE_REUSE='0',
        LD_LIBRARY_PATH=str(library.parent) + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
    for mode in ('none', 'mtp'):
        for concurrency in args.concurrency:
            result = out / f'native-{mode}-c{concurrency}.json'
            command = [sys.executable, str(Path(__file__).resolve()), '--native-child', '--mode', mode,
                '--child-concurrency', str(concurrency), '--library', str(library), '--model', str(model),
                '--threads', str(args.threads), '--draft', str(args.draft), '--repeats', str(args.repeats),
                '--max-tokens', str(args.max_tokens), '--timeout', str(args.timeout), '--out', str(result)]
            native_environment = dict(environment)
            if platform.system() == 'Linux':
                # Match the SDK's startup preload workaround for initial-exec
                # TLS in DenseCore/libgomp; late ctypes dlopen can exhaust TLS.
                existing = native_environment.get('LD_PRELOAD', '').split()
                native_environment['LD_PRELOAD'] = ' '.join(dict.fromkeys([str(library), *existing]))
            with (out / f'native-{mode}-c{concurrency}.log').open('w') as output:
                execution = subprocess.run(command, env=native_environment, stdout=output, stderr=subprocess.STDOUT,
                    timeout=args.timeout * (len(NATIVE_CASES) * args.repeats + 5))
            if execution.returncode or not result.exists():
                manifest['error'] = f'native {mode}/c{concurrency} failed, see preserved log'
                manifest['passed'] = False
                if result.exists(): manifest['failed_native_result'] = json.loads(result.read_text())
                write(out / 'summary.json', manifest)
                return 1
            native_row = json.loads(result.read_text())
            native_row['mtp_counters'] = extract_mtp_counters((out / f'native-{mode}-c{concurrency}.log').read_text())
            manifest['native'].append(native_row)
            write(out / 'summary.json', manifest)
        if args.server:
            environment['DENSECORE_MAX_NUM_SEQS'] = str(max(args.concurrency))
            manifest['http'].append(http_mode(args, mode, out, environment))
            write(out / 'summary.json', manifest)
    failures = compare_native(manifest['native'])
    if args.require_batched_mtp:
        for row in manifest['native']:
            if row['mode'] != 'mtp': continue
            counters = row['mtp_counters']
            if counters['maximum_target_width'] <= 1 or counters['target_rollbacks'] <= 0:
                failures.append(f"mtp/c{row['concurrency']}: actual batched verification/rollback not observed")
            if not counters['accepted'] or not counters['rejected']:
                failures.append(f"mtp/c{row['concurrency']}: both acceptance and rejection were not exercised")
            for request in row['requests']:
                if request['case'] == 'long_parity' and request['generated_tokens'] < 32:
                    failures.append(f"mtp/c{row['concurrency']}: long parity case produced fewer than 32 tokens")

    finals = {}
    for mode_result in manifest['http']:
        for group in mode_result['groups']:
            for row in group['requests']:
                if row.get('error'): failures.append(f"HTTP {group['mode']}/c{group['concurrency']}: {row['error']}")
                elif finals.setdefault(row['case'], row['final_answer']) != row['final_answer']:
                    failures.append(f"HTTP final answer differs across mode/stream/concurrency for {row['case']}")
        if (mode_result['disconnect_probe']['followup'].get('error') or not mode_result['disconnect_probe']['saw_event']
                or mode_result['disconnect_probe']['ready_after_disconnect'] != 200):
            failures.append('HTTP disconnect/recovery probe failed')
    if digest(library) != manifest['library_sha256'] or digest(model) != manifest['model_sha256']:
        failures.append('model/library bytes changed during verification')
    for path, expected in manifest['dependencies'].items():
        if digest(path) != expected: failures.append(f'dependency changed during verification: {path}')
    if args.server and digest(args.server) != manifest['server_sha256']:
        failures.append('server bytes changed during verification')
    manifest.update(passed=not failures, failures=failures, http_verified=bool(args.server),
        qualification='correctness only; no automatic MTP or performance promotion')
    write(out / 'summary.json', manifest)
    print(f"{'PASS' if not failures else 'FAIL'}: {out / 'summary.json'} ({len(failures)} failures)")
    return int(bool(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--server', type=Path)
    parser.add_argument('--build-lock', type=Path, help='Hold the existing build/model flock throughout verification (Linux)')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--baseline-summary', type=Path, help='reuse passed c1 native pair with identical model/binary/profile provenance')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--context', type=int, default=2048)
    parser.add_argument('--concurrency', type=int, nargs='+', choices=(1, 2, 4, 8), default=[1, 2, 4])
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--draft', type=int, choices=(1, 2, 3), default=2)
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--require-batched-mtp', action='store_true', help='require observed acceptance, rejection, K+1 graphs and rollback')
    parser.add_argument('--http-max-tokens', type=int, default=1024)
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--port', type=int, default=18089)
    parser.add_argument('--native-child', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--mode', choices=('none', 'mtp'), help=argparse.SUPPRESS)
    parser.add_argument('--child-concurrency', type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.require_batched_mtp and args.max_tokens < 128: parser.error('--require-batched-mtp needs --max-tokens >= 128')
    if args.repeats < 1 or args.threads < 1 or args.context < 1: parser.error('repeats, threads and context must be positive')
    if args.native_child:
        native_child(args)
        return 0
    if 1 not in args.concurrency: parser.error('concurrency must include 1 for the isolation baseline')
    args.concurrency = sorted(set(args.concurrency))
    try:
        if args.build_lock:
            import fcntl
            with args.build_lock.open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                return parent(args)
        return parent(args)
    except Exception as error:
        path = args.out.resolve() / 'summary.json'
        partial = json.loads(path.read_text()) if path.exists() else {}
        partial.update(passed=False, error=str(error))
        write(path, partial)
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
