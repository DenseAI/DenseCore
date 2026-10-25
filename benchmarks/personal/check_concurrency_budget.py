#!/usr/bin/env python3
"""Real native c8 correctness/admission qualification; never a speed/quality claim.

Requires an explicitly qualified artifact. No downloading/building is performed.
Run one context per invocation; do not retry at a shorter context after failure.
"""
import argparse
import ctypes as C
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import traceback

from check_mtp_parity import COMPLETE, TOKEN, InitOptions, RuntimePlan, digest, write

MODEL_SHA = 'f5b14da98939b60bbe1019a964eba656407e1e0b64f1fe3003ff6d650e93bfec'
PROMPT = ('Write every integer from 1 through 200, one integer per line. '
          'Do not skip any integers. Start with 1.')
TEXT = C.CFUNCTYPE(None, C.c_char_p, C.c_int, C.c_void_p)


class DetailedMetrics(C.Structure):
    _fields_ = [('active_requests', C.c_int), ('total_requests', C.c_long),
        ('completed_requests', C.c_long), ('failed_requests', C.c_long), ('pending_requests', C.c_int),
        ('total_tokens_generated', C.c_long), ('total_prompt_tokens', C.c_long),
        *[(key, C.c_float) for key in ('tokens_per_second', 'avg_time_to_first_token',
          'p50_time_to_first_token', 'p90_time_to_first_token', 'p99_time_to_first_token',
          'avg_inter_token_latency', 'p50_inter_token_latency', 'p90_inter_token_latency',
          'p99_inter_token_latency', 'avg_queue_wait_time', 'p99_queue_wait_time')],
        ('kv_cache_usage_blocks', C.c_int), ('kv_cache_total_blocks', C.c_int),
        ('kv_cache_usage_percent', C.c_float), ('avg_batch_size', C.c_float),
        ('current_batch_size', C.c_int), ('oom_errors', C.c_int), ('timeout_errors', C.c_int)]


def rejection_valid(row):
    match = re.search(r'required_bytes=(\d+) available_bytes=(\d+)', row.get('submit_error', ''))
    return (row.get('request_id', 0) < 0 and row.get('callback_events') == 0
            and row.get('completion_events') == 0 and bool(match)
            and int(match[1]) > int(match[2]))


def pressure_errors(probes):
    rejected = [row for row in probes if row.get('request_id', 0) < 0]
    if not rejected:
        return ['pressure not established: both bounded oversized probes were admitted']
    return [f"{row['label']}: rejection lacks required/available or called back"
            for row in rejected if not rejection_valid(row)]


def parity_errors(rows, baseline):
    errors = []
    for row in rows:
        if (row.get('terminal_events'), row.get('completion_events'), row.get('generated_tokens'),
            row.get('finish_reason')) != (1, 1, 128, 2):
            errors.append(f"{row['label']}: expected exactly one length terminal with 128 native samples")
        if len(row.get('token_ids', [])) != row.get('generated_tokens'):
            errors.append(f"{row['label']}: callback IDs differ in count from native samples")
        if row.get('token_ids') != baseline:
            errors.append(f"{row['label']}: sampled IDs differ from cap8/c1 baseline")
    return errors


def plan_errors(plan, context):
    expected = {'effective_context': context, 'effective_concurrency': 8, 'effective_threads': 8,
                'spec_selected': 1}
    return [f'runtime {key}: expected {value}, got {plan.get(key)}'
            for key, value in expected.items() if plan.get(key) != value]


def timed_daemon(call, timeout):
    """Bound a C call without joining a stalled native producer forever."""
    done, result = threading.Event(), {}
    def target():
        try:
            result['value'] = call()
        except BaseException as error:
            result['error'] = error
        finally:
            done.set()
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    if not done.wait(timeout):
        raise TimeoutError('native call did not return; child must exit without concurrent FreeEngine')
    if 'error' in result:
        raise result['error']
    return result.get('value')


def child(args):
    lib = C.CDLL(str(args.library))
    lib.DenseCoreGetLastError.restype = C.c_char_p
    lib.InitEngineWithOptions.argtypes = [C.c_char_p, C.c_char_p, C.c_int, C.c_int, C.c_int,
                                         C.c_int, C.POINTER(InitOptions)]
    lib.InitEngineWithOptions.restype = C.c_void_p
    lib.FreeEngine.argtypes = [C.c_void_p]
    lib.FreeEngine.restype = None
    lib.DenseCoreGetRuntimePlan.argtypes = [C.c_void_p, C.POINTER(RuntimePlan)]
    lib.DenseCoreSetGenerationCompletionCallback.argtypes = [C.c_void_p, COMPLETE]
    lib.SubmitRequestWithTokenResults.argtypes = [C.c_void_p, C.c_char_p, C.c_int, C.c_float,
                                                 C.c_float, C.c_int, C.c_float, TOKEN, C.c_void_p]
    lib.SubmitRequestWithTokenResults.restype = C.c_int
    lib.SubmitRequestWithSampling.argtypes = [C.c_void_p, C.c_char_p, C.c_int, C.c_float,
        C.c_float, C.c_int, C.c_float, C.POINTER(C.c_char_p), C.c_int, TEXT, C.c_void_p]
    lib.SubmitRequestWithSampling.restype = C.c_int
    lib.CancelRequest.argtypes = [C.c_void_p, C.c_int]
    lib.CancelRequest.restype = C.c_int
    lib.GetDetailedMetrics.argtypes = [C.c_void_p]
    lib.GetDetailedMetrics.restype = DetailedMetrics
    rows, samples, errors = {}, [], []
    mu, gate, gate_entered, stop_poll = threading.Lock(), threading.Event(), threading.Event(), threading.Event()
    gate.set()
    options = InitOptions(C.sizeof(InitOptions), 1, 1, 2)
    started = time.monotonic()
    handle = lib.InitEngineWithOptions(os.fsencode(args.model), None, 8, -1, 0, 0, C.byref(options))
    if not handle:
        raise RuntimeError((lib.DenseCoreGetLastError() or b'init failed').decode(errors='replace'))
    report = {'passed': False, 'initialization_seconds': time.monotonic()-started,
        'unknown': ['actual recurrent-state allocation count', 'admission remaining bytes',
                    'worker-deferred queue count (public pending metric excludes it)'],
        'metrics_semantics': 'pending is approximate native queue count; current_batch_size is active-list size',
        'caller_oversize_logical_bytes_per_probe': 8*1024*1024,
        'caller_buffer_note': 'Python ASCII str plus encoded bytes can transiently duplicate each bounded8MiB input; not native-owned memory',
        'requests': [], 'metrics': samples, 'failures': errors}

    def save():
        with mu:
            report['requests'] = [{k: v for k, v in row.items() if not isinstance(v, threading.Event)}
                                  for row in rows.values()]
            write(args.out, report)

    def observe():
        m = lib.GetDetailedMetrics(handle)
        return {'elapsed': time.monotonic()-started,
                **{key: getattr(m, key) for key, _ in m._fields_},
                'kv_free_blocks': m.kv_cache_total_blocks-m.kv_cache_usage_blocks}

    def poll():
        while not stop_poll.wait(.01):
            samples.append(observe())
    monitor = threading.Thread(target=poll, daemon=True)

    def receive(data, terminal, token_id=None, text=b''):
        key = int(data)
        with mu:
            row = rows[key]
            row['callback_events'] += 1
            if terminal:
                row['terminal_events'] += 1
                row['done'].set()
            elif token_id is not None and token_id >= 0:
                row['token_ids'].append(token_id)
                row['first'].set()
            elif text:
                row['first'].set()
            blocked = row['gate'] and not terminal
        if blocked:
            gate_entered.set()
            if not gate.wait(args.timeout):
                errors.append('callback gate timed out')

    @TOKEN
    def token(item, data):
        result = item.contents
        receive(data, result.is_finished, result.token_id, result.text)

    @TEXT
    def text(piece, terminal, data):
        receive(data, terminal, text=piece)

    @COMPLETE
    def completion(count, reason, data):
        with mu:
            row = rows[int(data)]
            row['completion_events'] += 1
            row.update(generated_tokens=count, finish_reason=reason)

    lib.DenseCoreSetGenerationCompletionCallback(handle, completion)

    def submit(label, prompt=PROMPT, max_tokens=128, stop=None, blocked=False):
        with mu:
            key = len(rows)+1
            row = dict(label=label, key=key, token_ids=[], callback_events=0, terminal_events=0,
                       completion_events=0, structured=stop is None, done=threading.Event(), first=threading.Event(), gate=blocked)
            rows[key] = row
        # Arguments/stop storage stay alive until the synchronous C submission returns.
        if stop is None:
            rid = lib.SubmitRequestWithTokenResults(handle, prompt.encode(), max_tokens, 0., 1., 1, 1., token, key)
        else:
            stops = (C.c_char_p*2)(stop.encode(), None)
            rid = lib.SubmitRequestWithSampling(handle, prompt.encode(), max_tokens, 0., 1., 1, 1., stops, 0, text, key)
        row['request_id'] = rid
        row['submit_error'] = (lib.DenseCoreGetLastError() or b'').decode(errors='replace') if rid < 0 else ''
        return row

    def wait(row, canceled=False):
        if row['request_id'] < 0:
            raise RuntimeError(f"{row['label']}: unexpected synchronous rejection: {row['submit_error']}")
        if not row['done'].wait(args.timeout):
            raise TimeoutError(f"{row['label']}: terminal waiter stranded")
        if (row['terminal_events'], row['completion_events']) != (1, 1):
            errors.append(f"{row['label']}: terminal/completion not exactly once")
        if row['structured'] and len(row['token_ids']) != row.get('generated_tokens'):
            errors.append(f"{row['label']}: raw sampled IDs/native usage mismatch")
        if canceled and row.get('finish_reason') != 3:
            errors.append(f"{row['label']}: cancellation not reported as native error terminal")

    def wave(label):
        barrier, output, failures = threading.Barrier(8), [], []
        def one(slot):
            try:
                barrier.wait(timeout=args.timeout)
                output.append(timed_daemon(lambda: submit(f'{label}-{slot}'), args.timeout))
            except BaseException as error:
                failures.append(str(error))
        threads = [threading.Thread(target=one, args=(i,), daemon=True) for i in range(8)]
        for thread in threads: thread.start()
        for thread in threads: thread.join(args.timeout+1)
        if failures or any(thread.is_alive() for thread in threads) or len(output) != 8:
            raise RuntimeError(f'{label}: bounded barrier submission failed: {failures}')
        for row in output: wait(row)
        return output

    safe_free = True
    try:
        plan = RuntimePlan(C.sizeof(RuntimePlan), 1)
        if lib.DenseCoreGetRuntimePlan(handle, C.byref(plan)) != 0:
            raise RuntimeError('runtime plan unavailable')
        report['runtime_plan'] = {key: (getattr(plan, key).decode(errors='replace') if key == 'spec_reason'
                                      else getattr(plan, key)) for key, _ in plan._fields_}
        errors.extend(plan_errors(report['runtime_plan'], args.context))
        if errors: raise RuntimeError('effective workload differs from explicit requested profile')
        monitor.start()
        baseline = timed_daemon(lambda: submit('baseline-cap8-c1'), args.timeout)
        wait(baseline)
        errors.extend(parity_errors([baseline], baseline['token_ids']))
        errors.extend(parity_errors(wave('c8-first'), baseline['token_ids']))
        save()

        target = timed_daemon(lambda: submit('active-cancel', max_tokens=args.context), args.timeout)
        peer = timed_daemon(lambda: submit('cancel-peer'), args.timeout)
        if not target['first'].wait(args.timeout): raise TimeoutError('active cancel produced no sample')
        target['cancel_status'] = lib.CancelRequest(handle, target['request_id'])
        if target['cancel_status'] != 0: errors.append('active cancellation was not accepted')
        wait(target, canceled=True)
        wait(peer)
        errors.extend(parity_errors([peer], baseline['token_ids']))

        # Hold asynchronous delivery while a bounded set of producers retains payload leases.
        gate.clear()
        held = [timed_daemon(lambda i=i: submit(f'held-{i}', max_tokens=256, blocked=True), args.timeout)
                for i in range(16)]
        if not gate_entered.wait(args.timeout): raise TimeoutError('no callback reached pressure gate')
        samples.append(observe())
        report['pressure_snapshot'] = samples[-1]
        held[-1]['cancel_status'] = lib.CancelRequest(handle, held[-1]['request_id'])
        report['pending_cancel_attempt_status'] = held[-1]['cancel_status']
        if held[-1]['cancel_status'] != 0:
            errors.append('queued cancellation was not established before native completion')
        for row in held:
            if row['request_id'] < 0:
                errors.append(f"{row['label']}: small bounded admission rejected")
        # Each client buffer is bounded at8MiB; never grow buffers until native/physical OOM.
        probes = [timed_daemon(lambda: submit('oversize-stop', stop='X'*(8*1024*1024)), args.timeout),
                  timed_daemon(lambda: submit('oversize-prompt', prompt='X'*(8*1024*1024)), args.timeout)]
        for row in probes:
            if row['request_id'] >= 0:
                row['oversize_outcome'] = 'accepted; bounded caller payload safely canceled/drained'
                row['cancel_status'] = lib.CancelRequest(handle, row['request_id'])
            else:
                row['oversize_outcome'] = 'synchronously rejected'
        errors.extend(pressure_errors(probes))
        report['pressure_established'] = any(rejection_valid(row) for row in probes)
        # Keep the original queued-cancel status; do not overwrite it with a second Cancel result.
        gate.set()
        for row in held:
            if row['request_id'] > 0: wait(row, canceled=row.get('cancel_status') == 0)
        for row in probes:
            if row['request_id'] > 0: wait(row, canceled=row.get('cancel_status') == 0)
        # Retain all rejected callback rows until after healthy recovery to catch late callbacks.
        errors.extend(parity_errors(wave('c8-recovery'), baseline['token_ids']))
        for row in probes:
            if row['request_id'] < 0 and not rejection_valid(row):
                errors.append(f"{row['label']}: rejected request later called back")
        report['max_observed_active'] = max((m['active_requests'] for m in samples), default=0)
        if report['max_observed_active'] != 8:
            errors.append('native active metric did not demonstrate exactly8 active requests')
        if any(m['active_requests'] > 8 for m in samples): errors.append('native active requests exceeded8')
        report['passed'] = not errors
    except BaseException:
        safe_free = False  # A daemon might still be inside C; never race FreeEngine with it.
        errors.append(traceback.format_exc())
    finally:
        gate.set()
        stop_poll.set()
        if monitor.ident:
            monitor.join(args.timeout)
            if monitor.is_alive():
                safe_free = False
                report['passed'] = False
                errors.append('metrics reader still in native call; refusing concurrent FreeEngine')
        save()
        if safe_free:
            try:
                timed_daemon(lambda: lib.FreeEngine(handle), args.timeout)
                # Native join also proves no late terminal dispatch occurred.
                for row in rows.values():
                    if row.get('request_id', -1) > 0 and (row['terminal_events'], row['completion_events']) != (1, 1):
                        errors.append(f"{row['label']}: post-drain terminal/completion not exactly once")
                report['passed'] = not errors
                save()
            except BaseException:
                report['passed'] = False
                errors.append(traceback.format_exc())
                save()
                os._exit(1)
        else:
            # Child exit destroys native process state; parent retains logs and releases shared lock.
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
    return int(not report['passed'])


def parent(args):
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {'passed': False, 'qualification': 'c8 native correctness/admission; no speed/final-answer claim'}
    with args.build_lock.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        artifact = json.loads(args.artifact.read_text())
        expected = artifact['files']
        for path, sha in expected.items():
            if digest(path) != sha: raise RuntimeError(f'qualified artifact changed: {path}')
        if str(args.library.resolve()) not in {str(Path(p).resolve()) for p in expected}:
            raise RuntimeError('library not owned by qualified artifact')
        if digest(args.model) != MODEL_SHA: raise RuntimeError('README pinned model SHA mismatch')
        revision = subprocess.check_output(['git','-C',artifact['source'],'rev-parse','HEAD'], text=True).strip()
        if revision != artifact['revision']: raise RuntimeError('qualified source revision changed')
        if subprocess.check_output(['git','-C',artifact['source'],'status','--porcelain','--untracked-files=no'], text=True):
            raise RuntimeError('qualified source has tracked modifications')
        dependencies = {str(Path(__file__).resolve()): digest(__file__),
                        str(Path(__file__).with_name('check_mtp_parity.py').resolve()):
                            digest(Path(__file__).with_name('check_mtp_parity.py'))}
        manifest.update(artifact=str(args.artifact), artifact_sha256=digest(args.artifact),
            source=artifact['source'], revision=revision, artifact_files=expected, harnesses=dependencies,
            model=str(args.model), model_sha256=MODEL_SHA, cpu_profile=Path('/proc/cpuinfo').read_text(),
            cpu_affinity=sorted(os.sched_getaffinity(0)), context=args.context, concurrency=8, threads=8,
            max_tokens=128, spec_type='none', sampling={'temperature':0,'top_p':1,'top_k':1,'repeat_penalty':1},
            python=sys.version, platform=list(os.uname()))
        env = dict(os.environ, DENSECORE_MAX_SEQ_LEN=str(args.context), DENSECORE_MAX_NUM_SEQS='8',
            DENSECORE_PREFIX_CACHE_REUSE='0', DENSECORE_QWEN35_PREFIX_CACHE_REUSE='0',
            DENSECORE_LOG_DECODE_RUNTIME_STATS='1', DENSECORE_DECODE_GRAPH_CACHE='1',
            LD_LIBRARY_PATH=str(args.library.parent)+':'+os.environ.get('LD_LIBRARY_PATH',''))
        env['LD_PRELOAD'] = ' '.join(dict.fromkeys([str(args.library), *env.get('LD_PRELOAD','').split()]))
        manifest['environment'] = {key:value for key,value in env.items() if key.startswith(('DENSECORE_','LD_','OMP_'))}
        write(args.out/'manifest.json', manifest)
        command = [sys.executable,str(Path(__file__).resolve()),'--child','--artifact',str(args.artifact),
            '--library',str(args.library),'--model',str(args.model),'--out',str(args.out/'native.json'),
            '--context',str(args.context),'--timeout',str(args.timeout)]
        with (args.out/'native.log').open('w') as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=args.lifetime_timeout)
        manifest['raw_decode_batch_stats_lines'] = [line for line in (args.out/'native.log').read_text(errors='replace').splitlines()
            if 'decode_physical_batch_hist[' in line]
        manifest['child_returncode'] = result.returncode
        manifest['passed'] = result.returncode == 0 and json.loads((args.out/'native.json').read_text()).get('passed') is True
        for path, sha in {**expected, **dependencies, str(args.model):MODEL_SHA}.items():
            if digest(path) != sha:
                manifest['passed'] = False
                manifest.setdefault('failures',[]).append(f'bytes changed during qualification: {path}')
        write(args.out/'manifest.json',manifest)
        return int(not manifest['passed'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact',type=Path,required=True)
    parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--context',type=int,choices=(512,1024),default=512)
    parser.add_argument('--timeout',type=float,default=180)
    parser.add_argument('--lifetime-timeout',type=float,default=1800)
    parser.add_argument('--build-lock',type=Path,default=Path('/home/jaewook/dc/build.lock'))
    parser.add_argument('--child',action='store_true',help=argparse.SUPPRESS)
    args = parser.parse_args()
    for key in ('artifact','library','model','out','build_lock'):
        setattr(args,key,getattr(args,key).resolve())
    return child(args) if args.child else parent(args)


if __name__ == '__main__':
    sys.exit(main())
