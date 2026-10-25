#!/usr/bin/env python3
"""Real probabilistic MTP integration smoke; never a distribution-equivalence test.

Public C/HTTP expose temperature, top-k/p and repetition penalty, but not min-p
or a native seed. Min-p is explicitly the native default0; never send an ignored
min_p field. Formal acceptance/residual tests and greedy128 parity are separate.
"""
import argparse
import ctypes as C
from concurrent.futures import ThreadPoolExecutor
import functools
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import threading
import time
import traceback

import check_mtp_parity as protocol
import compare_servers
from compare_servers import final_answer


def validate_final(text, case):
    text.encode('utf-8', errors='strict')
    if '\ufffd' in text:
        raise ValueError('replacement character in output')
    final = final_answer(text, '')
    if not final or not re.search(case['expected'], final, re.I):
        raise ValueError('empty/reasoning-only or incorrect final answer')
    if len(re.findall(r'\b\w+\b', final)) < case.get('min_words', 1):
        raise ValueError('multiword answer too short')
    if case.get('sentence') and not re.search(r'[.!?。！？](?:["\'»”])?$', final):
        raise ValueError('sentence lacks complete final punctuation')
    return final


def sampling(args):
    return {'temperature': args.temperature, 'top_p': args.top_p, 'top_k': args.top_k,
            'repetition_penalty': args.repetition_penalty}


def validate_native(row, case):
    protocol.validate_native(row)
    if row.get('finish_reason') != 1:
        raise ValueError('length-truncated native answer')
    if row.get('completion_events') != 1 or row.get('utf8_errors'):
        raise ValueError('duplicate/missing completion metadata or invalid UTF-8')
    row['final_answer'] = validate_final(''.join(row['pieces']), case)
    row['termination_contract'] = 'native finish1 stop/EOS; IDs include committed stop token when callback exposes it'


def native_child(args, cases):
    lib = C.CDLL(str(args.library.resolve()))
    lib.DenseCoreGetLastError.restype = C.c_char_p
    lib.InitEngineWithOptions.argtypes = [C.c_char_p, C.c_char_p, C.c_int, C.c_int, C.c_int, C.c_int,
                                         C.POINTER(protocol.InitOptions)]
    lib.InitEngineWithOptions.restype = C.c_void_p
    lib.FreeEngine.argtypes = [C.c_void_p]
    lib.FreeEngine.restype = None
    lib.DenseCoreGetRuntimePlan.argtypes = [C.c_void_p, C.POINTER(protocol.RuntimePlan)]
    lib.DenseCoreSetGenerationCompletionCallback.argtypes = [C.c_void_p, protocol.COMPLETE]
    lib.SubmitRequestWithTokenResults.argtypes = [C.c_void_p, C.c_char_p, C.c_int, C.c_float, C.c_float,
                                                 C.c_int, C.c_float, protocol.TOKEN, C.c_void_p]
    lib.SubmitRequestWithTokenResults.restype = C.c_int
    lib.CancelRequest.argtypes = [C.c_void_p, C.c_int]
    options = protocol.InitOptions(C.sizeof(protocol.InitOptions), 1, 1 if args.mode == 'none' else 2, args.draft)
    handle = lib.InitEngineWithOptions(os.fsencode(args.model.resolve()), None, args.threads, -1, 0, 0, C.byref(options))
    if not handle:
        raise RuntimeError((lib.DenseCoreGetLastError() or b'init failed').decode('utf-8', 'strict'))
    records, lock = {}, threading.Lock()

    @protocol.TOKEN
    def token_callback(result, data):
        with lock:
            row, item = records[int(data)], result.contents
            if item.is_finished:
                row['terminal_events'] += 1
                row['done'].set()
            elif item.token_id >= 0:
                row['token_ids'].append(item.token_id)
                try:
                    row['pieces'].append((item.text or b'').decode('utf-8', 'strict'))
                except UnicodeDecodeError as error:
                    row['utf8_errors'].append(str(error))
                row['first_token'].set()

    @protocol.COMPLETE
    def completed(count, reason, data):
        with lock:
            row = records[int(data)]
            row.update(generated_tokens=count, finish_reason=reason)
            row['completion_events'] += 1

    def submit(case, repeat, cancel=False):
        with lock:
            key = len(records) + 1
            row = {'case': case['id'], 'repeat': repeat, 'token_ids': [], 'pieces': [], 'utf8_errors': [],
                   'terminal_events': 0, 'completion_events': 0,
                   'done': threading.Event(), 'first_token': threading.Event()}
            records[key] = row
        request_id = lib.SubmitRequestWithTokenResults(handle, case['prompt'].encode('utf-8'), args.max_tokens,
                     args.temperature, args.top_p, args.top_k, args.repetition_penalty,
                     token_callback, C.c_void_p(key))
        if request_id < 0:
            raise RuntimeError((lib.DenseCoreGetLastError() or b'submit failed').decode('utf-8', 'strict'))
        row['request_id'] = request_id
        if cancel:
            protocol.wait_for_first_token(row, args.timeout)
            row['cancel_status'] = lib.CancelRequest(handle, request_id)
        if not row['done'].wait(args.timeout):
            lib.CancelRequest(handle, request_id)
            raise TimeoutError('native probabilistic request did not terminate')
        return {key: value for key, value in row.items() if key not in ('done', 'first_token')}

    result = {'mode': args.mode, 'concurrency': args.child_concurrency, 'sampling': sampling(args), 'requests': []}
    try:
        plan = protocol.RuntimePlan()
        plan.struct_size, plan.version = C.sizeof(plan), 1
        if lib.DenseCoreGetRuntimePlan(handle, C.byref(plan)) != 0:
            raise RuntimeError('runtime plan unavailable')
        result['runtime_plan'] = {name: getattr(plan, name) for name, _ in plan._fields_}
        result['runtime_plan']['spec_reason'] = plan.spec_reason.decode('utf-8', 'strict')
        if lib.DenseCoreSetGenerationCompletionCallback(handle, completed) != 0:
            raise RuntimeError('completion metadata callback unavailable')
        jobs = [(case, repeat) for repeat in range(args.repeats) for case in cases]
        with ThreadPoolExecutor(max_workers=args.child_concurrency) as pool:
            result['requests'] = list(pool.map(lambda job: submit(*job), jobs))
        by_id = {case['id']: case for case in cases}
        for row in result['requests']:
            validate_native(row, by_id[row['case']])
        cancel_case = {'id': 'cancel', 'prompt': 'List every integer from 1 through 10000, one per line. Start now.'}
        probe = submit(cancel_case, 0, True)
        recovery = submit(cases[0], args.repeats)
        validate_native(recovery, cases[0])
        result['requests'].append(recovery)
        passed = probe.get('cancel_status') == 0 and probe.get('finish_reason') == 3 and \
                 probe['terminal_events'] == probe['completion_events'] == 1 and not probe['utf8_errors'] and \
                 probe['generated_tokens'] == len(probe['token_ids']) and probe['generated_tokens'] < args.max_tokens
        result['cancellation'] = {'passed': passed, 'request': probe,
                                  'scope': 'cancel after first committed token, not necessarily first visible text'}
    except Exception as error:
        result['error'] = str(error)
        raise
    finally:
        lib.FreeEngine(handle)
        if result.get('error'):
            result['partial_requests'] = [{key: value for key, value in row.items()
                                           if key not in ('done', 'first_token')} for row in records.values()]
        protocol.write(args.out, result)


def http_request(base, case, stream, max_tokens, timeout, probability):
    row = protocol.http_request(base, case, stream, max_tokens, timeout, probability)
    if not row.get('error'):
        try:
            row.get('reasoning', '').encode('utf-8', 'strict')
            row['final_answer'] = validate_final(row['content'], case)
            if case.get('stop'):
                if any(stop in row['content'] for stop in case['stop']):
                    raise ValueError('requested stop text leaked')
                row['stop_attribution'] = 'API finish=stop does not distinguish EOS versus string stop'
        except Exception as error:
            row['error'] = str(error)
    return row


def branch_coverage(counters):
    return {'batched_target': counters['maximum_target_width'] > 1,
            'acceptance': counters['accepted'] > 0, 'rejection': counters['rejected'] > 0,
            'rollback': counters['target_rollbacks'] > 0}


def verify_artifacts(args):
    document = json.loads(args.artifact_manifest.read_text())
    expected = document.get('files', {})
    for path in [args.library, args.server]:
        if path is None:
            continue
        key = str(path.resolve())
        if key not in expected or protocol.digest(path) != expected[key]:
            raise ValueError(f'canonical artifact mismatch: {key}')
    if protocol.digest(args.model) != args.expected_model_sha256:
        raise ValueError('pinned model SHA mismatch')
    return {'path': str(args.artifact_manifest.resolve()),
            'sha256': protocol.digest(args.artifact_manifest), 'revision': document.get('revision')}


def parent(args, cases):
    canonical = verify_artifacts(args)
    out = args.out.resolve()
    library, model = args.library.resolve(), args.model.resolve()
    manifest = {'model': str(model), 'model_sha256': protocol.digest(model),
                'library': str(library), 'library_sha256': protocol.digest(library),
                'harness_sha256': protocol.digest(Path(__file__)),
                'protocol_helper_sha256': protocol.digest(Path(protocol.__file__)),
                'cases_sha256': protocol.digest(args.cases), 'sampling': sampling(args),
                'final_validator_sha256': protocol.digest(Path(compare_servers.__file__)),
                'canonical_artifact_manifest': canonical,
                'other_penalties': {'frequency_penalty': 0.0, 'presence_penalty': 0.0},
                'min_p': {'value': 0.0, 'configurable': False, 'source': 'native SamplingParams default; public C/HTTP do not expose it'},
                'native_seed_api': False, 'draft_n_max': args.draft,
                'requested_context': args.context, 'requested_threads': args.threads,
                'native_concurrency_cells': args.concurrency,
                'http_engine_concurrency_ceiling': max(args.concurrency),
                'native': [], 'http': [], 'failures': [],
                'interpretation': 'Integration smoke only; no ID equality, same-seed or population distribution equivalence claim.'}
    manifest['dependencies'] = {str(path): protocol.digest(path) for path in sorted({
        entry.resolve() for entry in library.parent.glob('lib*.so*') if entry.is_file()})}
    cpuinfo = Path('/proc/cpuinfo')
    if cpuinfo.exists():
        manifest['cpu_profile'] = cpuinfo.read_text().split('\n\n')[0]
    cache = library.parent / 'CMakeCache.txt'
    if cache.exists(): manifest['cmake_cache_sha256'] = protocol.digest(cache)
    manifest['host'] = platform.uname()._asdict()
    manifest['cpu_affinity'] = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None
    if args.server:
        manifest['server_sha256'] = protocol.digest(args.server)
    environment = dict(os.environ, DENSECORE_MAX_SEQ_LEN=str(args.context),
                       DENSECORE_MAX_NUM_SEQS=str(max(args.concurrency)), DENSECORE_PREFIX_CACHE_REUSE='0',
                       LD_LIBRARY_PATH=str(library.parent) + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
    protocol.write(out / 'summary.json', manifest)
    for mode in ('none', 'mtp'):
        for concurrency in args.concurrency:
            result = out / f'native-{mode}-c{concurrency}.json'
            command = [sys.executable, str(Path(__file__).resolve()), '--native-child', '--mode', mode,
                       '--child-concurrency', str(concurrency), '--library', str(library), '--model', str(model),
                       '--cases', str(args.cases.resolve()), '--out', str(result), '--threads', str(args.threads),
                       '--draft', str(args.draft), '--repeats', str(args.repeats), '--max-tokens', str(args.max_tokens),
                       '--timeout', str(args.timeout)]
            for key, value in sampling(args).items():
                command += ['--' + key.replace('_', '-'), str(value)]
            child_environment = dict(environment, DENSECORE_MAX_NUM_SEQS=str(concurrency))
            if platform.system() == 'Linux':
                child_environment['LD_PRELOAD'] = ' '.join(dict.fromkeys([str(library), *environment.get('LD_PRELOAD', '').split()]))
            log = out / f'native-{mode}-c{concurrency}.log'
            with log.open('w') as output:
                process = subprocess.run(command, env=child_environment, stdout=output, stderr=subprocess.STDOUT,
                                         timeout=args.timeout * (len(cases) * args.repeats + 5))
            row = json.loads(result.read_text()) if result.exists() else {'mode': mode, 'concurrency': concurrency}
            row['mtp_counters'] = protocol.extract_mtp_counters(log.read_text())
            manifest['native'].append(row)
            if process.returncode or row.get('error'):
                manifest['failures'].append(f'native {mode}/c{concurrency}: {row.get("error", "child failed")}')
            else:
                plan = row['runtime_plan']
                if plan['spec_selected'] != (1 if mode == 'none' else 2) or \
                   plan['effective_context'] != args.context or plan['effective_concurrency'] != concurrency:
                    manifest['failures'].append(f'native {mode}/c{concurrency}: runtime selection/context/concurrency mismatch')
                if not row['cancellation']['passed']:
                    manifest['failures'].append(f'native {mode}/c{concurrency}: cancellation failed')
            if mode == 'mtp':
                row['branch_coverage'] = branch_coverage(row['mtp_counters'])
                if not all(row['branch_coverage'].values()):
                    manifest['failures'].append(f'mtp/c{concurrency}: branch not exercised, see coverage flags')
            protocol.write(out / 'summary.json', manifest)
        if args.server:
            http_cases = cases + [{'id': 'stop_sequence', 'prompt': 'Reply exactly Paris<END>ignore.',
                                  'expected': r'\bParis\b', 'stop': ['<END>']}]
            result = protocol.http_mode(args, mode, out, environment,
                        request_fn=functools.partial(http_request, probability=sampling(args)),
                        cases=http_cases, probe_sampling=sampling(args))
            manifest['http'].append(result)
            for group in result['groups']:
                for row in group['requests']:
                    if row.get('error'):
                        manifest['failures'].append(f'HTTP {mode}/c{group["concurrency"]}: {row["error"]}')
            probe = result['disconnect_probe']
            if not probe['saw_event'] or probe['ready_after_disconnect'] != 200 or probe['followup'].get('error'):
                manifest['failures'].append(f'HTTP {mode}: disconnect/recovery failed')
            protocol.write(out / 'summary.json', manifest)
    for path, expected in [(library, manifest['library_sha256']), (model, manifest['model_sha256']),
                           *[(Path(path), sha) for path, sha in manifest['dependencies'].items()]]:
        if protocol.digest(path) != expected:
            manifest['failures'].append(f'artifact changed: {path}')
    if protocol.digest(args.artifact_manifest) != canonical['sha256']:
        manifest['failures'].append('canonical artifact manifest changed')
    if args.server and protocol.digest(args.server) != manifest['server_sha256']:
        manifest['failures'].append('server changed')
    manifest.update(passed=not manifest['failures'], http_verified=bool(args.server),
                    population_distribution_equivalence='not tested; see separate formal native fixtures',
                    greedy128_control='not part of this complete-answer stochastic smoke')
    protocol.write(out / 'summary.json', manifest)
    return int(not manifest['passed'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['library', 'model', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--server', type=Path)
    parser.add_argument('--artifact-manifest', type=Path)
    parser.add_argument('--expected-model-sha256')
    parser.add_argument('--build-lock', type=Path)
    parser.add_argument('--cases', type=Path, default=Path(__file__).with_name('mtp_sampling_cases.json'))
    parser.add_argument('--concurrency', type=int, nargs='+', choices=[1, 4], default=[1, 4])
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--context', type=int, default=2048)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--draft', type=int, choices=[1, 2, 3], default=2)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--http-max-tokens', type=int, default=1024)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--port', type=int, default=18091)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top-k', type=int, default=40)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--repetition-penalty', type=float, default=1.05)
    parser.add_argument('--native-child', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--mode', choices=['none', 'mtp'], help=argparse.SUPPRESS)
    parser.add_argument('--child-concurrency', type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not (0 < args.temperature <= 2 and args.top_k > 1 and 0 < args.top_p < 1 and args.repetition_penalty >= 1):
        parser.error('require positive temperature, top-k>1,0<top-p<1, repetition penalty>=1')
    if min(args.max_tokens, args.http_max_tokens) < 256 or min(args.context, args.threads, args.repeats) < 1:
        parser.error('complete-answer budgets>=256; context/threads/repeats positive')
    if not args.native_child and not (args.server and args.artifact_manifest and args.expected_model_sha256):
        parser.error('parent requires server, canonical artifact manifest and pinned model SHA')
    cases = json.loads(args.cases.read_text())
    if not cases or len({case['id'] for case in cases}) != len(cases):
        parser.error('cases must have unique IDs and be nonempty')
    for case in cases:
        re.compile(case['expected'])
    try:
        if args.native_child:
            native_child(args, cases)
            return 0
        if args.build_lock:
            import fcntl
            with args.build_lock.open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                return parent(args, cases)
        return parent(args, cases)
    except Exception as error:
        path = args.out if args.native_child else args.out / 'summary.json'
        partial = json.loads(path.read_text()) if path.exists() else {}
        partial.update(passed=False, error=str(error))
        protocol.write(path, partial)
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
