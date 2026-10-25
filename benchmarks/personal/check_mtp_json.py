#!/usr/bin/env python3
"""Prepare/run gated same-file native and HTTP JSON correctness probes, never benchmarks."""
import argparse
import codecs
import ctypes as C
import functools
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import check_mtp_parity as protocol
import check_mtp_sampling as sampling_probe

EX = C.CFUNCTYPE(None, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_void_p)
CASES = [
    {'id': 'answer42', 'prompt': 'Return exactly one JSON object with the single key "answer" and the integer 42. No other text.', 'expected': r'\{'},
    {'id': 'empty_object', 'prompt': 'Return exactly the empty JSON object {}. No other text.', 'expected': r'\{'},
]
CANCEL_PROMPT = 'Return one JSON object with key "numbers" whose value lists every integer from 1 through 10000 in an array. Start immediately.'
UNSUPPORTED_FIELDS = {'action_token_start', 'action_token_count', 'min_p', 'seed', 'stop'}


def probability(args):
    return {key: getattr(args, key) for key in ('temperature', 'top_k', 'top_p', 'repetition_penalty')}


def validate_public_fields(fields):
    forbidden = UNSUPPORTED_FIELDS.intersection(fields)
    if forbidden:
        raise ValueError('unsupported JSON probe public fields: ' + ','.join(sorted(forbidden)))


def parse_object(text, case):
    text.encode('utf-8', 'strict')
    if '\ufffd' in text or '<think' in text or '</think' in text:
        raise ValueError('replacement UTF-8 or reasoning markup in JSON output')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key: ' + key)
            result[key] = value
        return result
    def constant(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    if type(value) is not dict:
        raise ValueError('JSON output must be an object')
    if case['id'] == 'empty_object':
        valid = not value
    else:
        valid = set(value) == {'answer'} and type(value['answer']) is int and value['answer'] == 42
    if not valid:
        raise ValueError('JSON semantic object mismatch')
    return value


def new_record(case, repeat):
    return {'case': case['id'], 'repeat': repeat, 'pieces': [], 'delivered_token_ids': [], 'utf8_errors': [],
            'terminal_events': 0, 'completion_events': 0, 'protocol_errors': [], 'done': threading.Event(),
            'first_token': threading.Event(), '_decoder': codecs.getincrementaldecoder('utf-8')('strict')}


def record_piece(row, data, token_id, finished):
    """data is an owned copy of the callback's borrowed pointer, never retained as a pointer."""
    if row['terminal_events']:
        row['protocol_errors'].append('callback after terminal')
        if not finished:
            return
    if finished:
        if row['completion_events'] != 1:
            row['protocol_errors'].append('terminal before exactly one completion metadata event')
        row['terminal_events'] += 1
        try:
            row['terminal_text'] = data.decode('utf-8', 'strict')
        except UnicodeError as error:
            row['terminal_text'] = ''
            row['utf8_errors'].append(str(error))
        try:
            row['pieces'].append(row['_decoder'].decode(b'', final=True))
        except UnicodeError as error:
            row['utf8_errors'].append(str(error))
        row['done'].set()
        return
    if token_id >= 0:
        row['delivered_token_ids'].append(token_id)
    if token_id >= 0 or data:
        row['first_token'].set()
    try:
        row['pieces'].append(row['_decoder'].decode(data, final=False))
    except UnicodeError as error:
        row['utf8_errors'].append(str(error))


def record_completion(row, count, reason):
    if row['terminal_events']:
        row['protocol_errors'].append('completion metadata after terminal')
    row.update(generated_tokens=count, finish_reason=reason)
    row['completion_events'] += 1


def serializable(row):
    return {key: value for key, value in row.items() if key not in ('done', 'first_token', '_decoder', '_owners')}


def validate_native(row, case, cancelled=False):
    if row.get('terminal_events') != 1 or row.get('completion_events') != 1:
        raise ValueError('native terminal/metadata must each arrive exactly once')
    if row.get('finish_reason') != (3 if cancelled else 1):
        raise ValueError('native JSON requires stop completion (or error for explicit cancellation), never length')
    if row.get('protocol_errors'):
        raise ValueError('native callback order: ' + '; '.join(row['protocol_errors']))
    count = row.get('generated_tokens')
    if type(count) is not int or count < 1 or count < len(row['delivered_token_ids']) or row['utf8_errors']:
        raise ValueError('invalid native generated accounting or UTF-8')
    if cancelled and not re.search(r'\bcancel(?:ed|led)\b', row.get('terminal_text', ''), re.IGNORECASE):
        raise ValueError('native cancellation must explicitly identify request cancellation')
    if not cancelled:
        if row.get('terminal_text'):
            raise ValueError('unexpected native terminal/error text')
        row['parsed_object'] = parse_object(''.join(row['pieces']), case)
    row['id_scope'] = 'Delivered callback event IDs only; EOS/empty samples may be omitted. ID -1 text flush is not a sample.'


def brace_ids(lib, handle):
    ids = []
    for text in (b'{', b'}'):
        count = lib.CountTokens(handle, text, 0, 0)
        if count != 1:
            raise ValueError('unsupported brace tokenizer shape: require exactly one ID each for { and }')
        output = (C.c_int * 1)()
        written = lib.DenseCoreTokenizeText(handle, text, 0, 0, output, 1)
        if written != 1 or output[0] < 0:
            raise ValueError('unsupported brace tokenizer shape: native tokenize did not return one global ID')
        ids.append(int(output[0]))
    if ids[0] == ids[1]:
        raise ValueError('unsupported brace tokenizer shape: distinct structural IDs required')
    return sorted(ids)


def native_child(args):
    lib = C.CDLL(str(args.library.resolve()))
    lib.DenseCoreGetLastError.restype = C.c_char_p
    lib.InitEngineWithOptions.argtypes = [C.c_char_p, C.c_char_p, C.c_int, C.c_int, C.c_int, C.c_int, C.POINTER(protocol.InitOptions)]
    lib.InitEngineWithOptions.restype = C.c_void_p
    lib.FreeEngine.argtypes, lib.FreeEngine.restype = [C.c_void_p], None
    lib.DenseCoreGetRuntimePlan.argtypes = [C.c_void_p, C.POINTER(protocol.RuntimePlan)]
    lib.DenseCoreGetRuntimePlan.restype = C.c_int
    lib.DenseCoreSetGenerationCompletionCallback.argtypes = [C.c_void_p, protocol.COMPLETE]
    lib.DenseCoreSetGenerationCompletionCallback.restype = C.c_int
    lib.CountTokens.argtypes, lib.CountTokens.restype = [C.c_void_p, C.c_char_p, C.c_int, C.c_int], C.c_int
    lib.DenseCoreTokenizeText.argtypes = [C.c_void_p, C.c_char_p, C.c_int, C.c_int, C.POINTER(C.c_int), C.c_int]
    lib.DenseCoreTokenizeText.restype = C.c_int
    lib.CancelRequest.argtypes, lib.CancelRequest.restype = [C.c_void_p, C.c_int], C.c_int
    submit_fn = lib.SubmitRequestWithSamplingConstraintsCallbackEx
    submit_fn.argtypes = [C.c_void_p, C.c_char_p, C.c_int, C.c_char_p, C.c_float, C.c_float, C.c_int,
                         C.c_float, C.POINTER(C.c_char_p), C.c_int, C.POINTER(C.c_int), C.c_int, C.c_int,
                         C.POINTER(C.c_int), C.c_int, EX, C.c_void_p]
    submit_fn.restype = C.c_int
    options = protocol.InitOptions(C.sizeof(protocol.InitOptions), 1, 1 if args.mode == 'none' else 2, args.draft)
    handle = lib.InitEngineWithOptions(os.fsencode(args.model.resolve()), None, args.threads, -1, 0, 0, C.byref(options))
    if not handle:
        raise RuntimeError((lib.DenseCoreGetLastError() or b'init failed').decode('utf-8', 'strict'))
    rows, lock = {}, threading.Lock()
    result = {'mode': args.mode, 'concurrency': args.child_concurrency, 'sampling': probability(args), 'requests': []}
    @EX
    def token(data, size, token_id, finished, userdata):
        with lock:
            row = rows.get(int(userdata or 0))
            if row is not None:
                try:
                    record_piece(row, C.string_at(data, size) if data and size else b'', token_id, bool(finished))
                except Exception as error:  # noqa: BLE001 -- preserve raw probe failures, including callback boundaries.
                    row['utf8_errors'].append(str(error))
                    if finished:
                        row['done'].set()
    @protocol.COMPLETE
    def completed(count, reason, userdata):
        with lock:
            row = rows.get(int(userdata or 0))
            if row is not None:
                record_completion(row, count, reason)
    def submit(case, repeat, cancel=False):
        with lock:
            key = len(rows) + 1
            row = new_record(case, repeat)
            rows[key] = row
        mask = result['brace_token_ids'] if case['id'] == 'empty_object' else []
        array = (C.c_int * len(mask))(*mask) if mask else None
        # Retain every ctypes owner until FreeEngine has joined all producers.
        row['_owners'] = (array, token, completed)
        request_id = submit_fn(handle, case['prompt'].encode('utf-8'), args.max_tokens, None,
                               args.temperature, args.top_p, args.top_k, args.repetition_penalty, None, 1,
                               array, len(mask), int(bool(mask)), None, 0, token, C.c_void_p(key))
        if request_id < 0:
            raise RuntimeError((lib.DenseCoreGetLastError() or b'submit failed').decode('utf-8', 'strict'))
        row['request_id'] = request_id
        if cancel:
            protocol.wait_for_first_token(row, args.timeout)
            if row['done'].is_set():
                raise ValueError('cancel probe already finished before cancellation')
            row['cancel_status'] = lib.CancelRequest(handle, request_id)
        if not row['done'].wait(args.timeout):
            lib.CancelRequest(handle, request_id)
            raise TimeoutError('native JSON terminal deadline')
        snapshot = serializable(row)
        snapshot.pop('_owners', None)
        validate_native(snapshot, case, cancelled=cancel)
        return snapshot
    try:
        plan = protocol.RuntimePlan()
        plan.struct_size, plan.version = C.sizeof(plan), 1
        if lib.DenseCoreGetRuntimePlan(handle, C.byref(plan)) != 0:
            raise ValueError('native runtime plan unavailable')
        result['runtime_plan'] = {name: getattr(plan, name) for name, _ in plan._fields_}
        result['runtime_plan']['spec_reason'] = plan.spec_reason.decode('utf-8', 'strict')
        expected = (args.context, args.child_concurrency, 1 if args.mode == 'none' else 2)
        if (plan.effective_context, plan.effective_concurrency, plan.spec_selected) != expected or plan.effective_threads != args.threads:
            raise ValueError('effective context/concurrency/mode differs; no silent capacity lowering allowed')
        result['brace_token_ids'] = brace_ids(lib, handle)
        if lib.DenseCoreSetGenerationCompletionCallback(handle, completed) != 0:
            raise ValueError('authoritative completion callback unavailable')
        jobs = [(case, repeat) for repeat in range(args.repeats) for case in CASES]
        with ThreadPoolExecutor(max_workers=args.child_concurrency) as pool:
            result['requests'] = list(pool.map(lambda job: submit(*job), jobs))
        cancelled = submit({'id': 'cancel', 'prompt': CANCEL_PROMPT}, 0, True)
        if cancelled['cancel_status'] != 0 or cancelled['generated_tokens'] >= args.max_tokens:
            raise ValueError('native JSON cancellation did not interrupt active generation')
        result['cancellation'] = {'passed': True, 'request': cancelled, 'recovery': submit(CASES[0], args.repeats)}
    except Exception as error:
        result['error'] = str(error)
        raise
    finally:
        lib.FreeEngine(handle)
        final_error = None
        result['partial_requests'] = []
        for row in rows.values():
            clean = serializable(row)
            clean.pop('_owners', None)
            result['partial_requests'].append(clean)
            if 'request_id' in clean:
                try:
                    case = next((case for case in CASES if case['id'] == clean['case']), {'id': 'cancel'})
                    validate_native(clean, case, cancelled=clean['case'] == 'cancel')
                except ValueError as error:
                    final_error = str(error)
        if final_error:
            result.setdefault('error', final_error)
        protocol.write(args.out, result)
        if final_error:
            raise ValueError(final_error)


def http_request(base, case, stream, max_tokens, timeout, probability, capture_dir=None):
    validate_public_fields(probability)
    wire = dict(probability, response_format={'type': 'json_object'})
    if case['id'] == 'empty_object':
        wire.update(allowed_token_ids=case['brace_token_ids'], allowed_tokens_strict=True)
    row = protocol.http_request(base, case, stream, max_tokens, timeout, wire)
    if capture_dir is not None:
        try:
            with protocol.urllib.request.urlopen(base + '/v1/runtime/profile', timeout=timeout) as response:
                row['runtime_profile'] = json.load(response)
        except Exception as error:  # noqa: BLE001 -- preserve raw probe failures, including callback boundaries.
            row.setdefault('error', 'runtime profile unavailable: ' + str(error))
    if not row.get('error'):
        try:
            if row.get('finish_reason') != 'stop' or row.get('done') is not True:
                raise ValueError('HTTP JSON requires stop and stream [DONE], never length')
            count = row.get('usage', {}).get('completion_tokens')
            if type(count) is not int or count < 1:
                raise ValueError('HTTP JSON requires positive authoritative usage')
            if row.get('reasoning'):
                raise ValueError('JSON-only probe must not expose reasoning')
            if stream:
                finishes = [choice for event in row['events'] for choice in event['data'].get('choices', [])
                            if choice.get('finish_reason') is not None]
                if len(finishes) != 1:
                    raise ValueError('HTTP JSON requires one finish event')
            row['parsed_object'] = parse_object(row['content'], case)
        except Exception as error:  # noqa: BLE001 -- preserve raw probe failures, including callback boundaries.
            row['error'] = str(error)
    if capture_dir is not None:
        protocol.write(Path(capture_dir) / f'{case["id"]}-stream{int(stream)}-{time.time_ns()}.json', row)
    return row


def verify_qualified(args):
    canonical = sampling_probe.verify_artifacts(args)
    document = json.loads(args.artifact_manifest.read_text())
    if not re.fullmatch(r'[0-9a-f]{40}', args.qualified_source or '') or document.get('revision') != args.qualified_source:
        raise ValueError('qualified full source revision must match canonical manifest')
    if document.get('json_completion_unit_qualified') is not True or args.qualified_source.startswith('4545e05'):
        raise ValueError('future unit-qualified combined JSON-completion manifest required; old4545 runtime forbidden')
    if document.get('json_completion_unit_source') != args.qualified_source:
        raise ValueError('JSON unit qualification source must match artifact source')
    unit_report = document.get('json_completion_unit_report', {})
    report_path = Path(unit_report.get('path', ''))
    if not report_path.is_absolute() or not report_path.is_file() or protocol.digest(report_path) != unit_report.get('sha256'):
        raise ValueError('JSON unit qualification report hash/path mismatch')
    if json.loads(report_path.read_text()).get('source') != args.qualified_source:
        raise ValueError('JSON unit qualification report source mismatch')
    canonical['json_completion_unit_report'] = dict(unit_report)
    for path, expected in document['files'].items():
        if protocol.digest(Path(path)) != expected:
            raise ValueError('canonical dependency/artifact changed: ' + path)
    local_dependencies = {str(p.resolve()): protocol.digest(p) for p in args.library.parent.glob('lib*.so*') if p.is_file()}
    if not local_dependencies or any(document['files'].get(p) != value for p, value in local_dependencies.items()):
        raise ValueError('all local runtime dependencies must be pinned in canonical manifest')
    return canonical, local_dependencies


def observe_execution_log(path, mode, source):
    """Observe actual work; profile flags and API finish=stop are insufficient."""
    if not re.fullmatch(r'[0-9a-f]{40}', source):
        raise ValueError('execution log requires the full qualified artifact source SHA')
    path = Path(path).resolve()
    evidence = {'path': str(path), 'source_revision': source,
                'source_binding': 'Verified canonical artifact revision, not inferred from log text.',
                'attribution_scope': 'Aggregate process log only; native/log-local IDs are not mapped to concurrent HTTP rows.',
                'acceptance_scope': 'Algorithm acceptance including terminal EOS/JSON closing tokens, not retained speculative inputs.',
                'grammar_complete_events': []}
    try:
        text = path.read_text(errors='replace')
        evidence['sha256'] = protocol.digest(path)
    except OSError as error:
        evidence.update(error=str(error), passed=False, missing_coverage=['execution_log_available'])
        return evidence
    for number, line in enumerate(text.splitlines(), 1):
        summary = re.search(r'\[([A-Za-z0-9_]*DecodeSummary)\]', line)
        if not summary:
            continue
        fields = dict(re.findall(r'(\w+)=([^\s]+)', line))
        if fields.get('finish_cause') == 'grammar_complete':
            evidence['grammar_complete_events'].append({'line': number, 'summary': summary.group(1),
                                                        'request_id': fields.get('req')})
    counters = protocol.extract_mtp_counters(text)
    evidence['mtp_counters'] = counters
    required = {'grammar_completion_policy_observed': bool(evidence['grammar_complete_events'])}
    if mode == 'mtp':
        required.update(batched_target_observed=counters['maximum_target_width'] > 1,
                        target_verification_observed=counters['verified'] > 0,
                        algorithm_acceptance_observed=counters['accepted'] > 0)
    evidence.update(coverage=required, missing_coverage=[name for name, passed in required.items() if not passed])
    evidence['passed'] = not evidence['missing_coverage']
    return evidence


def attach_execution_log(manifest, path, mode, label, source):
    evidence = observe_execution_log(path, mode, source)
    evidence['profile'] = label
    manifest['execution_logs'].append(evidence)
    if not evidence['passed']:
        manifest['failures'].append(label + ': missing actual execution coverage: ' + ', '.join(evidence['missing_coverage']))
    return evidence


def verify_execution_log_hashes(evidence):
    failures = []
    for row in evidence:
        try:
            if row.get('sha256') != protocol.digest(Path(row['path'])):
                failures.append(row['profile'] + ': execution log changed after observation')
        except OSError as error:
            failures.append(row['profile'] + ': execution log unavailable: ' + str(error))
    return failures


def parent(args):
    canonical, dependencies = verify_qualified(args)
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('output directory must be empty; preserve each profile failure separately')
    manifest = {'qualified_artifact_manifest': canonical, 'model_sha256': protocol.digest(args.model),
                'library_sha256': protocol.digest(args.library), 'server_sha256': protocol.digest(args.server),
                'dependencies': dependencies, 'harness_sha256': protocol.digest(Path(__file__)),
                'source_revision': args.qualified_source, 'harness_worktree_revision': subprocess.check_output(
                    ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parent, text=True).strip(),
                'protocol_helper_sha256': protocol.digest(Path(protocol.__file__)),
                'sampling_helper_sha256': protocol.digest(Path(sampling_probe.__file__)),
                'cases': CASES, 'cases_sha256': hashlib.sha256(json.dumps(CASES, sort_keys=True).encode()).hexdigest(),
                'http_validator_helper_sha256': protocol.digest(Path(sampling_probe.compare_servers.__file__)),
                'cancel_prompt': CANCEL_PROMPT, 'sampling': probability(args),
                'requested_context': args.context, 'requested_threads': args.threads, 'concurrency': args.concurrency,
                'http_engine_concurrency_ceiling': max(args.concurrency),
                'host': platform.uname()._asdict(), 'cpu_affinity': sorted(os.sched_getaffinity(0)),
                'native': [], 'http': [], 'execution_logs': [], 'failures': [],
                'scope': 'JSON semantic/protocol/isolation smoke; no same-seed, full sample-ID parity, distribution, speed or automatic promotion claim. Global action range is fixture-only, not public API.'}
    cache = args.library.parent / 'CMakeCache.txt'
    if cache.exists(): manifest['cmake_cache_sha256'] = protocol.digest(cache)
    env = dict(os.environ, DENSECORE_MAX_SEQ_LEN=str(args.context), DENSECORE_MAX_NUM_SEQS=str(max(args.concurrency)),
               DENSECORE_PREFIX_CACHE_REUSE='0', LD_LIBRARY_PATH=str(args.library.parent.resolve()) + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
    protocol.write(out / 'summary.json', manifest)
    try:
        for mode in ('none', 'mtp'):
            brace_mask = None
            for concurrency in args.concurrency:
                result_path = out / f'native-{mode}-c{concurrency}.json'
                command = [sys.executable, str(Path(__file__).resolve()), '--native-child', '--mode', mode,
                           '--child-concurrency', str(concurrency)]
                for key in ('library', 'server', 'model', 'artifact_manifest', 'expected_model_sha256', 'qualified_source',
                            'threads', 'context', 'repeats', 'draft', 'max_tokens', 'timeout'):
                    command += ['--' + key.replace('_', '-'), str(getattr(args, key))]
                command += ['--out', str(result_path)]
                for key, value in probability(args).items(): command += ['--' + key.replace('_', '-'), str(value)]
                child_env = dict(env, DENSECORE_MAX_NUM_SEQS=str(concurrency))
                if platform.system() == 'Linux':
                    child_env['LD_PRELOAD'] = ' '.join(dict.fromkeys([str(args.library.resolve()), *env.get('LD_PRELOAD', '').split()]))
                native_log = out / f'native-{mode}-c{concurrency}.log'
                try:
                    with native_log.open('w') as log:
                        run = subprocess.run(command, check=False, env=child_env, stdout=log, stderr=subprocess.STDOUT,
                                             timeout=args.timeout * (len(CASES) * args.repeats + 4))
                finally:
                    execution = attach_execution_log(manifest, native_log, mode, f'native{mode}/c{concurrency}', args.qualified_source)
                row = json.loads(result_path.read_text()) if result_path.exists() else {'error': 'child produced no report'}
                row.update(mode=mode, concurrency=concurrency)
                row['execution_evidence'] = execution
                row['mtp_counters'] = execution.get('mtp_counters', {})
                manifest['native'].append(row)
                if run.returncode or row.get('error'):
                    manifest['failures'].append(f'native{mode}/c{concurrency}: {row.get("error", "child failed")}')
                else:
                    mask = row['brace_token_ids']
                    if brace_mask is not None and mask != brace_mask: raise ValueError('brace token IDs changed between profiles')
                    brace_mask = mask
                protocol.write(out / 'summary.json', manifest)
            if brace_mask is None:
                manifest['failures'].append(f'HTTP{mode}: unavailable supported brace tokenizer shape; not executed')
                continue
            cases = [dict(case, brace_token_ids=brace_mask) for case in CASES]
            probe = dict(probability(args), response_format={'type': 'json_object'},
                         messages=[{'role': 'user', 'content': CANCEL_PROMPT}])
            try:
                http = protocol.http_mode(args, mode, out, env, request_fn=functools.partial(http_request, probability=probability(args), capture_dir=out / f'http-{mode}-requests'),
                                          cases=cases, probe_sampling=probe)
            finally:
                execution = attach_execution_log(manifest, out / f'http-{mode}.log', mode, f'HTTP{mode}', args.qualified_source)
            http['execution_evidence'] = execution
            manifest['http'].append(http)
            for group in http['groups']:
                for row in group['requests']:
                    if row.get('error'): manifest['failures'].append(f'HTTP{mode}/c{group["concurrency"]}: {row["error"]}')
            disconnect = http['disconnect_probe']
            if not disconnect['saw_event'] or disconnect['ready_after_disconnect'] != 200 or disconnect['followup'].get('error'):
                manifest['failures'].append(f'HTTP{mode}: JSON disconnect/recovery failed')
            protocol.write(out / 'summary.json', manifest)
    except Exception as error:  # noqa: BLE001 -- preserve raw probe failures, including callback boundaries.
        manifest['failures'].append(str(error))
    finally:
        try:
            after, after_deps = verify_qualified(args)
            if after != canonical or after_deps != dependencies: raise ValueError('canonical manifest/dependencies changed')
        except Exception as error:  # noqa: BLE001 -- preserve raw probe failures, including callback boundaries.
            manifest['failures'].append(str(error))
        manifest['failures'].extend(verify_execution_log_hashes(manifest['execution_logs']))
        manifest['passed'] = not manifest['failures']
        protocol.write(out / 'summary.json', manifest)
    return int(not manifest['passed'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('library', 'server', 'model', 'out', 'artifact-manifest'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--expected-model-sha256', required=True)
    parser.add_argument('--qualified-source', required=True)
    parser.add_argument('--concurrency', type=int, choices=[1, 4], nargs='+', default=[1])
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--context', type=int, default=2048)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--draft', type=int, choices=[1, 2, 3], default=2)
    parser.add_argument('--max-tokens', type=int, choices=[512, 1024], default=512)
    parser.add_argument('--http-max-tokens', type=int, choices=[512, 1024], default=512)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--port', type=int, default=18093)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--top-k', type=int, default=1)
    parser.add_argument('--top-p', type=float, default=1)
    parser.add_argument('--repetition-penalty', type=float, default=1)
    parser.add_argument('--build-lock', type=Path, default=Path('/home/jaewook/dc/build.lock'))
    parser.add_argument('--native-child', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--mode', choices=['none', 'mtp'], help=argparse.SUPPRESS)
    parser.add_argument('--child-concurrency', type=int, choices=[1, 4], default=1, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.context < args.max_tokens or min(args.repeats, args.threads) < 1 or args.timeout <= 0 or \
       not (0 <= args.temperature <= 2 and args.top_k >= 0 and 0 < args.top_p <= 1 and args.repetition_penalty >= 1):
        parser.error('invalid context/budget/sampling/thread/repeat settings')
    try:
        if args.native_child:
            if args.out.exists():
                raise ValueError('native output already exists; profile evidence cannot be overwritten')
            verify_qualified(args)  # Gate before CDLL/engine allocation, including direct child invocations.
            native_child(args)
            verify_qualified(args)
            return 0
        import fcntl
        with args.build_lock.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return parent(args)
    except Exception as error:  # noqa: BLE001 -- preserve raw probe failures, including callback boundaries.
        # Never overwrite a previously completed profile or erase partial child evidence.
        target = args.out if args.native_child else args.out / 'harness-error.json'
        if target.exists():
            target = target.with_name(target.stem + f'-{time.time_ns()}.error.json')
        partial = {}
        partial.update(passed=False, error=str(error))
        protocol.write(target, partial)
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
