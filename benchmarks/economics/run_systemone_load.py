#!/usr/bin/env python3
"""Measure fixed pretokenized System One requests without result caching/retries."""
from __future__ import annotations

import argparse
import concurrent.futures
import ctypes as C
import datetime
import socket
import json
import math
import os
import platform
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from systemone_economics import digest, load_workload, require, validate_artifacts, validate_http_response, loaded_runtime_dependencies_sha256, validate_server_identity


class DecisionInput(C.Structure):
    _fields_ = [('tokens', C.POINTER(C.c_int32)), ('token_count', C.c_size_t),
                ('question_type', C.c_int), ('markers', C.POINTER(C.c_int32)), ('marker_count', C.c_size_t)]


class DecisionOutput(C.Structure):
    _fields_ = [('logits', C.POINTER(C.c_float)), ('act_logits', C.POINTER(C.c_float))]


def require_unpadded(seq):
    mask = seq.get('attention_mask')
    if mask is not None:
        require(isinstance(mask, list) and len(mask) == len(seq['token_ids']) and
                all(type(v) is int and v == 1 for v in mask),
                'native ABI requires unpadded all-one attention masks')


class Native:
    def __init__(self, library, model, threads, precision='fp32'):
        self.lib = C.CDLL(str(Path(library).resolve()))
        self.lib.DenseCoreDecisionLoad.argtypes = [C.c_char_p, C.c_int, C.c_char_p, C.c_size_t]
        self.lib.DenseCoreDecisionLoad.restype = C.c_void_p
        self.lib.DenseCoreDecisionFree.argtypes = [C.c_void_p]
        self.lib.DenseCoreDecisionFree.restype = None
        self.lib.DenseCoreDecisionPredict.argtypes = [C.c_void_p, C.POINTER(C.c_int32), C.c_size_t, C.c_int,
            C.POINTER(C.c_int32), C.c_size_t, C.POINTER(C.c_float), C.POINTER(C.c_float), C.c_char_p, C.c_size_t]
        self.lib.DenseCoreDecisionPredict.restype = C.c_int
        err = C.create_string_buffer(4096)
        require(precision in ('fp32', 'q8_0'), 'unsupported native precision')
        if precision == 'fp32':
            self.handle = self.lib.DenseCoreDecisionLoad(str(model).encode(), threads, err, len(err))
        else:
            require(hasattr(self.lib, 'DenseCoreDecisionLoadEx'), 'precision selector ABI unavailable')
            self.lib.DenseCoreDecisionLoadEx.argtypes = [C.c_char_p, C.c_int, C.c_int, C.c_char_p, C.c_size_t]
            self.lib.DenseCoreDecisionLoadEx.restype = C.c_void_p
            self.handle = self.lib.DenseCoreDecisionLoadEx(str(model).encode(), threads, 1, err, len(err))
        require(bool(self.handle), err.value.decode(errors='replace'))
        self.lock = threading.Lock()

    def __call__(self, row):
        require(row.get('padding_tokens', 0) == 0, 'native ABI cannot execute padded requests')
        out = []
        with self.lock:
            for seq in row['sequences']:
                require_unpadded(seq)
                ids = (C.c_int32 * len(seq['token_ids']))(*seq['token_ids'])
                markers = (C.c_int32 * len(seq['markers']))(*seq['markers'])
                logits, act, err = (C.c_float * len(markers))(), (C.c_float * 2)(), C.create_string_buffer(4096)
                ok = self.lib.DenseCoreDecisionPredict(self.handle, ids, len(ids), seq['question_type'], markers,
                    len(markers), logits, act, err, len(err))
                require(ok == 1, err.value.decode(errors='replace'))
                require(all(math.isfinite(v) for v in [*logits, *act]), 'nonfinite native output')
                out.append({'logits': list(logits), 'act_logits': list(act)})
        return out

    def call_batch(self, rows):
        require(all(row.get('padding_tokens', 0) == 0 for row in rows), 'native ABI cannot execute padded requests')
        sequences = [seq for row in rows for seq in row['sequences']]
        require(1 <= len(sequences) <= 64, 'native batch requires 1..64 encoder sequences')
        for seq in sequences:
            require_unpadded(seq)
        require(hasattr(self.lib, 'DenseCoreDecisionPredictBatch'), 'native batch ABI unavailable')
        predict = self.lib.DenseCoreDecisionPredictBatch
        predict.argtypes = [C.c_void_p, C.POINTER(DecisionInput), C.c_size_t,
                            C.POINTER(DecisionOutput), C.c_char_p, C.c_size_t]
        predict.restype = C.c_int
        # Keep owning arrays alive for the entire native call.
        tokens = [(C.c_int32 * len(seq['token_ids']))(*seq['token_ids']) for seq in sequences]
        markers = [(C.c_int32 * len(seq['markers']))(*seq['markers']) for seq in sequences]
        logits = [(C.c_float * len(seq['markers']))() for seq in sequences]
        acts = [(C.c_float * 2)() for _ in sequences]
        inputs = (DecisionInput * len(sequences))(*[DecisionInput(t, len(t), seq['question_type'], m, len(m))
                    for t, m, seq in zip(tokens, markers, sequences)])
        outputs = (DecisionOutput * len(sequences))(*[DecisionOutput(v, a) for v, a in zip(logits, acts)])
        err = C.create_string_buffer(4096)
        with self.lock:
            ok = predict(self.handle, inputs, len(sequences), outputs, err, len(err))
        require(ok == 1, err.value.decode(errors='replace'))
        flat = []
        for v, a in zip(logits, acts):
            require(all(math.isfinite(x) for x in [*v, *a]), 'nonfinite native batch output')
            flat.append({'logits': list(v), 'act_logits': list(a)})
        out, offset = [], 0
        for row in rows:
            n = len(row['sequences'])
            out.append(flat[offset:offset+n])
            offset += n
        return out

    def runtime_dependencies_sha256(self):
        return loaded_runtime_dependencies_sha256()

    def close(self):
        self.lib.DenseCoreDecisionFree(self.handle)


def http_call(endpoint, timeout, expected_model, row):
    req = urllib.request.Request(endpoint, json.dumps(row['request']).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        out = json.load(response)
    return validate_http_response(out, expected_model, row)


def observe_host():
    model = platform.processor() or platform.machine()
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.lower().startswith(('model name', 'hardware')) and ':' in line:
                model = line.split(':', 1)[1].strip()
                break
    except OSError:
        pass
    try:
        memory_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    except (OSError, ValueError, AttributeError):
        memory_bytes = None
    return {'cpu_model': model, 'architecture': platform.machine(), 'os': platform.platform(),
            'logical_cpu_count': os.cpu_count(),
            'affinity_logical_cpu_count': len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
            'host_memory_bytes': memory_bytes,
            'identity_limit': 'Host observations do not independently verify cloud provider, region, or SKU'}


class NativeRSSSampler:
    """Sample whole-process RSS; this is never server RSS for HTTP runs."""
    def __init__(self, interval_s=.05):
        self.interval_s = interval_s
        self.samples = []
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _sample(self):
        try:
            for line in Path('/proc/self/status').read_text().splitlines():
                if line.startswith('VmRSS:'):
                    self.samples.append(int(line.split()[1])*1024)
                    break
        except (OSError, ValueError, IndexError):
            pass

    def _loop(self):
        self._sample()
        while not self.stopped.wait(self.interval_s):
            self._sample()
        self._sample()

    def start(self):
        self.thread.start()

    def finish(self):
        self.stopped.set()
        self.thread.join()
        if not self.samples:
            return {'status': 'unavailable', 'reason': '/proc/self/status VmRSS unavailable'}
        return {'status': 'measured', 'scope': 'whole_native_client_process_including_python_and_model',
                'source': '/proc/self/status VmRSS sampling', 'sample_interval_s': self.interval_s,
                'samples': len(self.samples), 'peak_rss_bytes': max(self.samples),
                'limit': 'Sampled maximum may miss short peaks; includes Python and native model allocations'}


def observe_boot_id():
    value = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    require(str(uuid.UUID(value)) == value, 'invalid observed boot identity')
    return value


def observe_monotonic_clock():
    info = time.get_clock_info('monotonic')
    require(info.monotonic and not info.adjustable, 'nonadjustable monotonic clock required')
    return {'name': 'time.monotonic_ns', 'implementation': info.implementation,
            'monotonic': info.monotonic, 'adjustable': info.adjustable}


def run(call, rows, duration, concurrency, output, manifest_hash, boundary, threads=1, warmup=0, batch_size=1, resource=None, runtime_dependencies_sha256=None, server_identity=None):
    records = []
    run_id = str(uuid.uuid4())
    lock = threading.Lock()
    boot_id = observe_boot_id()
    monotonic_clock = observe_monotonic_clock()
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    origin_ns = time.monotonic_ns()
    deadline_ns = origin_ns + round(duration*1_000_000_000)
    next_id = 0
    rss = NativeRSSSampler() if boundary == 'full_model' else None
    if rss:
        rss.start()
    def worker():
        nonlocal next_id
        while True:
            with lock:
                now_ns = time.monotonic_ns()
                if now_ns >= deadline_ns:
                    return
                i = next_id
                started = (now_ns-origin_ns)/1_000_000_000
                next_id += batch_size
            batch_rows = [rows[(i+j) % len(rows)] for j in range(batch_size)]
            batch_records = [{'run_id': run_id, 'attempt_id': i+j, 'batch_id': i, 'batch_size': batch_size,
                'workload_id': row['id'], 'start_s': started, 'useful_tokens': row['_useful_tokens'],
                'compute_tokens': row['_compute_tokens'], 'success': False, 'error': None,
                'unresolved_backend_work': False} for j, row in enumerate(batch_rows)]
            try:
                outputs = [call(batch_rows[0])] if batch_size == 1 else call.call_batch(batch_rows)
                require(len(outputs) == len(batch_records), 'batch output count mismatch')
                for rec, response in zip(batch_records, outputs):
                    rec['response'] = response
                    rec['success'] = True
            except urllib.error.HTTPError as exc:
                for rec in batch_records:
                    rec['error'] = f'HTTP {exc.code}'
                    rec['unresolved_backend_work'] = exc.code in (408, 502, 504)
            except Exception as exc:
                for rec in batch_records:
                    rec['error'] = f'{type(exc).__name__}: {exc}'
                    rec['unresolved_backend_work'] = boundary == 'http_e2e'
            ended = (time.monotonic_ns()-origin_ns)/1_000_000_000
            for rec in batch_records:
                rec['end_s'] = ended
            with lock:
                records.extend(batch_records)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(worker) for _ in range(concurrency)]
            for f in futures:
                f.result()
    finally:
        finished_ns = time.monotonic_ns()
        finished_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        elapsed = (finished_ns-origin_ns)/1_000_000_000
        memory = rss.finish() if rss else {'status': 'unavailable', 'reason': 'HTTP client RSS does not establish server memory'}
    require(observe_boot_id() == boot_id, 'observed boot identity changed during run')
    if runtime_dependencies_sha256 is not None and boundary == 'full_model':
        require(loaded_runtime_dependencies_sha256() == runtime_dependencies_sha256, 'runtime dependencies changed during run')
    records.sort(key=lambda r: r['attempt_id'])
    Path(str(output)+'.records.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in records))
    metadata = {'schema': 'systemone-run-v2', 'boot_id': boot_id,
                'monotonic_clock': monotonic_clock, 'monotonic_started_ns': origin_ns, 'monotonic_finished_ns': finished_ns, 'run_id': run_id,
                'records_sha256': digest(Path(str(output)+'.records.jsonl')), 'manifest_sha256': manifest_hash, 'boundary': boundary,
                'submission_stop_s': duration, 'elapsed_s': elapsed, 'record_count': len(records),
                'concurrency': concurrency, 'threads': threads, 'batch_size': batch_size,
                'resource': dict(resource) if resource is not None else None,
                'runtime_dependencies_sha256': runtime_dependencies_sha256,
                'runtime_dependencies_stable': runtime_dependencies_sha256 is not None and boundary == 'full_model',
                'runtime_dependencies_origin': 'observed_process_maps' if boundary == 'full_model' else 'operator_attested_server_deployment' if server_identity else 'unavailable',
                'server_identity_sha256': server_identity['sha256'] if server_identity else None,
                'endpoint': server_identity['endpoint'] if server_identity else None,
                'resource_identity_status': 'operator_attested' if resource is not None else 'unavailable',
                'observed_host': observe_host(), 'host_id': socket.gethostname(),
                'started_at': started_at, 'finished_at': finished_at,
                'warmup_rounds': warmup, 'warmup_requests': warmup*batch_size, 'memory': memory,
                'stage_times': {'status': 'unavailable', 'reason': 'No stage instrumentation exposed'}}
    Path(str(output)+'.metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata, records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ('manifest', 'workload', 'quality', 'pricing', 'output'):
        p.add_argument('--'+field, type=Path, required=True)
    p.add_argument('--server-identity', type=Path, help='Operator-attested server deployment identity JSON, required for HTTP')
    p.add_argument('--endpoint', default='http://127.0.0.1:8080/v1/systemone')
    p.add_argument('--library', type=Path)
    p.add_argument('--model', type=Path)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--concurrency', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--duration', type=float, default=120)
    p.add_argument('--timeout', type=float, default=120)
    p.add_argument('--warmup', type=int, default=2, help='Warmup rounds; each round submits batch-size requests')
    args = p.parse_args()
    require(args.duration > 0 and math.isfinite(args.duration) and args.concurrency > 0 and args.timeout > 0 and args.threads > 0 and args.warmup >= 0, 'invalid run controls')
    require(1 <= args.batch_size <= 64, 'batch size 1..64 required')
    manifest = json.loads(args.manifest.read_text())
    validate_artifacts(manifest, args.workload, args.quality, args.pricing, args.server_identity)
    workload = load_workload(args.workload)
    require(manifest['boundary'] != 'encoder_pipeline', 'encoder_pipeline requires the Go benchmark-encoder-pipeline command')
    native = None
    server_identity = None
    if manifest['boundary'] == 'full_model':
        require(args.library is not None and args.model is not None, 'native library/model required')
        require(digest(args.model) == manifest['model_sha256'], 'model hash mismatch')
        require(digest(args.library) == manifest['implementation_sha256'], 'library hash mismatch')
        native = Native(args.library, args.model, args.threads, manifest['precision'])
        call = native
    else:
        require(args.batch_size == 1, 'HTTP runner uses server batching; client batch size must be 1')
        server_identity = validate_server_identity(manifest, args.server_identity, args.endpoint)
        server_identity['sha256'] = digest(args.server_identity)
        call = lambda row: http_call(args.endpoint, args.timeout, manifest['model_name'], row)
    try:
        for i in range(args.warmup):
            warm_rows = [workload['requests'][(i*args.batch_size+j) % len(workload['requests'])]
                         for j in range(args.batch_size)]
            call(warm_rows[0]) if args.batch_size == 1 else call.call_batch(warm_rows)
        dependencies = native.runtime_dependencies_sha256() if native else None
        if native:
            require(dependencies == manifest['runtime_dependencies_sha256'], 'actual runtime dependencies mismatch')
        else:
            dependencies = server_identity['runtime_dependencies_sha256']
        run(call, workload['requests'], args.duration, args.concurrency, args.output, digest(args.manifest), manifest['boundary'], args.threads, args.warmup, args.batch_size, manifest['resource'], dependencies, server_identity)
    finally:
        if native:
            native.close()


if __name__ == '__main__':
    main()
