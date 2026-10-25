#!/usr/bin/env python3
"""Strict artifact validation and accounting for encoder economics (stdlib only)."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, name, positive=False):
    require(type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0), name)
    return value


def load_workload(path):
    obj = json.loads(Path(path).read_text())
    require(obj.get('schema') == 'systemone-workload-v1', 'workload schema')
    rows = obj['requests']
    require(bool(rows), 'empty workload')
    seen = set()
    for row in rows:
        require(isinstance(row['id'], str) and row['id'] not in seen, 'duplicate workload id')
        seen.add(row['id'])
        require(row.get('truncated') is False, 'truncated or unverified workload')
        keys = {}
        padding_count = 0
        masks_complete = True
        require(bool(row['sequences']), 'empty sequences')
        for seq in row['sequences']:
            ids, token_keys = seq['token_ids'], seq['token_keys']
            require(bool(ids) and len(ids) == len(token_keys), 'token accounting length')
            mask = seq.get('attention_mask')
            if mask is None:
                masks_complete = False
            else:
                require(isinstance(mask, list) and len(mask) == len(ids) and all(type(v) is int and v in (0, 1) for v in mask), 'attention mask shape or value')
                require(all(key is None for key, keep in zip(token_keys, mask) if keep == 0), 'masked padding token counted as useful')
                padding_count += mask.count(0)
            require(all(type(t) is int and 0 <= t < 2**31 for t in ids), 'invalid token id')
            require(seq['question_type'] in (0, 1, 2), 'question type')
            require(bool(seq['markers']) and all(type(m) is int and 0 <= m < len(ids) for m in seq['markers']), 'markers')
            # A null key denotes padding/special computation, never useful input.
            # Equal keys across questions represent the same logical input token.
            local = set()
            for token, key in zip(ids, token_keys):
                if key is None:
                    continue
                require(isinstance(key, str) and bool(key) and key not in local, 'duplicate token key within sequence')
                local.add(key)
                require(key not in keys or keys[key] == token, 'shared token key changes token')
                keys[key] = token
        if 'padding_tokens' in row:
            require(type(row['padding_tokens']) is int and row['padding_tokens'] >= 0, 'invalid padding count')
            require(masks_complete or row['padding_tokens'] == 0, 'nonzero padding requires explicit attention masks')
            require(row['padding_tokens'] == padding_count, 'padding count mismatch')
        require(bool(keys), 'no useful input tokens')
        row['_useful_tokens'] = len(keys)
        row['_compute_tokens'] = sum(len(s['token_ids']) for s in row['sequences'])
        require(len(row['request']['questions']) == len(row['sequences']), 'question/sequence mismatch')
    return obj


def validate_http_response(out, expected_model, row):
    require(out.get('model') == expected_model, 'response model mismatch')
    require(set(out.get('answers', {})) == set(row['request']['questions']), 'missing or extra answers')
    for (qid, question), seq in zip(row['request']['questions'].items(), row['sequences']):
        answer = out['answers'][qid]
        require(isinstance(answer, dict) and answer.get('type') == question['type'], 'invalid answer type')
        for field in ('confidence', 'answer_confidence'):
            number(answer[field], 'invalid answer confidence')
            require(answer[field] <= 1, 'invalid answer confidence')
        if question['type'] == 'noul':
            number(answer['noul'], 'invalid noul')
            require(answer['noul'] <= 1, 'invalid noul')
        else:
            probabilities = answer['probabilities']
            require(isinstance(probabilities, dict) and len(probabilities) == len(seq['markers']), 'invalid probabilities')
            for value in probabilities.values():
                number(value, 'invalid probability')
                require(value <= 1, 'invalid probability')
            require(abs(sum(probabilities.values())-1) <= 1e-4, 'probabilities do not sum to one')
            if question['type'] == 'choice':
                require(set(probabilities) == set(question['criteria']), 'probability choices mismatch')
                require(answer['choice'] in probabilities, 'invalid choice')
    usage = out.get('usage', {})
    require(usage.get('truncated') is False and usage.get('state_tokens_dropped') == 0, 'truncated response')
    require(usage.get('input_tokens') == row['_compute_tokens'], 'tokenizer/sequence mismatch')
    require(not usage.get('collapsed_options'), 'collapsed options')
    return out


def validate_native_response(out, row):
    require(isinstance(out, list) and len(out) == len(row['sequences']), 'native output count')
    for values, seq in zip(out, row['sequences']):
        require(len(values['logits']) == len(seq['markers']) and len(values['act_logits']) == 2, 'native output shape')
        require(all(type(v) in (int, float) and math.isfinite(v) for v in values['logits']+values['act_logits']), 'nonfinite native output')


def validate_resource(resource, expected, context):
    require(isinstance(resource, dict), f'{context} resource missing')
    for key in ('provider', 'region', 'machine_type'):
        require(isinstance(resource.get(key), str) and bool(resource[key].strip()), f'{context} resource {key} missing')
        require(resource[key] == expected.get(key), f'{context} resource {key} mismatch')


def validate_runtime_dependencies(value):
    require(isinstance(value, dict) and bool(value), 'runtime dependencies missing')
    for name, sha256 in value.items():
        require(isinstance(name, str) and Path(name).name == name and name.startswith('libggml') and '.so' in name, 'runtime dependency name')
        require(isinstance(sha256, str) and len(sha256) == 64 and all(c in '0123456789abcdef' for c in sha256), 'runtime dependency hash')
    for prefix in ('libggml-cpu.so', 'libggml-base.so'):
        require(any(name == prefix or name.startswith(prefix+'.') for name in value), 'required runtime dependency missing: '+prefix)
    return value


def loaded_runtime_dependencies_sha256():
    dependencies = {}
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        path = Path(fields[5])
        name = path.name
        if not name.startswith('libggml') or '.so' not in name:
            continue
        value = digest(path)
        require(name not in dependencies or dependencies[name] == value, 'conflicting loaded runtime dependency: '+name)
        dependencies[name] = value
    return validate_runtime_dependencies(dependencies)


def pipeline_implementation_sha256(executable_sha256, native_library_sha256):
    # Match Go encoding/json.Marshal for this string-only, ASCII-key object.
    canonical = json.dumps({'executable_sha256': executable_sha256,
                            'native_library_sha256': native_library_sha256},
                           sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('ascii')
    return hashlib.sha256(canonical).hexdigest()


def binding_keys(manifest):
    base = ('model_sha256', 'tokenizer_sha256', 'implementation_sha256', 'precision', 'workload_sha256', 'runtime_dependencies_sha256')
    return base + (('executable_sha256', 'native_library_sha256') if manifest['boundary'] == 'encoder_pipeline' else ())


def validate_server_identity(manifest, identity_path, endpoint=None):
    require(identity_path is not None, 'HTTP server identity artifact required')
    require(digest(identity_path) == manifest.get('server_identity_sha256'), 'HTTP server identity artifact hash mismatch')
    identity = json.loads(Path(identity_path).read_text())
    require(identity.get('origin') == 'operator_attested_server_deployment', 'HTTP server identity origin')
    require(isinstance(identity.get('endpoint'), str) and bool(identity['endpoint']), 'HTTP server endpoint missing')
    require(identity['endpoint'] == manifest.get('endpoint'), 'HTTP server manifest endpoint mismatch')
    if endpoint is not None:
        require(identity['endpoint'] == endpoint, 'HTTP server CLI endpoint mismatch')
    for key in ('model_sha256', 'tokenizer_sha256', 'implementation_sha256', 'precision', 'runtime_dependencies_sha256'):
        require(identity.get(key) == manifest[key], f'HTTP server identity {key} mismatch')
    validate_runtime_dependencies(identity['runtime_dependencies_sha256'])
    return identity


def validate_artifacts(manifest, workload_path, quality_path, pricing_path, server_identity_path=None):
    for key, path in [('workload_sha256', workload_path), ('quality_sha256', quality_path), ('pricing_sha256', pricing_path)]:
        require(manifest[key] == digest(path), f'{key} mismatch')
    workload = json.loads(Path(workload_path).read_text())
    for key in ('model_sha256', 'tokenizer_sha256'):
        require(workload.get('provenance', {}).get(key) == manifest[key], f'workload provenance {key} mismatch')
    require(manifest['boundary'] in ('http_e2e', 'full_model', 'encoder_pipeline'), 'measurement boundary')
    require(manifest.get('result_cache') is False, 'result cache must be disabled')
    validate_runtime_dependencies(manifest.get('runtime_dependencies_sha256'))
    for key in ('model_sha256', 'tokenizer_sha256', 'implementation_sha256') + (('executable_sha256', 'native_library_sha256') if manifest['boundary'] == 'encoder_pipeline' else ()):
        v = manifest[key]
        require(isinstance(v, str) and len(v) == 64 and all(c in '0123456789abcdef' for c in v), key)
    if manifest['boundary'] == 'encoder_pipeline':
        require(manifest['implementation_sha256'] == pipeline_implementation_sha256(manifest['executable_sha256'], manifest['native_library_sha256']), 'pipeline implementation composite mismatch')
    quality = json.loads(Path(quality_path).read_text())
    for key in binding_keys(manifest):
        require(quality[key] == manifest[key], f'quality {key} mismatch')
    require(quality.get('status') == 'pass', 'quality not passed')
    evidence = Path(quality_path).parent / quality['evaluation_artifact']
    require(digest(evidence) == quality['evaluation_artifact_sha256'], 'quality evidence hash mismatch')
    evaluation = json.loads(evidence.read_text())
    require(evaluation.get('status') == 'pass', 'independent evaluation not passed')
    require(evaluation.get('runtime_dependencies_stable') is True, 'evaluation runtime dependencies stability unverified')
    for key in binding_keys(manifest):
        require(evaluation.get(key) == manifest[key], f'evaluation {key} mismatch')
    price = json.loads(Path(pricing_path).read_text())
    require(price.get('currency') == 'USD', 'price currency')
    for key in ('snapshot_id', 'captured_at', 'purchase_basis', 'provider', 'region', 'machine_type', 'source_url'):
        require(bool(price.get(key)), f'price {key}')
    require(price.get('includes_all_instance_and_disk_costs') is True, 'incomplete instance price')
    number(price['hourly_usd'], 'hourly price', positive=True)
    validate_resource(manifest.get('resource'), price, 'manifest/price')
    if manifest['boundary'] == 'http_e2e':
        validate_server_identity(manifest, server_identity_path)
    return quality, price


def summarize(manifest, workload, metadata, records, price):
    require(metadata['manifest_sha256'] == manifest['_sha256'], 'run manifest mismatch')
    require(metadata['boundary'] == manifest['boundary'], 'run boundary mismatch')
    validate_runtime_dependencies(metadata.get('runtime_dependencies_sha256'))
    require(metadata['runtime_dependencies_sha256'] == manifest['runtime_dependencies_sha256'], 'actual runtime dependencies mismatch')
    if manifest['boundary'] == 'http_e2e':
        require(metadata.get('runtime_dependencies_origin') == 'operator_attested_server_deployment', 'HTTP server dependency origin mismatch')
        require(bool(manifest.get('server_identity_sha256')) and metadata.get('server_identity_sha256') == manifest['server_identity_sha256'], 'HTTP server identity metadata mismatch')
        require(metadata.get('endpoint') == manifest.get('endpoint') and bool(manifest.get('endpoint')), 'HTTP endpoint metadata mismatch')
    else:
        require(metadata.get('runtime_dependencies_origin') == 'observed_process_maps', 'native runtime dependency origin mismatch')
        require(metadata.get('runtime_dependencies_stable') is True, 'runtime dependencies stability unverified')
    if manifest['boundary'] == 'encoder_pipeline':
        for key in ('model_sha256', 'tokenizer_sha256', 'workload_sha256', 'precision', 'executable_sha256', 'native_library_sha256'):
            require(metadata.get(key) == manifest.get(key) and bool(manifest.get(key)), f'actual pipeline {key} mismatch')
        require(metadata.get('native_library', {}).get('sha256') == metadata['native_library_sha256'], 'actual loaded native library mismatch')
        actual_implementation = pipeline_implementation_sha256(metadata['executable_sha256'], metadata['native_library_sha256'])
        require(metadata.get('implementation_sha256') == manifest['implementation_sha256'] == actual_implementation, 'actual pipeline implementation mismatch')
    validate_resource(manifest.get('resource'), price, 'manifest/price')
    validate_resource(metadata.get('resource'), manifest['resource'], 'run/manifest')
    require(metadata.get('resource_identity_status') == 'operator_attested', 'resource identity attestation missing')
    duration = number(metadata['elapsed_s'], 'elapsed', positive=True)
    stop = number(metadata['submission_stop_s'], 'submission stop')
    require(stop <= duration, 'drain omitted from duration')
    require(metadata['record_count'] == len(records) and bool(records), 'missing request records')
    by_id = {r['id']: r for r in workload['requests']}
    attempts, completed, useful, computed, decisions = set(), [], 0, 0, 0
    for rec in records:
        require(rec.get('run_id') == metadata.get('run_id') and bool(metadata.get('run_id')), 'request run identity mismatch')
        require(type(rec['attempt_id']) is int and rec['attempt_id'] not in attempts, 'duplicate attempt')
        attempts.add(rec['attempt_id'])
        require(rec.get('unresolved_backend_work') is False, 'backend drain unverified after transport failure')
        row = by_id[rec['workload_id']]
        start, end = number(rec['start_s'], 'start'), number(rec['end_s'], 'end')
        require(start <= stop and start <= end <= duration, 'request outside cost interval')
        require(rec['useful_tokens'] == row['_useful_tokens'] and rec['compute_tokens'] == row['_compute_tokens'], 'token count mismatch')
        require(type(rec['success']) is bool, 'success flag')
        if rec['success']:
            require(rec.get('error') is None, 'successful request has error')
            if manifest['boundary'] == 'full_model':
                validate_native_response(rec['response'], row)
            else:
                validate_http_response(rec['response'], manifest['model_name'], row)
            useful += row['_useful_tokens']
            computed += row['_compute_tokens']
            decisions += len(row['sequences'])
            completed.append(end-start)
    require(attempts == set(range(len(records))), 'missing attempts')
    cost = duration * price['hourly_usd'] / 3600
    ordered = sorted(completed)
    pct = lambda q: ordered[min(len(ordered)-1, math.ceil(q*len(ordered))-1)] if ordered else None
    return {'status': 'measured', 'boundary': manifest['boundary'], 'requests': len(records),
            'successful_requests': len(completed), 'error_rate': 1-len(completed)/len(records),
            'useful_tokens': useful, 'successful_compute_tokens': computed, 'decisions': decisions,
            'elapsed_s': duration, 'drain_s': duration-stop, 'cost_usd': cost,
            'tokens_per_s': useful/duration, 'tokens_per_dollar': useful/cost,
            'decisions_per_s': decisions/duration,
            'latency_s': {'p50': pct(.5), 'p95': pct(.95), 'p99': pct(.99)},
            'comparators': {name: {'status': 'unavailable', 'reason': 'No bound comparable run supplied'}
                            for name in ('densecore_baseline', 'llama_cpp', 'onnxruntime', 'public_gpu')},
            'gpu_2x_verdict': 'unavailable'}
