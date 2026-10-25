#!/usr/bin/env python3
"""Export calibrated native encoder predictions and task accuracy from fixed inputs.

Each completed native batch is flushed to .raw.jsonl; interrupted runs preserve partial
records and never masquerade as a complete evaluator input. Existing paths are not
overwritten. This measures task predictions, not independent implementation parity.
"""
from __future__ import annotations
import argparse
import ctypes as C
import json
import math
import statistics
import time
from pathlib import Path

from run_systemone_load import Native
from evaluate_encoder_quality import quality_target
from systemone_economics import digest, load_workload, require, validate_runtime_dependencies


def calibrated_prediction(row, output, metadata):
    require(len(row['sequences']) == 1 and len(row['request']['questions']) == 1, 'one question per quality row required')
    question = next(iter(row['request']['questions'].values()))
    require(question['type'] == 'choice', 'quality exporter currently requires choice questions')
    criteria = question['criteria']
    labels = list(criteria) if isinstance(criteria, dict) else criteria
    require(len(labels) == 4, 'four options required')
    order = question.get('option_order', list(range(len(labels))))
    require(sorted(order) == list(range(len(labels))), 'invalid option order')
    logits = output[0]['logits']
    require(len(logits) == len(labels) and all(math.isfinite(v) for v in logits), 'invalid option logits')
    architecture = metadata.get('general.architecture', 'laya')
    require(architecture in ('laya', 'julia'), 'unsupported model architecture')
    if architecture == 'julia':
        require(all(metadata.get(f'julia.temperature.{i}', 1.) == 1. for i in range(3)), 'Julia requires raw softmax temperatures')
        temperature = 1.
    else:
        buckets = metadata.get('laya.temperature_by_options', {})
        if isinstance(buckets, str):
            buckets = json.loads(buckets)
        temperature = buckets.get('choice:3-5', metadata.get('laya.temperature.0', 1.))
        if not math.isfinite(temperature):
            temperature = 1.
        temperature = max(.5, min(5., temperature))
    exps = [math.exp((value-max(logits))/temperature) for value in logits]
    denominator = sum(exps)
    canonical_probs, canonical_logits = [0.] * len(labels), [0.] * len(labels)
    for i, index in enumerate(order):
        canonical_probs[index] = exps[i]/denominator
        canonical_logits[index] = logits[i]
    gold = labels.index(row['gold_label'])
    target = quality_target(row, labels)
    predicted = max(range(len(labels)), key=canonical_probs.__getitem__)
    return {'id': row['id'], 'source_id': row['source_id'], 'labels': labels,
            'logits': canonical_logits, 'act_logits': output[0]['act_logits'],
            'probabilities': canonical_probs, 'temperature': temperature,
            'gold_label': row['gold_label'], 'predicted_label': labels[predicted],
            'correct': predicted == gold,
            'brier': sum((p-target[i])**2 for i, p in enumerate(canonical_probs)),
            'gold_probabilities': target,
            'bucket_tokens': row.get('bucket_tokens')}


def prediction_batches(rows, batch_size):
    require(type(batch_size) is int and 1 <= batch_size <= 64, 'batch size must be 1..64')
    buckets = {}
    for index, row in enumerate(rows):
        require(len(row['sequences']) == 1, 'quality batching requires one sequence per request')
        length = len(row['sequences'][0]['token_ids'])
        buckets.setdefault(length, []).append((index, row))
    # Stable input order within each exact-length group, no padding.
    for length in sorted(buckets):
        group = buckets[length]
        for offset in range(0, len(group), batch_size):
            yield group[offset:offset+batch_size]


def run_predictions(native, rows, metadata, batch_size, raw, progress=None):
    predictions = [None] * len(rows)
    actual_batches = []
    completed = 0
    for batch_id, batch in enumerate(prediction_batches(rows, batch_size)):
        batch_rows = [row for _, row in batch]
        before = time.monotonic()
        outputs = [native(batch_rows[0])] if len(batch) == 1 else native.call_batch(batch_rows)
        elapsed = time.monotonic()-before
        require(len(outputs) == len(batch), 'native batch output count mismatch')
        # Calibrate the whole successful batch before recording any of it.
        decoded = [calibrated_prediction(row, out, metadata) for row, out in zip(batch_rows, outputs)]
        for (index, _), prediction in zip(batch, decoded):
            prediction.update({'batch_id': batch_id, 'actual_batch_size': len(batch),
                               'batch_elapsed_s': elapsed, 'workload_index': index})
            raw.write(json.dumps(prediction, allow_nan=False)+'\n')
            predictions[index] = prediction
        raw.flush()
        actual_batches.append(len(batch))
        completed += len(batch)
        if progress is not None:
            progress({'completed': completed, 'total': len(rows), 'batch_id': batch_id,
                      'actual_batch_size': len(batch), 'input_tokens_per_sequence': len(batch_rows[0]['sequences'][0]['token_ids'])})
    require(all(prediction is not None for prediction in predictions), 'missing exported predictions')
    return predictions, actual_batches


def export(args):
    require(type(args.batch_size) is int and 1 <= args.batch_size <= 64, 'batch size must be 1..64')
    workload = load_workload(args.workload)
    provenance = workload['provenance']
    model_hash, library_hash = digest(args.model), digest(args.library)
    require(model_hash == provenance['model_sha256'], 'actual model checksum differs from workload')
    output = Path(args.output)
    raw_path = Path(str(output) + '.raw.jsonl')
    require(not output.exists() and not raw_path.exists(), 'output/raw prediction file already exists; use a fresh output path')
    native = Native(args.library, args.model, args.threads, args.precision)
    try:
        for name in ('DenseCoreDecisionMetadata', 'DenseCoreDecisionTokenizerJSON'):
            fn = getattr(native.lib, name)
            fn.argtypes = [C.c_void_p]
            fn.restype = C.c_char_p
        metadata_raw = native.lib.DenseCoreDecisionMetadata(native.handle)
        tokenizer_raw = native.lib.DenseCoreDecisionTokenizerJSON(native.handle)
        import hashlib
        require(hashlib.sha256(tokenizer_raw).hexdigest() == provenance['tokenizer_sha256'], 'loaded model tokenizer differs from workload')
        metadata = json.loads(metadata_raw)
        architecture = metadata.get('general.architecture', 'laya')
        require(architecture in ('laya','julia'), 'unsupported loaded model architecture')
        if 'model_architecture' in provenance:
            require(provenance['model_architecture'] == architecture, 'loaded model architecture differs from workload')
        if 'densecore.execution_precision' in metadata:
            require(metadata['densecore.execution_precision'] == args.precision, 'actual loaded precision differs from requested precision')
        dependencies = validate_runtime_dependencies(native.runtime_dependencies_sha256())
        bindings = {'schema': 'densecore.encoder-predictions.v1', 'model_sha256': model_hash,
                    'tokenizer_sha256': provenance['tokenizer_sha256'], 'implementation_sha256': library_hash,
                    'model_artifact_sha256': model_hash, 'workload_sha256': digest(args.workload),
                    'precision': args.precision, 'threads': args.threads,
                    'runtime_dependencies_sha256': dependencies,
                    'model_architecture': architecture, 'model_name': 'julia-1' if architecture == 'julia' else 'laya',
                    'requested_batch_size': args.batch_size, 'batch_grouping': 'exact_token_count_no_padding',
                    'purpose': workload.get('purpose', 'task_quality_evaluation'),
                    'metadata_sha256': hashlib.sha256(metadata_raw).hexdigest()}
        started = time.monotonic()
        with raw_path.open('x', encoding='utf-8') as raw:
            raw.write(json.dumps({'bindings': bindings})+'\n'); raw.flush()
            predictions, actual_batches = run_predictions(native, workload['requests'], metadata, args.batch_size, raw,
                progress=lambda update: print(json.dumps(update), flush=True))
        require(native.runtime_dependencies_sha256() == dependencies, 'runtime dependencies changed during prediction export')
        result = {**bindings, 'runtime_dependencies_stable': True, 'predictions': predictions, 'raw_predictions_sha256': digest(raw_path),
                  'elapsed_s': time.monotonic()-started, 'complete': True,
                  'actual_batch_sizes': actual_batches,
                  'accuracy': statistics.mean(float(p['correct']) for p in predictions),
                  'brier': statistics.mean(p['brier'] for p in predictions),
                  'quality_qualification': 'not_evaluated',
                  'independent_action_parity': 'not_evaluated'}
        output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        return result
    finally:
        native.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('workload', 'library', 'model', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--precision', choices=('fp32','q8_0'), default='fp32')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=1)
    a = p.parse_args()
    require(1 <= a.threads <= 256, 'threads outside model limits')
    result = export(a)
    print(json.dumps({k: result[k] for k in ('accuracy','brier','elapsed_s','precision','quality_qualification')}))


if __name__ == '__main__': main()
