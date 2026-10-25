#!/usr/bin/env python3
"""Replay saved independent FP32 fixtures; no upstream code or Torch execution."""
import argparse
import ctypes as C
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks' / 'economics'))
from systemone_economics import loaded_runtime_dependencies_sha256


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def probability(values, temperature=1.0):
    maximum = max(values)
    exponentials = [math.exp((x - maximum) / temperature) for x in values]
    total = sum(exponentials)
    return [x / total for x in exponentials]


class Input(C.Structure):
    _fields_ = [('tokens', C.POINTER(C.c_int32)), ('token_count', C.c_size_t),
                ('question_type', C.c_int), ('markers', C.POINTER(C.c_int32)), ('marker_count', C.c_size_t)]


class Output(C.Structure):
    _fields_ = [('logits', C.POINTER(C.c_float)), ('act_logits', C.POINTER(C.c_float))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['model', 'library', 'fixtures', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--model-sha256', required=True)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    actual_sha = digest(args.model)
    if actual_sha != args.model_sha256:
        raise ValueError('model checksum mismatch')
    fixture = json.loads(args.fixtures.read_text())
    cases = fixture['cases']
    if not cases or len(cases) > 64:
        raise ValueError('fixture count must be1..64')
    for case in cases:
        count = len(case['markers'])
        if count < 1 or len(case['reference_logits']) != count or len(case['reference_probabilities']) != count or len(case['reference_act_logits']) != 2:
            raise ValueError('invalid independent fixture output shapes')
        temperature = case.get('temperature', 1.0)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('invalid fixture temperature')
    lib = C.CDLL(str(args.library.resolve()))
    lib.DenseCoreDecisionLoad.argtypes = [C.c_char_p, C.c_int, C.c_char_p, C.c_size_t]
    lib.DenseCoreDecisionLoad.restype = C.c_void_p
    lib.DenseCoreDecisionFree.argtypes = [C.c_void_p]
    lib.DenseCoreDecisionMetadata.argtypes = [C.c_void_p]
    lib.DenseCoreDecisionMetadata.restype = C.c_char_p
    lib.DenseCoreDecisionPredictBatch.argtypes = [C.c_void_p, C.POINTER(Input), C.c_size_t, C.POINTER(Output), C.c_char_p, C.c_size_t]
    lib.DenseCoreDecisionPredictBatch.restype = C.c_int
    error = C.create_string_buffer(4096)
    handle = lib.DenseCoreDecisionLoad(str(args.model).encode(), args.threads, error, len(error))
    if not handle:
        raise RuntimeError(error.value.decode())
    try:
        dependencies = loaded_runtime_dependencies_sha256()
    except Exception:
        lib.DenseCoreDecisionFree(handle)
        raise
    report = {'library': str(args.library), 'library_sha256': digest(args.library),
              'checker_sha256': digest(Path(__file__)),
              'model': str(args.model), 'model_sha256': actual_sha, 'fixtures': str(args.fixtures),
              'fixtures_sha256': digest(args.fixtures), 'threads': args.threads,
              'runtime_dependencies_sha256': dependencies,
              'runtime_dependencies_origin': 'observed_process_maps',
              'limits': {'logit_atol': 1e-3, 'probability_atol': 1e-4, 'action_atol': 1e-3,
                         'action_rtol': 1e-6, 'action_probability_atol': 1e-4}, 'runs': []}
    passed = True
    try:
        report['metadata'] = json.loads(lib.DenseCoreDecisionMetadata(handle))
        if report['metadata'].get('densecore.execution_precision') != 'fp32':
            raise ValueError('independent strict replay requires FP32 candidate')
        if fixture.get('repository') == 'SupersonicLabs/Julia-1' and report['metadata'].get('general.architecture') != 'julia':
            raise ValueError('Julia fixture cannot qualify another architecture')
        if fixture.get('repository') == 'SupersonicLabs/Julia-1':
            if report['metadata'].get('julia.source_sha256') != fixture.get('weights_sha256'):
                raise ValueError('Julia reference and GGUF source checkpoint differ')
            lib.DenseCoreDecisionTokenizerJSON.argtypes = [C.c_void_p]
            lib.DenseCoreDecisionTokenizerJSON.restype = C.c_char_p
            tokenizer = lib.DenseCoreDecisionTokenizerJSON(handle)
            if hashlib.sha256(tokenizer).hexdigest() != fixture.get('tokenizer_sha256'):
                raise ValueError('Julia reference and GGUF tokenizers differ')
        def execute(indices, label):
            nonlocal passed
            inputs = (Input * len(indices))()
            outputs = (Output * len(indices))()
            keep = []
            observed = []
            for i, index in enumerate(indices):
                case = cases[index]
                tokens = (C.c_int32 * len(case['token_ids']))(*case['token_ids'])
                markers = (C.c_int32 * len(case['markers']))(*case['markers'])
                logits = (C.c_float * len(case['markers']))()
                acts = (C.c_float * 2)()
                inputs[i] = Input(tokens, len(tokens), case['question_type'], markers, len(markers))
                outputs[i] = Output(logits, acts)
                keep.append((tokens, markers, logits, acts))
            if not lib.DenseCoreDecisionPredictBatch(handle, inputs, len(indices), outputs, error, len(error)):
                raise RuntimeError(error.value.decode())
            for index, (_, _, logits, acts) in zip(indices, keep):
                case = cases[index]
                logits, acts = list(logits), list(acts)
                logit_error = max(abs(x - y) for x, y in zip(logits, case['reference_logits']))
                prob_error = max(abs(x - y) for x, y in zip(probability(logits, case.get('temperature', 1.0)), case['reference_probabilities']))
                act_error = max(abs(x - y) for x, y in zip(acts, case['reference_act_logits']))
                act_pass = all(abs(x - y) <= 1e-3 + 1e-6 * abs(y) for x, y in zip(acts, case['reference_act_logits']))
                act_prob_error = max(abs(x - y) for x, y in zip(probability(acts), probability(case['reference_act_logits'])))
                success = all(math.isfinite(v) for v in logits + acts) and logit_error <= 1e-3 and prob_error <= 1e-4 and act_pass and act_prob_error <= 1e-4
                passed &= success
                observed.append({'case': case['name'], 'tokens': len(case['token_ids']), 'markers': case['markers'],
                    'native_logits': logits, 'native_action_logits': acts, 'max_abs_logit': logit_error,
                    'max_abs_probability': prob_error, 'max_abs_action_logit': act_error,
                    'max_abs_action_probability': act_prob_error, 'passed': success})
            report['runs'].append({'label': label, 'indices': indices, 'results': observed})
        for i, case in enumerate(cases):
            execute([i], 'single-' + case['name'])
            execute([i, i], 'batch2-' + case['name'])
        execute(list(range(len(cases))), 'mixed')
        execute(list(reversed(range(len(cases)))), 'mixed-reversed')
        longest = max(range(len(cases)), key=lambda i: len(cases[i]['token_ids']))
        shortest = min(range(len(cases)), key=lambda i: len(cases[i]['token_ids']))
        execute([longest], 'long-before-short')
        execute([shortest], 'short-after-long')
        if loaded_runtime_dependencies_sha256() != dependencies:
            raise ValueError('runtime dependencies changed during independent replay')
        report['runtime_dependencies_stable'] = True
    finally:
        lib.DenseCoreDecisionFree(handle)
    report['passed'] = passed
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'passed': passed, 'runs': len(report['runs']), 'output': str(args.output)}))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
