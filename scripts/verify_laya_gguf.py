#!/usr/bin/env python3
"""Independent upstream PyTorch versus native Laya parity (no model downloads).

Requires torch, transformers, safetensors, gguf, numpy. Pass the pinned upstream
common.py, checkpoint, encoder config and agent config, plus the native library.
Runs FP32 arithmetic on the published F16 weights. Fixed acceptance uses absolute
option/probability limits and a mixed absolute/relative act-logit limit, documented
with the original strict-gate failures in benchmarks/fixtures/laya/.
"""
import argparse
import ctypes as C
import hashlib
import importlib.util
import json
import platform
from pathlib import Path

PINNED = {
    'common': 'cb77c34b3b5abfc1f59eb1a73357ad80238df397ffdddcfaf634c01949f89b3f',
    'checkpoint': '891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c',
    'encoder_config': 'bf3ab80598fdccf414855a2ce80f22859e4492d06ca8a62ddd1cfb63972f8979',
    'agent_config': 'ae287b56bbcf5f8c4f4541ae9dfd00c914c4c48b940b8398c3058af37ba92bbd',
    'gguf': '62db2affc0fe4b9f2538ba61299e5fd8bf876f50f765089a343cd4113eef21a4',
}


def fixtures():
    choice = {'t': 'choice', 'ins': 'Classify the support request.', 'crit': {'billing': 'Payment or invoice', 'technical': 'Software problem', 'other': 'Other request'}}
    return [
        ('choice', 'I was charged twice for my subscription.', choice),
        ('score', 'The response fully addresses the question and cites evidence.', {'t': 'score', 'ins': 'Rate the answer quality.', 'crit': ['poor', 'adequate', 'excellent']}),
        ('noul_unicode', '서울에서 결제한 금액을 환불해 주세요. café 😊', {'t': 'noul', 'ins': 'The customer requests a refund.'}),
        ('single_option', 'A routine message.', {'t': 'choice', 'ins': 'Choose the category.', 'crit': {'message': ''}}),
        ('local_attention', 'An invoice was received. ' * 35, choice),
        ('choice_12', 'An invoice was received.', {'t': 'choice', 'ins': 'Choose the category.', 'crit': {f'category{i}': f'Category number {i}' for i in range(12)}}),
        ('choice_reordered', 'I was charged twice for my subscription.', {**choice, 'crit': dict(reversed(list(choice['crit'].items())))}),
        ('max_length', 'An invoice was received. ' * 150, choice),
        ('choice_after_max_length', 'I was charged twice for my subscription.', choice),
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in PINNED:
        p.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--fixtures-output', type=Path)
    args = p.parse_args()
    import numpy as np
    import torch
    import transformers
    from gguf import GGUFReader
    from safetensors.torch import load_file
    from transformers import ModernBertConfig, ModernBertModel, PreTrainedTokenizerFast

    for name, expected in PINNED.items():
        with getattr(args, name).open('rb') as f:
            actual = hashlib.file_digest(f, 'sha256').hexdigest()
        if actual != expected:
            raise ValueError(f'{name}: unrecognized input SHA256 {actual}')
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    spec = importlib.util.spec_from_file_location('laya_pinned_reference', args.common)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    reader = GGUFReader(str(args.gguf))
    state = load_file(str(args.checkpoint), device='cpu')
    # Every stored learned tensor must exactly equal the independent checkpoint.
    names = set(state) - {'temperature'}
    compared = 0
    for tensor in reader.tensors:
        if tensor.name in ('rope.freq_full', 'rope.freq_sliding'):
            continue
        if tensor.name not in names:
            raise ValueError(f'Unexpected GGUF tensor: {tensor.name}')
        np.testing.assert_array_equal(tensor.data, state[tensor.name].numpy())
        names.remove(tensor.name)
        compared += 1
    if names:
        raise ValueError(f'Missing GGUF tensors: {names}')
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=__import__('tokenizers').Tokenizer.from_str(
        reader.fields['tokenizer.huggingface.json'].contents()),
        cls_token='[CLS]', sep_token='[SEP]', mask_token='[MASK]', pad_token='[PAD]', unk_token='[UNK]')
    config = ModernBertConfig.from_dict(json.loads(args.encoder_config.read_text()))
    config._attn_implementation = 'eager'
    # Keep non-persistent RoPE buffers real while skipping random weight initialization.
    with upstream._no_init_weights():
        encoder = ModernBertModel(config)
    reference = upstream.DecisionModel(encoder, head_layers=2, no_init=True)
    reference.load_state_dict({k: v.float() for k, v in state.items()}, strict=True)
    del state
    reference.eval()
    agent_config = json.loads(args.agent_config.read_text())
    for index, expected in enumerate(agent_config['temperature']):
        np.testing.assert_allclose(reader.fields[f'laya.temperature.{index}'].contents(), expected, rtol=1e-7, atol=0)
    buckets = json.loads(reader.fields['laya.temperature_by_options'].contents())
    assert set(buckets) == set(agent_config['temperature_by_options'])
    for key, expected in agent_config['temperature_by_options'].items():
        np.testing.assert_allclose(buckets[key], expected, rtol=1e-7, atol=0)
    lib = C.CDLL(str(args.library.resolve()))
    lib.DenseCoreDecisionLoad.argtypes = [C.c_char_p, C.c_int, C.c_char_p, C.c_size_t]
    lib.DenseCoreDecisionLoad.restype = C.c_void_p
    lib.DenseCoreDecisionFree.argtypes = [C.c_void_p]
    lib.DenseCoreDecisionPredict.argtypes = [C.c_void_p, C.POINTER(C.c_int32), C.c_size_t, C.c_int,
        C.POINTER(C.c_int32), C.c_size_t, C.POINTER(C.c_float), C.POINTER(C.c_float), C.c_char_p, C.c_size_t]
    lib.DenseCoreDecisionPredict.restype = C.c_int
    error = C.create_string_buffer(2048)
    handle = lib.DenseCoreDecisionLoad(str(args.gguf).encode(), args.threads, error, len(error))
    if not handle:
        raise RuntimeError(error.value.decode())
    with args.library.open('rb') as binary:
        library_sha256 = hashlib.file_digest(binary, 'sha256').hexdigest()
    report = {'inputs_sha256': PINNED, 'native_library_sha256': library_sha256,
              'platform': platform.platform(), 'python': platform.python_version(), 'versions': {'torch': torch.__version__, 'transformers': transformers.__version__},
              'threads': args.threads, 'learned_tensors_exact': compared,
              'limits': {'max_abs_logit': 1e-3, 'max_abs_probability': 1e-4, 'act_logit_atol': 1e-3, 'act_logit_rtol': 1e-6, 'max_abs_act_probability': 1e-4}, 'cases': []}
    records = []
    passed = True
    try:
        for name, text, question in fixtures():
            tokens, markers = upstream.build_sequence(tokenizer, text, question, max_len=512, head_max_len=192)
            if name == 'local_attention':
                assert 128 < len(tokens) < 512
            if name == 'max_length':
                assert len(tokens) == 512
            qtype = upstream.QTYPES[question['t']]
            ids = torch.tensor([tokens], dtype=torch.long)
            pos = torch.tensor([markers], dtype=torch.long)
            logits, act = reference(ids, torch.ones_like(ids), pos, torch.ones_like(pos, dtype=torch.bool), torch.tensor([qtype]))
            expected = logits[0].numpy()
            expected_act = act[0].numpy()
            native_logits = (C.c_float * len(markers))()
            native_act = (C.c_float * 2)()
            ok = lib.DenseCoreDecisionPredict(handle, (C.c_int32 * len(tokens))(*tokens), len(tokens), qtype,
                (C.c_int32 * len(markers))(*markers), len(markers), native_logits, native_act, error, len(error))
            if not ok:
                raise RuntimeError(error.value.decode())
            observed = np.array(native_logits, dtype=np.float32)
            observed_act = np.array(native_act, dtype=np.float32)
            count = len(markers)
            bucket = upstream.temp_bucket(qtype, count)
            temperature = upstream.clamp_temperature(agent_config['temperature_by_options'].get(bucket, agent_config['temperature'][qtype]))
            def probs(values):
                return torch.softmax(torch.from_numpy(values) / temperature, dim=0).numpy()
            logit_error = float(np.max(np.abs(observed - expected)))
            act_error = float(np.max(np.abs(observed_act - expected_act)))
            probability_error = float(np.max(np.abs(probs(observed) - probs(expected))))
            expected_act_prob = torch.softmax(torch.from_numpy(expected_act), dim=0).numpy()
            observed_act_prob = torch.softmax(torch.from_numpy(observed_act), dim=0).numpy()
            act_probability_error = float(np.max(np.abs(observed_act_prob - expected_act_prob)))
            act_pass = bool(np.all(np.abs(observed_act - expected_act) <= 1e-3 + 1e-6 * np.abs(expected_act)))
            success = logit_error <= 1e-3 and probability_error <= 1e-4 and act_pass and act_probability_error <= 1e-4
            passed &= success
            result = {'name': name, 'tokens': len(tokens), 'markers': markers, 'reference_logits': expected.tolist(),
                      'native_logits': observed.tolist(), 'reference_act_logits': expected_act.tolist(), 'native_act_logits': observed_act.tolist(),
                      'temperature': temperature, 'reference_probabilities': probs(expected).tolist(), 'native_probabilities': probs(observed).tolist(),
                      'max_abs_logit': logit_error, 'max_abs_act_logit': act_error, 'max_abs_probability': probability_error, 'max_abs_act_probability': act_probability_error,
                      'reference_act_probabilities': expected_act_prob.tolist(), 'native_act_probabilities': observed_act_prob.tolist(), 'passed': success}
            report['cases'].append(result)
            records.append({'name': name, 'state': text, 'question': question, 'token_ids': tokens, 'markers': markers,
                            'question_type': qtype, 'reference_logits': expected.tolist(), 'reference_act_logits': expected_act.tolist(),
                            'temperature': temperature, 'reference_probabilities': probs(expected).tolist()})
            print(json.dumps(result), flush=True)
    finally:
        lib.DenseCoreDecisionFree(handle)
    report['passed'] = passed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    if args.fixtures_output:
        args.fixtures_output.parent.mkdir(parents=True, exist_ok=True)
        args.fixtures_output.write_text(json.dumps({'upstream_common_sha256': PINNED['common'], 'cases': records}, ensure_ascii=False, indent=2) + '\n')
    if not passed:
        raise SystemExit('Laya parity FAILED; see report')


if __name__ == '__main__':
    main()
