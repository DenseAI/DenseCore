#!/usr/bin/env python3
"""Convert the immutable Julia-1 F32 checkpoint to canonical GGUF, losslessly.

No upstream Python is imported or executed. Every learned tensor is copied
byte-for-byte; only generated RoPE buffers and explicit inference metadata are
added. Native numerical compatibility requires a separate reference parity run.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile

from download_julia_model import FILES, REPOSITORY, REVISION, verify

# Actual pinned ModernBERT initialization: CPU PyTorch 2.8.0 / Transformers 5.0.0.
ROPE_SHA256 = 'adb41194debe8e6c185d47754f68cc2dd59960a03cb4ccbc71acd14b860c4604'


def expected_shapes():
    d, inter = 384, 1152
    shapes = {'encoder.embeddings.tok_embeddings.weight': [256000, d],
              'encoder.embeddings.norm.weight': [d], 'encoder.final_norm.weight': [d],
              'type_emb.weight': [3, d], 'temperature': [3],
              'scorer.0.weight': [d], 'scorer.0.bias': [d],
              'scorer.1.weight': [d, d], 'scorer.1.bias': [d],
              'scorer.3.weight': [1, d], 'scorer.3.bias': [1],
              'act_head.0.weight': [256, d + 4], 'act_head.0.bias': [256],
              'act_head.2.weight': [2, 256], 'act_head.2.bias': [2]}
    for i in range(22):
        p = f'encoder.layers.{i}'
        shapes.update({f'{p}.attn.Wqkv.weight': [3*d, d], f'{p}.attn.Wo.weight': [d, d],
                       f'{p}.mlp.Wi.weight': [2*inter, d], f'{p}.mlp.Wo.weight': [d, inter],
                       f'{p}.mlp_norm.weight': [d]})
        if i:
            shapes[f'{p}.attn_norm.weight'] = [d]
    for i in range(2):
        p = f'head.layers.{i}'
        for norm in ('norm1', 'norm2'):
            shapes[f'{p}.{norm}.weight'] = [d]
            shapes[f'{p}.{norm}.bias'] = [d]
        shapes.update({f'{p}.self_attn.in_proj_weight': [3*d, d], f'{p}.self_attn.in_proj_bias': [3*d],
                       f'{p}.self_attn.out_proj.weight': [d, d], f'{p}.self_attn.out_proj.bias': [d],
                       f'{p}.linear1.weight': [4*d, d], f'{p}.linear1.bias': [4*d],
                       f'{p}.linear2.weight': [d, 4*d], f'{p}.linear2.bias': [d]})
    return shapes


def checkpoint_header(path):
    with path.open('rb') as source:
        raw = source.read(8)
        if len(raw) != 8:
            raise ValueError('truncated safetensors header')
        length = struct.unpack('<Q', raw)[0]
        if not 2 <= length <= 1024 * 1024:
            raise ValueError('invalid safetensors header length')
        header = json.loads(source.read(length))
    tensors = {k: v for k, v in header.items() if k != '__metadata__'}
    expected = expected_shapes()
    if set(tensors) != set(expected):
        raise ValueError(f'tensor names differ: missing={set(expected)-set(tensors)}, extra={set(tensors)-set(expected)}')
    size = path.stat().st_size - 8 - length
    ranges = []
    for name, tensor in tensors.items():
        start, end = tensor['data_offsets']
        if tensor['dtype'] != 'F32' or tensor['shape'] != expected[name]:
            raise ValueError(f'unsupported tensor dtype/shape: {name}')
        if start < 0 or end < start or end > size or end-start != math.prod(expected[name])*4:
            raise ValueError(f'invalid tensor byte range: {name}')
        ranges.append((start, end))
    ranges.sort()
    if ranges[0][0] != 0 or ranges[-1][1] != size or any(a[1] != b[0] for a, b in zip(ranges, ranges[1:])):
        raise ValueError('safetensors payload has overlapping or unaccounted bytes')
    return tensors, 8 + length


def metadata(root):
    fields = {'general.architecture': 'julia', 'general.name': 'Julia-1', 'general.alignment': 32,
              'general.source.huggingface.repository': REPOSITORY, 'general.source.huggingface.revision': REVISION,
              'julia.weight_layout': 'ggml-canonical', 'julia.source_sha256': FILES['model.safetensors'][1],
              'julia.tokenizer_sha256': FILES['tokenizer/tokenizer.json'][1],
              'julia.layer_norm_epsilon': 1e-5, 'julia.attention_scale': 64 ** -.25,
              'julia.temperature.0': 1.0, 'julia.temperature.1': 1.0, 'julia.temperature.2': 1.0,
              'julia.probability_policy': 'raw-softmax-no-calibration',
              'julia.upstream_quality_gate': 'false',
              'tokenizer.huggingface.json': (root/'tokenizer/tokenizer.json').read_text()}
    for key, value in {'hidden_size':384, 'num_layers':22, 'num_heads':6, 'head_dim':64,
                       'intermediate_size':1152, 'vocab_size':256000, 'sliding_window':64,
                       'global_layer_every':3, 'head_layers':2, 'head_intermediate_size':1536,
                       'act_hidden_size':256, 'act_input_features':388, 'question_types':3,
                       'max_len':8192, 'head_max_len':512}.items():
        fields['julia.'+key] = value
    for key, value in {'bos_token_id':2, 'cls_token_id':2, 'eos_token_id':1,
                       'sep_token_id':1, 'mask_token_id':4, 'padding_token_id':0}.items():
        fields['tokenizer.ggml.'+key] = value
    return fields


def string(value):
    encoded = value.encode('utf-8')
    return struct.pack('<Q', len(encoded)) + encoded


def kv(key, value):
    if isinstance(value, str):
        kind, raw = 8, string(value)
    elif isinstance(value, int):
        kind, raw = 4, struct.pack('<I', value)
    elif isinstance(value, float):
        kind, raw = 6, struct.pack('<f', value)
    else:
        raise ValueError('unsupported GGUF value')
    return string(key) + struct.pack('<I', kind) + raw


def convert(root, destination, rope_path):
    for name in ['model.safetensors','tokenizer/tokenizer.json','encoder/config.json','julia_config.json','inference-policy.json']:
        if not verify(root/name, *FILES[name]):
            raise ValueError(f'missing or mismatched pinned input: {root/name}')
    if destination.exists():
        raise ValueError(f'refusing to replace existing output: {destination}')
    # Exact FP32 frequencies must come from the pinned independent reference
    # initialization, not a double-precision Python approximation to torch.pow.
    rope = json.loads(rope_path.read_text())
    if rope.get('source_revision') != REVISION or rope.get('encoder_config_sha256') != FILES['encoder/config.json'][1]:
        raise ValueError('RoPE reference provenance mismatch')
    frequency = {}
    for name in ['rope.freq_full','rope.freq_sliding']:
        values = rope['frequencies'][name]
        if len(values) != 32 or any(not math.isfinite(x) or x <= 0 for x in values):
            raise ValueError('invalid reference RoPE frequencies')
        frequency[name] = struct.pack('<32f', *values)
        if hashlib.sha256(frequency[name]).hexdigest() != ROPE_SHA256:
            raise ValueError('RoPE frequency bytes differ from pinned reference initialization')
    tensors, source_offset = checkpoint_header(root/'model.safetensors')
    fields = metadata(root)
    descriptors, offset = [], 0
    for name in sorted(set(tensors)-{'temperature'} | set(frequency)):
        shape = tensors[name]['shape'] if name in tensors else [32]
        size = math.prod(shape)*4
        descriptors.append((name, shape, size, offset))
        offset = (offset + size + 31) // 32 * 32
    header = struct.pack('<4sIQQ', b'GGUF', 3, len(descriptors), len(fields))
    header += b''.join(kv(key, value) for key, value in fields.items())
    for name, shape, size, offset in descriptors:
        header += string(name) + struct.pack('<I', len(shape))
        header += b''.join(struct.pack('<Q', dim) for dim in reversed(shape))
        header += struct.pack('<IQ', 0, offset)  # GGML_TYPE_F32
    header += bytes((-len(header)) % 32)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.part', delete=False) as sink, (root/'model.safetensors').open('rb') as source:
            temporary = Path(sink.name)
            sink.write(header)
            for name, shape, size, offset in descriptors:
                if name in tensors:
                    source.seek(source_offset+tensors[name]['data_offsets'][0])
                    remaining = size
                    while remaining:
                        chunk = source.read(min(1024*1024, remaining))
                        if not chunk:
                            raise ValueError('truncated source tensor')
                        sink.write(chunk)
                        remaining -= len(chunk)
                else:
                    sink.write(frequency[name])
                sink.write(bytes((-size) % 32))
        temporary.chmod(0o644)
        destination.hardlink_to(temporary)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    with destination.open('rb') as result:
        digest = hashlib.file_digest(result, 'sha256').hexdigest()
    return {'schema_version':1, 'repository':REPOSITORY, 'revision':REVISION,
            'source_sha256':FILES['model.safetensors'][1], 'gguf':str(destination.resolve()),
            'gguf_sha256':digest, 'gguf_bytes':destination.stat().st_size,
            'copied_tensors':len(tensors)-1, 'generated_tensors':list(frequency),
            'tensor_identity':'all learned tensors copied byte-for-byte; unused temperature omitted',
            'precision':'F32', 'independent_numerical_parity':'pending', 'upstream_quality_gate':False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rope-reference', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = convert(args.model, args.output, args.rope_reference)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
