#!/usr/bin/env python3
"""Strict Julia-1 input serialization and raw/display probability separation.

Implements the pinned julia/data.py sequence contract for offline preparation.
No model code is imported. Native inference and numerical parity remain pending.
"""
import argparse
import json
import math
from pathlib import Path

QTYPES = {'choice': 0, 'score': 1, 'noul': 2}


def encode_request(tokenizer, row, max_length=8192, head_length=512):
    """Prepare a lossless request using a Hugging Face tokenizers.Tokenizer.

    Overflow is rejected rather than counted as valid input tokens. The caller
    supplies the pinned tokenizer; load_tokenizer verifies it for CLI use.
    """
    if type(max_length) is not int or not 1 <= max_length <= 8192:
        raise ValueError('max_length must be in 1..8192')
    if type(head_length) is not int or head_length < 1 or head_length + 4 >= max_length:
        raise ValueError('max_length must leave room beyond the question head')
    if not isinstance(row, dict) or not isinstance(row.get('state'), (str, dict, list)) or not isinstance(row.get('question'), str):
        raise ValueError('request requires text/JSON state and text question')
    options = row.get('options')
    if not isinstance(options, list) or not 2 <= len(options) <= 20 or not all(isinstance(x, str) and x for x in options):
        raise ValueError('options must contain 2..20 nonempty descriptions')
    qtype = row.get('type', 'choice')
    if not isinstance(qtype, str) or qtype not in QTYPES or qtype == 'noul' and len(options) != 2:
        raise ValueError('invalid question type or noul option count')
    state = row['state'] if isinstance(row['state'], str) else json.dumps(row['state'], ensure_ascii=False)
    if any('<mask>' in text for text in [state, row['question'], *options]):
        raise ValueError('reserved model marker in request')
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False).ids
    head = encode(f"{qtype} question: {row['question']}")
    option_ids = [encode(' ' + text) for text in options]
    if any(len(ids) > 48 for ids in option_ids):
        raise ValueError('option exceeds 48-token model contract')
    packed = [[4] + ids for ids in option_ids]
    budget = head_length - sum(map(len, packed))
    if budget < 16:
        per_option = max(4, (head_length - 16) // len(packed))
        packed = [ids[:per_option] for ids in packed]
        budget = head_length - sum(map(len, packed))
    if len(head) > budget or any(len(ids) != len(raw) + 1 for ids, raw in zip(packed, option_ids)):
        raise ValueError('question/options exceed lossless head budget')
    ids = [2] + head[:max(8, budget)] + [1]
    markers = []
    for option in packed:
        markers.append(len(ids))
        ids.extend(option)
    ids.append(1)
    state_ids = encode(state)
    room = max_length - len(ids) - 1
    if room < 1 or len(state_ids) > room:
        raise ValueError('state exceeds lossless context budget')
    ids += state_ids + [1]
    return {'ids': ids, 'markers': markers, 'qtype': QTYPES[qtype],
            'truncated': False, 'option_tokens': list(map(len, option_ids)),
            'valid_input_tokens': len(ids)}


def probabilities(logits):
    """Return raw softmax and upstream presentation values separately.

    Selection and quality scoring must use raw probabilities. Display thresholding
    is upstream UI behavior and must not conceal calibration differences.
    """
    if not 2 <= len(logits) <= 20 or any(not math.isfinite(x) for x in logits):
        raise ValueError('logits must be finite and contain 2..20 options')
    peak = max(logits)
    weights = [math.exp(x - peak) for x in logits]
    total = sum(weights)
    raw = [x / total for x in weights]
    winner = max(range(len(raw)), key=raw.__getitem__)
    if raw[winner] > .95 and all(x < .045 for i, x in enumerate(raw) if i != winner):
        display = [1.0 if i == winner else 0.0 for i in range(len(raw))]
    else:
        display = [x if x >= .01 else 0.0 for x in raw]
        denominator = sum(display)
        display = [x / denominator for x in display]
    return {'index': winner, 'raw_probabilities': raw, 'display_probabilities': display}


def load_tokenizer(root):
    from download_julia_model import FILES, verify
    name = 'tokenizer/tokenizer.json'
    if not verify(root / name, *FILES[name]):
        raise ValueError('missing or invalid pinned Julia tokenizer')
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(root / name))
    for token, expected in [('<pad>', 0), ('<eos>', 1), ('<bos>', 2), ('<mask>', 4)]:
        if tokenizer.token_to_id(token) != expected:
            raise ValueError('unsupported Julia special token IDs')
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--max-length', type=int, default=8192)
    parser.add_argument('--head-length', type=int, default=512)
    args = parser.parse_args()
    tokenizer = load_tokenizer(args.model)
    # Validate the whole corpus before writing any output.
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    encoded = [encode_request(tokenizer, row, args.max_length, args.head_length) for row in rows]
    for row in encoded:
        print(json.dumps(row))


if __name__ == '__main__':
    main()
