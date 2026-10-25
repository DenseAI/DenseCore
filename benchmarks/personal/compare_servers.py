#!/usr/bin/env python3
"""Record completed OpenAI chat requests, TTFT and final-answer gates (stdlib only)."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request


def final_answer(content, reasoning=''):
    # DenseCore can return the thinking block inline; never score an unfinished thought.
    if '<think>' in content:
        if '</think>' not in content:
            return ''
        content = content.rsplit('</think>', 1)[1]
    return content.strip()


def validate_response(content, reasoning, finish_reason, usage, expected):
    final = final_answer(content, reasoning)
    if not final or finish_reason != 'stop':
        raise ValueError('response has no complete final answer (reasoning-only and length-limited output fail)')
    if expected and not re.search(expected, final, flags=re.IGNORECASE):
        raise ValueError(f'final answer failed expected pattern {expected!r}')
    if not isinstance(usage.get('completion_tokens'), int) or usage['completion_tokens'] < 1:
        raise ValueError('usage.completion_tokens missing or invalid')
    return final


def request(url, prompt, max_tokens, expected, timeout=600):
    body = {'model': 'local', 'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0, 'seed': 42, 'max_tokens': max_tokens, 'stream': True,
        'stream_options': {'include_usage': True}, 'cache_prompt': False,
        'chat_template_kwargs': {'enable_thinking': False}}
    started = time.perf_counter()
    first = None
    content, reasoning, events = [], [], []
    usage, finish, done = {}, None, False
    req = urllib.request.Request(url.rstrip('/') + '/v1/chat/completions',
        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        for line in response:
            if not line.startswith(b'data:'): continue
            payload = line[5:].strip()
            if payload == b'[DONE]':
                done = True; break
            event = json.loads(payload)
            events.append({'elapsed_seconds': time.perf_counter() - started, 'event': event})
            if event.get('error'): raise ValueError(f'stream error: {event["error"]}')
            if event.get('usage'): usage = event['usage']
            for choice in event.get('choices', []):
                delta = choice.get('delta', {})
                text = delta.get('content') or ''
                thought = delta.get('reasoning_content') or ''
                if (text or thought) and first is None: first = time.perf_counter() - started
                content.append(text); reasoning.append(thought)
                if choice.get('finish_reason'): finish = choice['finish_reason']
    wall = time.perf_counter() - started
    raw = {'body': body, 'events': events, 'done': done, 'wall_seconds': wall,
        'ttft_seconds': first, 'finish_reason': finish, 'usage': usage}
    if not done: raw['quality_error'] = 'stream missing [DONE]'
    try:
        final = validate_response(''.join(content), ''.join(reasoning), finish, usage, expected)
        raw['final_answer'] = final
        raw['final_sha256'] = hashlib.sha256(final.encode()).hexdigest()
    except ValueError as error:
        raw['quality_error'] = str(error)
    return raw


def group(url, prompts, concurrency, max_tokens, expected):
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows = list(pool.map(lambda prompt: request(url, prompt, max_tokens, expected), prompts))
    wall = time.perf_counter() - start
    valid = all(not row.get('quality_error') for row in rows)
    return {'concurrency': concurrency, 'wall_seconds': wall, 'quality_pass': valid, 'requests': rows,
        'aggregate_completed_output_tps': sum(row['usage'].get('completion_tokens', 0) for row in rows) / wall if valid else None}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--prompts', type=Path, required=True, help='JSON array of prompt strings')
    parser.add_argument('--expected-pattern', required=True, help='Pattern required in final answer, never reasoning')
    parser.add_argument('--concurrency', type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    prompts = json.loads(args.prompts.read_text())
    if not isinstance(prompts, list) or not prompts or not all(isinstance(p, str) and p for p in prompts):
        parser.error('--prompts must contain a nonempty JSON list of strings')
    result = group(args.url, prompts, args.concurrency, args.max_tokens, args.expected_pattern)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    raise SystemExit(0 if result['quality_pass'] else 1)
