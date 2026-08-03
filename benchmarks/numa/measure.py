#!/usr/bin/env python3
"""Token-accurate, cross-engine measurement for OpenAI-compatible /v1/chat/completions.

Fixes the chunk-counting flaw: we use NON-STREAM responses and read the real token
counts from `usage` (prompt_tokens / completion_tokens), which both DenseCore and
llama-server return. Decode rate is isolated from prefill with a two-call method:

  prefill probe : max_tokens=1  -> wall t_pf,  usage.prompt_tokens P
  decode run    : max_tokens=N  -> wall t_N,   usage.completion_tokens G  (require G>=min)
  decode_tps    = (G - 1) / (t_N - t_pf)        # token-accurate, no SSE chunk counting
  prefill_tps   = P / t_pf                       # t_pf ~= prefill (+1 trivial decode step)

Hard-fails (exit!=0) on HTTP error, missing usage, empty content, or G<min_tokens, so a
broken server can NEVER be silently scored or pass the parity gate on empty output.

stdlib only.
"""
import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request


class ReqError(Exception):
    pass


def read_prompts(path):
    with open(path, "r", encoding="utf-8") as f:
        parts = [p.strip() for p in f.read().split("\n===\n")]
    return [p for p in parts if p]


def extract_text(choice):
    """Return (text, channel) for one choice.

    Reasoning models (Qwen3.x and friends) route generated tokens into
    `reasoning_content` and leave `message.content` empty when the whole token
    budget is spent inside the thinking block. Both DenseCore and llama-server
    do this, so reading only `content` reports "empty content" for a server that
    generated hundreds of perfectly good tokens. Those tokens are real decode
    work and are already counted in `usage.completion_tokens`, so they must
    count here too or the decode rate is thrown away for a non-problem.
    """
    msg = choice.get("message") or {}
    content = (msg.get("content") or "").strip()
    reasoning = (msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
    # `text` is the legacy completions-style field some servers still emit.
    legacy = (choice.get("text") or "").strip()
    if content:
        return content, ("content+reasoning" if reasoning else "content")
    if reasoning:
        return reasoning, "reasoning_only"
    if legacy:
        return legacy, "text"
    return "", "empty"


def _norm(s):
    return " ".join(s.lower().split())


def output_defects(prompt, text, no_thinking=False):
    """Mechanical checks for output that is *present* but not an answer.

    Every existing check here asks "did tokens come out" (non-empty, >= min_tokens,
    decode window > 0). A server that replays the prompt back passes all of them, and
    so does the engine's own telemetry: the run in NUMA_TIER1_FINDINGS.md section 3.5
    reported model_execution_contract_valid=1 and graph_kernel_coverage=100 while
    returning turn 1's user text on every turn. Nothing downstream could tell.

    The parity gate cannot catch it either: it diffs N1 against B0, and two runs of the
    same broken engine echo *identically*, so the gate goes green. Detection has to be
    per-response and independent of any comparison.

    No model-as-judge; these are the two shapes actually observed. Deliberately
    conservative -- a false abort on an expensive run is its own failure. In particular
    containment is required in the whole-output direction only, so a legitimate answer
    that quotes the prompt (the quicksort stub in parity_prompts.txt is completed
    verbatim by a *correct* answer) is not flagged.
    """
    defects = []
    p, t = _norm(prompt), _norm(text)
    words = t.split()
    # 1. Prompt echo: the entire response appears verbatim inside the prompt. The
    #    observed failure was the prompt minus its first token, which lands here.
    #    The word floor keeps a legitimately terse answer ("yes") from tripping it.
    if len(words) >= 8 and t in p:
        defects.append("ECHOES_PROMPT")
    # 2. Degenerate repetition: a short chunk looped to fill the token budget. Normal
    #    English runs well above 0.15 distinct-word ratio at this length.
    if len(words) >= 30 and (len(set(words)) / len(words)) < 0.15:
        defects.append("DEGENERATE_REPETITION")
    if no_thinking and re.search(r"(?:</?think>|here(?:'s| is) a thinking process|analyze user input)", t):
        defects.append("THINKING_LEAK")
    return defects


def build_chat_body(model, prompt, max_tokens, temperature, no_thinking=False):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
        "cache_prompt": False,  # llama-server: never reuse prefill (DenseCore ignores; disabled via env too)
    }
    if no_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return body


def post_chat(url, model, prompt, max_tokens, temperature, timeout, no_thinking=False):
    body = build_chat_body(model, prompt, max_tokens, temperature, no_thinking)
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        raise ReqError(f"HTTP {e.code}: {e.read()[:300]!r}")
    except urllib.error.URLError as e:
        raise ReqError(f"URL error: {e}")
    wall = time.perf_counter() - t0
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        raise ReqError(f"non-JSON response: {raw[:300]!r}")
    choices = obj.get("choices") or []
    if not choices:
        raise ReqError(f"no choices: {raw[:300]!r}")
    text, channel = extract_text(choices[0])
    usage = obj.get("usage") or {}
    return {
        "wall": wall,
        "text": text,
        "channel": channel,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "timings": obj.get("timings"),  # llama-server native (bonus cross-check)
    }


def measure_prompt(url, model, prompt, args):
    # prefill probe (median over a couple of calls; prefill is the expensive part so keep small)
    pf_walls = []
    P = None
    for _ in range(args.prefill_repeats):
        r = post_chat(url, model, prompt, 1, 0.0, args.timeout, args.no_thinking)
        pf_walls.append(r["wall"])
        if r["prompt_tokens"]:
            P = r["prompt_tokens"]
    t_pf = statistics.median(pf_walls)

    # decode runs
    dec_tps, gens, native, output_hashes, output_channels = [], [], [], [], []
    for _ in range(args.repeats):
        r = post_chat(url, model, prompt, args.max_tokens, args.temperature, args.timeout, args.no_thinking)
        G = r["completion_tokens"]
        if G is None:
            raise ReqError("response missing usage.completion_tokens (cannot count tokens accurately)")
        if not r["text"].strip():
            raise ReqError(f"empty content (channel={r['channel']}, completion_tokens={G}) - "
                           f"server reported tokens but returned no text in content/reasoning_content")
        if G < args.min_tokens:
            raise ReqError(f"only {G} tokens (< min_tokens={args.min_tokens}); decode rate unreliable")
        defects = output_defects(prompt, r["text"], args.no_thinking)
        if defects:
            raise ReqError(f"output is not an answer ({', '.join(defects)}, channel={r['channel']}, "
                           f"completion_tokens={G}): {r['text'][:160]!r} - a tok/s number measured "
                           f"on broken output is not a result")
        dt = r["wall"] - t_pf
        # If the decode window collapses (or goes negative) the prompt was served from
        # cache: t_N has no prefill while t_pf does. Fail LOUDLY instead of emitting a
        # garbage rate (the old max(dt,1e-9) clamp). Disable prompt/prefix cache.
        if dt <= 0.0:
            raise ReqError(f"decode window {dt:.3f}s <= 0 -> prompt cache active? "
                           f"set DENSECORE_AGENT_PROMPT_CACHE=off + DENSECORE_DEBUG_DISABLE_PREFIX_CACHE_REUSE=1 "
                           f"(llama: cache_prompt=false)")
        dec_tps.append((G - 1) / dt)
        gens.append(G)
        output_hashes.append(hashlib.sha256(r["text"].encode("utf-8")).hexdigest())
        output_channels.append(r["channel"])
        # llama-server native rate (cross-check only; primary metric is wall-clock, same for both engines)
        tm = r.get("timings") or {}
        if isinstance(tm, dict) and tm.get("predicted_per_second"):
            native.append(float(tm["predicted_per_second"]))
    return {
        "prefill_ms": t_pf * 1000.0,
        "prompt_tokens": P,
        "prefill_tps": (P / t_pf) if P else None,
        "decode_tps": dec_tps,
        "gen_tokens": gens,
        "native_tps": native,
        "output_sha256": output_hashes,
        "output_channels": output_channels,
    }


def do_measure(args, prompts):
    url = args.url.rstrip("/") + "/v1/chat/completions"
    # warmup (discarded, OUTSIDE any numastat window the caller sets)
    for _ in range(args.warmup):
        post_chat(url, args.model, prompts[0], min(32, args.max_tokens), 0.0, args.timeout, args.no_thinking)
    all_dec, all_pf, all_gen, all_pftps, all_native = [], [], [], [], []
    all_output_hashes, all_output_channels = [], []
    for prompt in prompts:
        m = measure_prompt(url, args.model, prompt, args)
        all_dec += m["decode_tps"]
        all_gen += m["gen_tokens"]
        all_native += m["native_tps"]
        all_output_hashes += m["output_sha256"]
        all_output_channels += m["output_channels"]
        all_pf.append(m["prefill_ms"])
        if m["prefill_tps"]:
            all_pftps.append(m["prefill_tps"])
    out = {
        "label": args.label,
        "n_samples": len(all_dec),
        "prefill_ms_median": round(statistics.median(all_pf), 1),
        "prefill_tps_median": round(statistics.median(all_pftps), 2) if all_pftps else None,
        "decode_tps_median": round(statistics.median(all_dec), 3),
        "decode_tps_min": round(min(all_dec), 3),
        "decode_tps_max": round(max(all_dec), 3),
        "decode_tps_native_median": round(statistics.median(all_native), 3) if all_native else None,
        "gen_tokens_median": int(statistics.median(all_gen)),
        "quality_gate": "passed_nonempty_min_tokens_no_echo_no_repetition_no_thinking_leak",
        "output_sha256": all_output_hashes,
        "output_channels": all_output_channels,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out))


def do_capture(args, prompts):
    """Greedy deterministic capture for the parity gate. FAILS on empty/short output."""
    url = args.url.rstrip("/") + "/v1/chat/completions"
    os.makedirs(args.out, exist_ok=True)
    for i, prompt in enumerate(prompts):
        r = post_chat(url, args.model, prompt, args.max_tokens, 0.0, args.timeout, args.no_thinking)
        G = r["completion_tokens"]
        if not r["text"].strip():
            raise ReqError(f"parity capture prompt {i}: empty output (channel={r['channel']}, "
                           f"completion_tokens={G}) (gate would falsely pass)")
        if G is not None and G < args.min_tokens:
            raise ReqError(f"parity capture prompt {i}: only {G} tokens (< {args.min_tokens})")
        # Runs before the files are written, so Gate D can never diff two echoes and
        # call the match a pass.
        defects = output_defects(prompt, r["text"], args.no_thinking)
        if defects:
            raise ReqError(f"parity capture prompt {i}: output is not an answer "
                           f"({', '.join(defects)}, channel={r['channel']}): {r['text'][:160]!r}")
        with open(os.path.join(args.out, f"p{i:02d}.txt"), "w", encoding="utf-8") as f:
            f.write(r["text"])
    print(f"[ok] captured {len(prompts)} non-empty greedy outputs -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["measure", "capture"], default="measure")
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="local")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--label", default="run")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--min-tokens", type=int, default=64,
                    help="fail if generation shorter than this (guards empty/short outputs)")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--prefill-repeats", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--no-thinking", action="store_true",
                    help="send chat_template_kwargs.enable_thinking=false to both engines")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    prompts = read_prompts(args.prompts)
    if not prompts:
        print("[error] no prompts", file=sys.stderr); sys.exit(1)
    try:
        if args.mode == "measure":
            do_measure(args, prompts)
        else:
            do_capture(args, prompts)
    except ReqError as e:
        print(f"[FAIL] {args.label}/{args.mode}: {e}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
