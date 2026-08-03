#!/usr/bin/env python3
"""Real multi-turn conversation e2e against an OpenAI-compatible server.

Not a token-count probe: it holds an actual requested-length conversation, feeds each
assistant reply back as history, and judges whether the replies are coherent
answers rather than echoes or fragments. Writes a JSON transcript so two runs
(e.g. NUMA off vs on) can be compared byte-for-byte.
"""
import argparse, json, re, sys, time, urllib.request

TURNS = [
    "Hi! I'm benchmarking a CPU inference server. In two or three sentences, what does 'memory bound' mean for LLM decoding?",
    "Got it. So if I double the memory bandwidth but keep the same CPU, roughly what happens to tokens per second?",
    "Now compare that to prefill. Why does prefill behave differently from decode on the same hardware?",
    "Summarize everything you told me as three short bullet points.",
]


def post(url, payload, timeout):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode()), time.time() - t0


def reply_text(choice):
    m = choice.get("message") or {}
    content = m.get("content") or ""
    reasoning = m.get("reasoning_content") or ""
    if content.strip():
        return content, "content"
    if reasoning.strip():
        return reasoning, "reasoning_only"
    return "", "empty"


def judge(user_msgs_so_far, text, tokens, no_thinking=False):
    """Cheap, mechanical coherence checks. No model-as-judge needed to catch
    the failure modes we actually saw (empty, 2-token fragment, prompt echo).

    The echo check must compare against EVERY user message seen so far, not
    just the current turn. DenseCore's observed failure was replaying turn 1's
    text on turns 2 and 3; checking only the current turn scored those as OK
    and reported 2/4 when the true value was 0/4.
    """
    problems = []
    if not text:
        problems.append("EMPTY")
    if tokens is not None and tokens < 15:
        problems.append(f"TOO_SHORT({tokens} tok)")
    norm_t = " ".join(text.lower().split())
    if no_thinking and re.search(r"(?:</?think>|here(?:'s| is) a thinking process|analyze user input)", norm_t):
        problems.append("THINKING_LEAK")
    for idx, u in enumerate(user_msgs_so_far, 1):
        norm_u = " ".join(u.lower().split())
        if norm_t and (norm_t in norm_u or norm_u in norm_t):
            problems.append(f"ECHOES_PROMPT(turn{idx})")
            break
    if text and len(text.split()) < 8:
        problems.append("FRAGMENT")
    words = norm_t.split()
    if len(words) >= 30 and (len(set(words)) / len(words)) < 0.15:
        problems.append("DEGENERATE_REPETITION")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", default="local")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--turns", type=int, choices=range(1, len(TURNS) + 1), default=len(TURNS),
                    help="number of conversation turns to run")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--no-thinking", action="store_true",
                    help="send chat_template_kwargs.enable_thinking=false")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    history, records, seen_user_msgs = [], [], []
    total_gen_tokens, total_gen_time = 0, 0.0

    print(f"\n{'='*70}\n### CONVERSATION e2e :: {args.label}\n{'='*70}")
    selected_turns = TURNS[:args.turns]
    for i, user_msg in enumerate(selected_turns, 1):
        wire_user_msg = f"{user_msg}\n/no_think" if args.no_thinking else user_msg
        history.append({"role": "user", "content": wire_user_msg})
        seen_user_msgs.append(user_msg)
        payload = {"model": args.model, "messages": list(history),
                   "max_tokens": args.max_tokens, "temperature": 0.0, "stream": False}
        if args.no_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            obj, dt = post(args.url, payload, args.timeout)
        except Exception as e:
            print(f"\n--- turn {i} FAILED: {e}")
            records.append({"turn": i, "user": user_msg, "error": str(e)})
            break

        ch = (obj.get("choices") or [{}])[0]
        text, channel = reply_text(ch)
        usage = obj.get("usage") or {}
        gen = usage.get("completion_tokens")
        problems = judge(seen_user_msgs, text, gen, no_thinking=args.no_thinking)
        if gen:
            total_gen_tokens += gen
            total_gen_time += dt

        print(f"\n--- turn {i} ---")
        print(f"USER : {user_msg}")
        print(f"ASSISTANT [{channel}, {gen} tok, {dt:.1f}s, finish={ch.get('finish_reason')}]:")
        print(f"  {text[:700]}" if text else "  <EMPTY>")
        print(f"VERDICT: {'OK' if not problems else 'BROKEN -> ' + ', '.join(problems)}")

        records.append({"turn": i, "user": user_msg, "assistant": text, "channel": channel,
                        "completion_tokens": gen, "prompt_tokens": usage.get("prompt_tokens"),
                        "finish_reason": ch.get("finish_reason"), "seconds": round(dt, 3),
                        "problems": problems, "timings": obj.get("timings")})
        # Feed the real reply back so later turns depend on earlier ones.
        history.append({"role": "assistant", "content": text})

    ok_turns = sum(1 for r in records if not r.get("problems") and not r.get("error"))
    rate = (total_gen_tokens / total_gen_time) if total_gen_time > 0 else 0.0
    print(f"\n{'-'*70}")
    print(f"SUMMARY {args.label}: {ok_turns}/{len(selected_turns)} turns coherent | "
          f"{total_gen_tokens} tokens generated | {rate:.2f} tok/s aggregate")
    print(f"{'-'*70}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"label": args.label, "turns": records, "ok_turns": ok_turns,
                       "total_gen_tokens": total_gen_tokens,
                       "aggregate_tok_per_s": round(rate, 3)}, f, indent=2)
    return 0 if ok_turns == len(selected_turns) else 1


if __name__ == "__main__":
    sys.exit(main())
