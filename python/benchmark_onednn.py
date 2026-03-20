import argparse
import os
import statistics
import subprocess
import sys
import time


def run_child(args, mode):
    env = os.environ.copy()
    env["DENSECORE_ONEDNN"] = "1" if mode in ("auto", "onednn") else "0"
    if mode == "onednn":
        env["FORCE_ONEDNN"] = "1"
    elif "FORCE_ONEDNN" in env:
        env.pop("FORCE_ONEDNN")

    cmd = [sys.executable, __file__, "--mode", mode]
    if args.model_path:
        cmd += ["--model-path", args.model_path]
    if args.hf_repo:
        cmd += ["--hf-repo", args.hf_repo]
    cmd += ["--max-tokens", str(args.max_tokens)]
    cmd += ["--prompt-tokens", str(args.prompt_tokens)]
    cmd += ["--concurrency", str(args.concurrency)]
    cmd += ["--threads", str(args.threads)]
    subprocess.run(cmd, env=env, check=True)


def generate_prompt_tokens(n_tokens):
    # approximate token count by repeating a short pattern
    token = " hello"
    return token * n_tokens


def run_benchmark(args):
    import densecore

    prompt = generate_prompt_tokens(args.prompt_tokens)
    model = densecore.DenseCore(
        args.model_path or args.hf_repo,
        hf_repo_id=args.hf_repo,
        threads=args.threads,
        verbose=True,
    )

    prefill_latencies = []
    decode_latencies = []
    total_tokens = 0
    total_time = 0.0

    def run_request():
        nonlocal total_tokens, total_time
        t0 = time.perf_counter()
        first_token_time = None
        prev_time = None
        tokens = 0

        for token in model.stream(prompt, max_tokens=args.max_tokens):
            now = time.perf_counter()
            if first_token_time is None:
                first_token_time = now
                prefill_latencies.append(first_token_time - t0)
                prev_time = now
            else:
                decode_latencies.append(now - prev_time)
                prev_time = now
            tokens += 1

        total_tokens += tokens
        total_time += time.perf_counter() - t0

    for _ in range(args.concurrency):
        run_request()

    tps = total_tokens / total_time if total_time > 0 else 0.0
    p50_prefill = statistics.median(prefill_latencies) if prefill_latencies else 0.0
    p95_prefill = (
        statistics.quantiles(prefill_latencies, n=20)[-1]
        if len(prefill_latencies) >= 20
        else p50_prefill
    )
    p50_decode = statistics.median(decode_latencies) if decode_latencies else 0.0
    p95_decode = (
        statistics.quantiles(decode_latencies, n=20)[-1]
        if len(decode_latencies) >= 20
        else p50_decode
    )

    print("RESULTS")
    print(f"tokens_total={total_tokens}")
    print(f"tokens_sec={tps:.2f}")
    print(f"prefill_p50_ms={p50_prefill * 1000:.2f}")
    print(f"prefill_p95_ms={p95_prefill * 1000:.2f}")
    print(f"decode_p50_ms={p50_decode * 1000:.2f}")
    print(f"decode_p95_ms={p95_decode * 1000:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "auto", "onednn"], help="benchmark mode")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--hf-repo", type=str, default="")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()

    if args.mode is None:
        for mode in ("baseline", "auto", "onednn"):
            run_child(args, mode)
        return

    print(f"MODE={args.mode}")
    run_benchmark(args)


if __name__ == "__main__":
    main()
