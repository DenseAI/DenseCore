import argparse
import time

import densecore


def run_benchmark(model_id, duration_seconds=10):
    print(f"Benchmarking model: {model_id}")
    # Use native API
    try:
        model_path = densecore.download_model(model_id)
        # Derive tokenizer from model_id (strip -GGUF suffix if present)
        tokenizer_id = model_id.replace("-GGUF", "").replace("-Instruct-GGUF", "-Instruct")
        print(f"Loading model from {model_path} with tokenizer from {tokenizer_id}")
        model = densecore.DenseCore(model_path, hf_repo_id=tokenizer_id, threads=16)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Warming up...")
    try:
        result1 = model.generate("Hello", max_tokens=24)  # Warmup
        print(result1)
    except Exception as e:
        print(f"Warmup warning: {e}")

    print("Starting benchmark...")
    prompt = "Write a long essay about the history of computing."
    start_time = time.time()
    token_count = 0

    try:
        # Use generate for reliable token counting
        result = model.generate(prompt, max_tokens=32)
        print(result)
        # Rough token count based on output (space-separated)
        token_count = len(result.split())
    except Exception as e:
        print(f"Error during generation: {e}")

    elapsed = time.time() - start_time
    # Avoid div by zero
    if elapsed > 0:
        tps = token_count / elapsed
    else:
        tps = 0
    print(f"\nGenerated {token_count} tokens in {elapsed:.2f} seconds")
    print(f"TPS: {tps:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-4B-GGUF", help="Model to benchmark"
    )
    args = parser.parse_args()

    run_benchmark(args.model)
