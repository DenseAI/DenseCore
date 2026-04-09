import time
import os
from pathlib import Path
from densecore import DenseCore

MODEL_PATH = "Qwen3.5-2B-Q4_K_M.gguf"


def main():
    if not Path(MODEL_PATH).exists():
        print(f"Missing model: {MODEL_PATH}")
        return

    print(f"--- Benchmarking {MODEL_PATH} ---")
    # Initialize engine
    engine = DenseCore(model_path=MODEL_PATH, threads=8)

    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n한국의 인공지능 기술 발전에 대해 짧게 설명해줘.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    try:
        print("Warmup...")
        engine.generate(prompt, max_tokens=10)

        print("Running Quality & Speed Benchmark...")
        start = time.perf_counter()
        # Quality check: set temperature=0 for deterministic output
        output = engine.generate(prompt, max_tokens=128, temperature=0.0)
        elapsed = time.perf_counter() - start

        # Get metrics if available
        metrics = engine.get_metrics() or {}
        # In some versions metrics might be total, let's calculate from output if needed
        # But typically engine tracking is more accurate

        print("\n[Output Preview]")
        print(output)
        print("-" * 40)

        print(f"Elapsed Time: {elapsed:.2f}s")
        # If engine doesn't provide token count directly, we'll estimate or use its internal count
        # Let's check what metrics we have
        print(f"Raw Metrics: {metrics}")

        if "total_tokens_generated" in metrics:
            gen_tokens = int(metrics["total_tokens_generated"])
            print(f"Generated Tokens: {gen_tokens}")
            print(f"TPS (Tokens Per Second): {gen_tokens / elapsed:.2f}")
        else:
            # Estimate by words as fallback
            tokens_est = len(output.split()) * 1.3  # rough factor for Korean
            print(f"Estimated Tokens: {tokens_est:.1f}")
            print(f"Estimated TPS: {tokens_est / elapsed:.2f}")

    finally:
        engine.close()


if __name__ == "__main__":
    main()
