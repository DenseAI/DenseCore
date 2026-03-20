import os
import sys
import time
from pathlib import Path

# Suppress TRACE/DEBUG logs from C++ engine
os.environ.setdefault("DENSECORE_LOG_LEVEL", "error")

MODEL_PATH = "/mnt/c/Users/jwsong/PycharmProjects/DenseCore/models/Qwen3-4B-Q4_K_M.gguf"
MODEL_PATH_qwen35 = "/mnt/c/Users/jwsong/CLionProjects/DenseCore/Qwen3.5-2B-Q4_K_M.gguf"


def apply_qwen3_chat_template(prompt: str) -> str:
    """Wrap prompt in Qwen3 instruction format (no-thinking mode)."""
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def test_qwen35_2b():
    print("=" * 70)
    print("DenseCore E2E Test - Qwen3-4B")
    print("=" * 70)

    if not Path(MODEL_PATH).exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return False

    try:
        from densecore import DenseCore
    except ImportError:
        print("ERROR: Failed to import densecore.")
        return False

    print("\nInitializing DenseCore Engine...")
    start_init = time.perf_counter()
    engine = DenseCore(model_path=MODEL_PATH_qwen35, threads=8)
    init_time = time.perf_counter() - start_init
    print(f"Engine initialized in {init_time:.2f}s")

    test_prompts = [
        "한국의 수도는 어디인가요?",
        "안녕! 너는 누구니? 간단하게 자기소개 부탁해.",
        "Write a python function to compute the Fibonacci sequence efficiently.",
        "DenseCore라는 AI 추론 엔진이 있어. 이 엔진이 왜 중요할까? 3가지 이유를 들어서 설명해줘.",
    ]

    print("\n" + "=" * 70)
    print("INFERENCE QUALITY & SPEED TEST")
    print("=" * 70)

    all_passed = True

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[Test {i}/{len(test_prompts)}]")
        print(f'Prompt: "{prompt}"')
        print("-" * 70)

        formatted = apply_qwen3_chat_template(prompt)

        start = time.perf_counter()
        output = engine.generate(formatted, max_tokens=256)
        elapsed = time.perf_counter() - start

        print(f"Output:\n{output}")

        # Rough tokens/sec: Qwen3 averages ~2 chars/token for Korean, ~4 for English
        est_tokens = max(1, len(output) / 3.0)
        tps = est_tokens / (elapsed + 1e-6)
        print(f"\nTime: {elapsed:.3f}s  (~{tps:.1f} tok/sec estimated)")

        # Quality checks
        passed = True
        issues = []
        if not output or len(output.strip()) == 0:
            passed = False
            issues.append("Empty output")
        elif len(set(output)) < 5:
            passed = False
            issues.append("Too repetitive")
        elif "\x00" in output:
            passed = False
            issues.append("Contains null bytes")

        if passed:
            print("PASS")
        else:
            print(f"FAIL: {', '.join(issues)}")
            all_passed = False

    engine.close()
    return all_passed


if __name__ == "__main__":
    ok = test_qwen35_2b()
    sys.exit(0 if ok else 1)
