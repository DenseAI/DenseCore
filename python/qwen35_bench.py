import time
from pathlib import Path
import sys

from densecore import DenseCore

MODEL_PATH = "/mnt/c/Users/jwsong/CLionProjects/DenseCore/Qwen3.5-2B-Q4_K_M.gguf"


def main() -> int:
    if not Path(MODEL_PATH).exists():
        print(f"missing model: {MODEL_PATH}")
        return 1

    max_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nExplain what a database index is in 5 bullet points.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    engine = DenseCore(model_path=MODEL_PATH, threads=8)
    try:
        before = engine.get_metrics() or {}
        start = time.perf_counter()
        output = engine.generate(
            prompt, max_tokens=max_tokens, do_sample=False, temperature=0.0, top_p=1.0
        )
        elapsed = time.perf_counter() - start
        after = engine.get_metrics() or {}
        generated_tokens = int(after.get("total_tokens_generated", 0)) - int(
            before.get("total_tokens_generated", 0)
        )
        prompt_ids = (
            engine.tokenizer.encode(prompt, add_special_tokens=False) if engine.tokenizer else []
        )
        print(f"elapsed_s={elapsed:.4f}")
        print(f"prompt_tokens={len(prompt_ids)}")
        print(f"output_tokens={generated_tokens}")
        if elapsed > 0.0:
            print(f"decode_tps={generated_tokens / elapsed:.4f}")
        if after:
            print(f"engine_tps={float(after.get('tokens_per_second', 0.0)):.4f}")
        print(f"preview={output[:240]!r}")
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
