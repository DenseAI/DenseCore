import os
from pathlib import Path

os.environ.setdefault("DENSECORE_LOG_LEVEL", "error")

MODEL_PATH = "/mnt/c/Users/jwsong/CLionProjects/DenseCore/Qwen3.5-2B-Q4_K_M.gguf"


def apply_qwen_chat_template(prompt: str) -> str:
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def main() -> int:
    if not Path(MODEL_PATH).exists():
        print(f"missing model: {MODEL_PATH}")
        return 1

    from densecore import DenseCore

    prompt = apply_qwen_chat_template("한국의 수도는 어디인가요? 한 문장으로 답하세요.")
    engine = DenseCore(model_path=MODEL_PATH, threads=8)
    try:
        if engine.tokenizer is not None:
            prompt_arg = engine.tokenizer.encode(prompt, add_special_tokens=False)
        else:
            prompt_arg = prompt
        output = engine.generate(prompt_arg, max_tokens=8, temperature=0.0, top_p=1.0)
        print(f"len={len(output)}")
        print(repr(output))
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
