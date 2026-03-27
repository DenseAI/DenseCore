import os
import sys
from pathlib import Path

os.environ.setdefault("DENSECORE_LOG_LEVEL", "error")

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from densecore import DenseCore

MODEL_PATH = REPO_ROOT / "Qwen3.5-2B-Q4_K_M.gguf"


def chat_prompt(user_text: str) -> str:
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def main() -> int:
    if not MODEL_PATH.exists():
        print(f"missing model: {MODEL_PATH}")
        return 1

    prompt = "The capital of France is"
    engine = DenseCore(model_path=str(MODEL_PATH), threads=8)
    try:
        greedy_1 = engine.generate(prompt, max_tokens=24, temperature=0.0, top_p=1.0)
        greedy_2 = engine.generate(prompt, max_tokens=24, temperature=0.0, top_p=1.0)
        plain_ko = engine.generate(
            "한국의 수도는 어디인가요? 한 문장으로 답하세요.",
            max_tokens=24,
            temperature=0.0,
            top_p=1.0,
        )
        os.environ["DENSECORE_QWEN35_ENABLE_THINKING"] = "0"
        no_think_chat = engine.generate(
            chat_prompt("한국의 수도는 어디인가요? 한 문장으로 답하세요."),
            max_tokens=32,
            temperature=0.0,
            top_p=1.0,
        )
    finally:
        engine.close()

    has_think_tag = any(tag in no_think_chat for tag in ("<think>", "</think>", "Thinking Process"))
    greedy_equal = greedy_1 == greedy_2

    print("GREEDY_1=", repr(greedy_1))
    print("GREEDY_2=", repr(greedy_2))
    print("GREEDY_EQUAL=", greedy_equal)
    print("PLAIN_KO=", repr(plain_ko))
    print("NO_THINK_CHAT=", repr(no_think_chat))
    print("HAS_THINK_TAG=", has_think_tag)
    sys.stdout.flush()

    if not greedy_equal:
        return 2
    if has_think_tag:
        return 3
    if not no_think_chat.strip():
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
