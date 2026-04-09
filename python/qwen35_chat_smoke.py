import os
import sys
from pathlib import Path

os.environ.setdefault("DENSECORE_LOG_LEVEL", "error")
os.environ.setdefault("DENSECORE_QWEN35_ENABLE_THINKING", "0")

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from densecore import DenseCore

MODEL_PATH = REPO_ROOT / "Qwen3.5-2B-Q4_K_M.gguf"


def main() -> int:
    if not MODEL_PATH.exists():
        print(f"missing model: {MODEL_PATH}")
        return 1

    engine = DenseCore(model_path=str(MODEL_PATH), threads=8)
    try:
        out = engine.chat(
            [{"role": "user", "content": "The capital of France is"}],
            max_tokens=4,
            temperature=0.0,
            top_p=1.0,
        )
        print(repr(out))
        return 0 if out and out.strip() else 2
    finally:
        engine.close()


if __name__ == "__main__":
    raise SystemExit(main())
