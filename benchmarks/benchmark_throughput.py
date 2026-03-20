import argparse
import time
import sys
import os
import platform
import random
import glob
from pathlib import Path
from typing import Optional, List, Any

# Mock classes for standalone testing
class MockDenseCore:
    def __init__(self, model_path: str):
        pass
    def generate(self, prompt: str, max_tokens: int = 128) -> str:
        return " Mock response" * (max_tokens // 2)
    def stream(self, prompt: str, max_tokens: int = 128):
        time.sleep(0.1) # TTFT
        yield "Start"
        for i in range(max_tokens):
            time.sleep(0.01) # TPS
            yield f" token_{i}"

try:
    import densecore
except ImportError:
    densecore = None

try:
    from densecore.integrations.langchain import DenseCoreChatModel
    from langchain_core.tools import tool
    from langchain_core.messages import HumanMessage
except ImportError as e:
    print(f"  [Info] LangChain not found: {e}")
    DenseCoreChatModel = None # type: ignore
    tool = lambda x: x # dummy decorator
    HumanMessage = None

def scan_cached_models(cache_dir: Optional[str] = None) -> List[tuple[str, str]]:
    """
    Scans HuggingFace cache directory for available models.
    Returns list of (display_name, full_path) tuples.
    """
    if not cache_dir:
        home = Path.home()
        cache_dir = home / ".cache" / "huggingface" / "hub"

    base_path = Path(cache_dir)
    if not base_path.exists():
        print(f"Cache directory not found: {base_path}")
        return []

    models = []
    print(f"Scanning {base_path}...")

    # Look for directories starting with models--
    for model_dir in base_path.glob("models--*"):
        try:
            # Parse org and model name from dir name
            # Format: models--{org}--{model}
            parts = model_dir.name.split("--")
            if len(parts) >= 3:
                org = parts[1]
                model_name = "--".join(parts[2:])
                display_name = f"{org}/{model_name}"

                # Find the most recent snapshot
                snapshots_dir = model_dir / "snapshots"
                if not snapshots_dir.exists():
                    continue

                # Get the most recent snapshot (usually there's only one or we pick the last one)
                snapshots = sorted(list(snapshots_dir.iterdir()), key=os.path.getmtime, reverse=True)
                if not snapshots:
                    continue

                latest_snapshot = snapshots[0]

                # Check for GGUF files in the snapshot
                gguf_files = list(latest_snapshot.glob("*.gguf"))
                if gguf_files:
                    # Prefer q4_k_m or the largest file if multiple exist
                    # For simplicity, just pick the first one, or let user know
                    target_file = gguf_files[0]
                    models.append((display_name, str(target_file)))
        except Exception as e:
            continue

    return sorted(models, key=lambda x: x[0])

def get_simd_level():
    # Attempt to detect SIMD level from CPU info or engine
    # In a real scenario, this might come from densecore.get_device_info()
    try:
        if densecore and hasattr(densecore, 'get_device_info'):
             return densecore.get_device_info().get("simd", "Unknown")
        elif densecore:
             # Fallback
             return "AVX-512" if "avx512" in platform.processor().lower() else "AVX2"
    except:
        pass
    return "Unknown/Mock"

def infer_repo_id_from_path(model_path: str) -> Optional[str]:
    """
    Infer HuggingFace repo ID from model cache path.

    HuggingFace cache structure:
    ~/.cache/huggingface/hub/models--{org}--{model}/snapshots/{revision}/{file}.gguf

    Example:
    /home/user/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct-GGUF/snapshots/.../file.gguf
    -> Qwen/Qwen2.5-0.5B-Instruct (strips -GGUF suffix for tokenizer compatibility)
    """
    import re

    # Pattern: models--{org}--{model}
    match = re.search(r'models--([^/]+)--([^/]+)', model_path)
    if match:
        org, model = match.groups()
        # Strip -GGUF suffix for tokenizer repo (tokenizer is in non-GGUF repo)
        tokenizer_model = re.sub(r'-GGUF$', '', model)
        return f"{org}/{tokenizer_model}"
    return None

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    try:
        return str(eval(expression))
    except:
        return "Error"

class DummyTokenizer:
    def encode(self, text, **kwargs): return [1, 2, 3]
    def decode(self, tokens, **kwargs): return "x" * len(tokens)

def benchmark_raw(model_path: str, repo_id: Optional[str] = None, n_tokens: int = 128, threads: int = 0, mock: bool = False, args: Any = None):
    print(f"Running Scenario A: Raw Engine Throughput...")
    if mock:
        engine = MockDenseCore(model_path)
    else:
        # Auto-infer repo_id from model path if not provided (unless dummy requested)
        effective_repo_id = repo_id
        use_dummy = args.dummy_tokenizer if args else False

        if not effective_repo_id and not use_dummy:
             effective_repo_id = infer_repo_id_from_path(model_path)
             if effective_repo_id:
                 print(f"  [Info] Auto-detected tokenizer repo: {effective_repo_id}")

        engine = densecore.DenseCore(model_path=model_path, hf_repo_id=effective_repo_id, threads=threads, verbose=True)
        if not hasattr(engine, 'tokenizer') or engine.tokenizer is None:
             print("  [Warn] Tokenizer not found/initialized. Using DummyTokenizer (results may be unreliable).")
             engine.tokenizer = DummyTokenizer()

    # TTFT and TPS measurement
    start_time = time.time()
    first_token_time = None
    count = 0

    # Real prompt for quality check
    prompt_text = "The capital of France is"

    # Manually tokenize to avoid potential issues in _submit_request/string handling
    if hasattr(engine, 'tokenizer') and engine.tokenizer:
        try:
            prompt_ids = engine.tokenizer.encode(prompt_text, add_special_tokens=True)
        except Exception as e:
            print(f"  [Warn] Tokenizer failed: {e}. Using dummy IDs.")
            prompt_ids = [1, 2, 3, 4, 5]
    else:
        prompt_ids = [1, 2, 3, 4, 5]

    try:
        # Warmup - use text prompts instead of raw token IDs (workaround for stream([list]) hang)
        print("  Warming up...", end="\r")
        for _ in engine.stream("Warmup", max_tokens=5): pass

        print(f"  Generating (Prompt: '{prompt_text}')... ")
        gen_start = time.time()

        # Stream and print output - use text prompt directly
        full_text = ""
        for i, chunk in enumerate(engine.stream(prompt_text, max_tokens=n_tokens)):
            if i == 0:
                first_token_time = time.time()

            print(chunk, end="", flush=True)
            full_text += chunk
            count += 1

        print("\n") # Newline after generation
        end_time = time.time()

        if first_token_time:
            ttft = (first_token_time - gen_start) * 1000
            gen_time = end_time - first_token_time
            tps = (count - 1) / gen_time if count > 1 else 0
        else:
            ttft = 0
            tps = 0

        return ttft, tps
    except Exception as e:
        print(f"  Error: {e}")
        return 0.0, 0.0

def benchmark_langchain(model_path: str, repo_id: Optional[str] = None, threads: int = 16, mock: bool = False):
    print(f"Running Scenario B: LangChain Agent Latency...")
    if mock:
        return 1234.5 # Mock latency ms

    if not DenseCoreChatModel:
        print("LangChain not installed, skipping.")
        return 0.0

    chat = DenseCoreChatModel(model_path=model_path, hf_repo_id=repo_id, temperature=0, threads=threads) # Explicit threads
    chat_with_tools = chat.bind_tools([calculator])

    start_time = time.time()
    try:
        # Simple math query that forces tool use
        # Force a prompt that triggers tool use
        response = chat_with_tools.invoke([HumanMessage(content="Calculate 25 * 4")])
        # In a real agent loop, we'd execute the tool, but here we measure Model Latency for tool call generation
    except Exception as e:
        print(f"LangChain Error: {e}")

    end_time = time.time()
    return (end_time - start_time) * 1000

def main():
    parser = argparse.ArgumentParser(description="DenseCore Throughput & Latency Benchmark")
    parser.add_argument("--model", type=str, default=None, help="Path to GGUF model")
    parser.add_argument("--repo_id", type=str, default=None, help="HuggingFace Repo ID for tokenizer")
    parser.add_argument("--threads", type=int, default=8, help="Number of threads (0=auto)")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode")
    parser.add_argument("--n-predict", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--scan-cache", "-L", action="store_true", help="Scan local cache for models")
    parser.add_argument("--dummy-tokenizer", action="store_true", help="Use dummy tokenizer (bypass HuggingFace/Torch)")
    args = parser.parse_args()

    model_path = args.model
    repo_id = args.repo_id

    if args.dummy_tokenizer:
        print("  [Info] User requested dummy tokenizer. Bypassing HuggingFace tokenizer initialization.")
        repo_id = None # Force None to skip loading in DenseCore constructor

    if args.scan_cache:
        models = scan_cached_models()
        if not models:
            print("No GGUF models found in local cache.")
            sys.exit(1)

        print("\nAvailable cached models:")
        for idx, (name, path) in enumerate(models):
            print(f" [{idx}] {name}")

        try:
            selection = input("\nSelect model index: ")
            idx = int(selection)
            if 0 <= idx < len(models):
                print(f"Selected: {models[idx][0]}")
                model_path = models[idx][1]
                # Try to infer repo_id from the display name if possible,
                # though benchmark_raw also does inference from path.
                # display_name matches org/model format which is usually sufficient.
                if not repo_id:
                   # Strip -GGUF from display name to often get the tokenizer repo
                   base_name = models[idx][0].replace("-GGUF", "").replace("-gguf", "")
                   repo_id = base_name
                   print(f"Using inferred tokenizer repo: {repo_id}")
            else:
                print("Invalid index.")
                sys.exit(1)
        except ValueError:
            print("Invalid input.")
            sys.exit(1)

    if not model_path and not args.mock:
         print("Error: --model argument is required unless using --scan-cache or --mock")
         sys.exit(1)

    # Header
    print(f"{'='*60}")
    print(f"DenseCore Performance Benchmark (SIMD: {get_simd_level()})")
    print(f"{'='*60}")

    # Run Scenario A
    ttft, tps = benchmark_raw(model_path, repo_id=repo_id, n_tokens=args.n_predict, threads=args.threads, mock=args.mock or (densecore is None), args=args)

    # Run Scenario B
    agent_latency = benchmark_langchain(model_path, repo_id=repo_id, threads=args.threads if args.threads > 0 else 16, mock=args.mock or (densecore is None))

    # Output Table
    print(f"\n{'-'*60}")
    print(f"{'Metric':<25} | {'Value':<15} | {'Unit':<10}")
    print(f"{'-'*60}")
    print(f"{'Time To First Token':<25} | {ttft:>10.2f}      | ms")
    print(f"{'Generation Throughput':<25} | {tps:>10.2f}      | tok/s")
    print(f"{'Agent Response Latency':<25} | {agent_latency:>10.2f}      | ms")
    print(f"{'-'*60}")

if __name__ == "__main__":
    main()
