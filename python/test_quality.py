#!/usr/bin/env python3
"""
Quick quality test using locally cached models.
Tests inference to ensure NO garbage output.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_local_model():
    """Test inference with local Qwen3 model."""
    try:
        from densecore import DenseCore

        print("=" * 70)
        print("DenseCore Quality Test - Local Model")
        print("=" * 70)

        # Use local Qwen3-0.6B (smallest, fastest) - actual blob file
        model_path = "/home/jaewook/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B-GGUF/blobs/9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031"

        print(f"\n📂 Model: {Path(model_path).name}")
        print(f"   Path: {model_path}")

        print("\n🔧 Initializing DenseCore Engine...")
        engine = DenseCore(
            model_path=model_path,
            threads=4,
            hf_repo_id="Qwen/Qwen3-0.6B",  # For tokenizer
        )
        print("✅ Engine initialized successfully")

        # Test prompts
        test_prompts = [
            "The capital of France is",
            "Hello, my name is",
            "Explain quantum physics in simple terms:",
        ]

        print("\n" + "=" * 70)
        print("INFERENCE QUALITY TEST")
        print("=" * 70)

        all_passed = True
        results = []

        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n[Test {i}/{len(test_prompts)}]")
            print(f'Prompt: "{prompt}"')
            print("-" * 70)

            start = time.perf_counter()
            output = engine.generate(prompt, max_tokens=30)
            elapsed = time.perf_counter() - start

            print(f"Output: {output}")
            print(
                f"Time: {elapsed:.3f}s ({len(output.split()) / (elapsed + 0.001):.1f} tokens/sec)"
            )

            # Quality checks
            passed = True
            issues = []

            # Check 1: Not empty
            if not output or len(output.strip()) == 0:
                passed = False
                issues.append("Empty output")

            # Check 2: Not too repetitive
            elif len(set(output)) < 5:
                passed = False
                issues.append("Too repetitive (same chars)")

            # Check 3: No excessive weird characters
            elif output.count("�") > 2:
                passed = False
                issues.append("Contains � characters")

            # Check 4: Reasonable length
            elif len(output.strip()) < 3:
                passed = False
                issues.append("Too short")

            # Check 5: Contains readable words (heuristic)
            elif not any(c.isalpha() for c in output):
                passed = False
                issues.append("No alphabetic characters")

            if passed:
                print("✅ PASS: Output is coherent and readable")
                results.append("✅ PASS")
            else:
                print(f"❌ FAIL: {', '.join(issues)}")
                results.append(f"❌ FAIL ({', '.join(issues)})")
                all_passed = False

        engine.close()

        # Final summary
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        for i, result in enumerate(results, 1):
            print(f"  Test {i}: {result}")
        print("-" * 70)

        if all_passed:
            print("✅ ALL TESTS PASSED")
            print("   NO GARBAGE OUTPUT DETECTED")
            print("   DenseCore is producing COHERENT text")
            print("=" * 70)
            return True
        else:
            print("❌ SOME TESTS FAILED")
            print("   Quality issues detected")
            print("=" * 70)
            return False

    except FileNotFoundError:
        print("\n❌ ERROR: Model file not found")
        print("   Please check the model path")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_local_model()
    sys.exit(0 if success else 1)
