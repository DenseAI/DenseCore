"""
Examples demonstrating LoRA adapter and RAM-aware smart loading.
"""

import densecore

## Example 1: RAM-Aware Smart Loading
## Automatically select optimal quantization based on available RAM


def example_smart_loading():
    """
    Smart loading automatically picks the best quantization for your hardware.
    """
    print("=" * 60)
    print("Example 1: RAM-Aware Smart Loading")
    print("=" * 60)

    # Simple usage - DenseCore detects your RAM and picks the best model
    model = densecore.smart_load("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", verbose=True)
    # Output:
    # [Smart Loader] Available RAM: 16.0 GB (using 12.8 GB with 80% safety margin)
    # [Smart Loader] Selected Q4_K_M (0.7 GB) - highest quality that fits in 16.0 GB RAM

    response = model.generate("Hello! How are you?", max_tokens=50)
    print(f"\nResponse: {response}\n")


## Example 2: Using from_pretrained with auto_select_quant


def example_from_pretrained_auto_quant():
    """
    Use from_pretrained with auto_select_quant for RAM-aware loading.
    """
    print("=" * 60)
    print("Example 2: from_pretrained with auto_select_quant")
    print("=" * 60)

    model = densecore.from_pretrained(
        "TheBloke/Llama-2-7B-GGUF",
        auto_select_quant=True,  # Enable RAM-aware selection
        verbose=True,
    )

    # If no suitable quantization fits, you'll get a helpful error:
    # MemoryError: Insufficient RAM for any quantization...


## Example 3: Manual Quantization Recommendation


def example_manual_recommendation():
    """
    Get quantization recommendation without loading.
    """
    print("=" * 60)
    print("Example 3: Manual Quantization Recommendation")
    print("=" * 60)

    filename, message = densecore.recommend_quantization("TheBloke/Llama-2-13B-GGUF", verbose=True)

    print(f"\nRecommended file: {filename}")
    print(f"Reason: {message}\n")


## Example 4: LoRA Adapter Loading


def example_lora_loading():
    """
    Load a model with a LoRA adapter.
    """
    print("=" * 60)
    print("Example 4: LoRA Adapter Loading")
    print("=" * 60)

    # Method 1: Load adapter during initialization
    model = densecore.DenseCore(
        model_path="./base_model.gguf",
        lora_adapter_path="./my-adapter.gguf",
        lora_scale=1.0,
        hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
    )

    # Method 2: Load adapter at runtime
    model = densecore.DenseCore(
        model_path="./base_model.gguf", hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct"
    )
    model.load_lora("./my-adapter.gguf", scale=0.8)

    # Check adapter status
    print(f"Has LoRA: {model.has_lora}")
    print(f"Active adapters: {model.list_lora_adapters()}")

    # Generate with adapter
    response = model.generate("Tell me about fine-tuning", max_tokens=100)
    print(f"Response with LoRA: {response}")

    # Disable adapter temporarily
    model.disable_lora()
    response_base = model.generate("Tell me about fine-tuning", max_tokens=100)
    print(f"Response without LoRA: {response_base}")

    # Re-enable
    model.enable_lora()


## Example 5: LoRA from HuggingFace Hub


def example_lora_from_hub():
    """
    Auto-detect and load LoRA adapters from HuggingFace Hub.
    """
    print("=" * 60)
    print("Example 5: LoRA from HuggingFace Hub")
    print("=" * 60)

    # If repo contains a LoRA adapter, it's auto-detected
    try:
        model = densecore.from_pretrained(
            "user/my-lora-adapter-repo",
            base_model_path="./base_model.gguf",  # Provide base model
        )
        # Output:
        # [LoRA] Detected LoRA adapter repository: user/my-lora-adapter-repo
        # [LoRA] Downloading adapter...

    except ValueError as e:
        print(f"Note: {e}")
        print("You need to provide base_model_path when loading LoRA adapters")


## Example 6: System Resource Information


def example_system_info():
    """
    Get information about your system resources.
    """
    print("=" * 60)
    print("Example 6: System Resource Information")
    print("=" * 60)

    resources = densecore.get_system_resources()
    print("\nYour system:")
    print(f"  Total RAM: {resources.total_ram_gb:.1f} GB")
    print(f"  Available RAM: {resources.available_ram_gb:.1f} GB")
    print(f"  CPU Cores: {resources.cpu_cores}")
    print(f"  CPU Threads: {resources.cpu_threads}")
    print()


## Example 7: Advanced - Combined Features


def example_combined_features():
    """
    Combine smart loading with LoRA adapters.
    """
    print("=" * 60)
    print("Example 7: Combined Smart Loading + LoRA")
    print("=" * 60)

    # Smart load the base model
    model = densecore.smart_load("TheBloke/Llama-2-7B-GGUF", verbose=True)

    # Add a LoRA adapter
    model.load_lora("./chat-adapter.gguf", scale=1.0, name="chat")

    # Generate with adapter
    response = model.generate("Hello! Can you help me?", max_tokens=200)
    print(f"\nResponse: {response}")


if __name__ == "__main__":
    # Note: These examples assume you have the necessary model files downloaded
    # Uncomment the examples you want to run

    # example_smart_loading()
    # example_from_pretrained_auto_quant()
    # example_manual_recommendation()
    # example_lora_loading()
    # example_lora_from_hub()
    example_system_info()
    # example_combined_features()
