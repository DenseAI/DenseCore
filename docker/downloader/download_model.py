#!/usr/bin/env python3
"""
Model Downloader for DenseCore Kubernetes Deployments.

This script handles downloading models from Hugging Face Hub with:
- Existence checking (skip if already present)
- Retry logic with exponential backoff
- Proper symlink creation for the main model path
- Graceful error handling with clear exit codes
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional

# Constants
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 2.0
MAX_BACKOFF_SECONDS = 60.0
MODELS_DIR = Path("/models")
MAIN_MODEL_NAME = "main_model.gguf"


def log(level: str, message: str) -> None:
    """Structured logging to stdout."""
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[{timestamp}] [{level.upper()}] {message}", flush=True)


def get_env_or_default(key: str, default: str) -> str:
    """Get environment variable with fallback default."""
    value = os.environ.get(key, "").strip()
    return value if value else default


def download_with_retry(
    repo_id: str,
    filename: str,
    local_dir: Path,
    token: Optional[str],
) -> Path:
    """
    Download a model file with exponential backoff retry.

    Args:
        repo_id: Hugging Face repository ID (e.g., "Qwen/Qwen2.5-0.5B-Instruct-GGUF")
        filename: Name of the file to download
        local_dir: Directory to save the file
        token: Optional Hugging Face API token

    Returns:
        Path to the downloaded file

    Raises:
        RuntimeError: If all retries are exhausted
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import (
        HfHubHTTPError,
        EntryNotFoundError,
        RepositoryNotFoundError,
    )

    backoff = INITIAL_BACKOFF_SECONDS
    last_exception: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log(
                "info",
                f"Attempt {attempt}/{MAX_RETRIES}: Downloading {repo_id}/{filename}",
            )

            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(local_dir),
                token=token,
            )

            log("info", f"Download successful: {path}")
            return Path(path)

        except (EntryNotFoundError, RepositoryNotFoundError) as e:
            # Non-retryable: file or repo doesn't exist
            log("error", f"Resource not found (non-retryable): {e}")
            raise RuntimeError(f"Resource not found: {e}") from e

        except HfHubHTTPError as e:
            last_exception = e
            if attempt < MAX_RETRIES:
                log(
                    "warn",
                    f"HTTP error (attempt {attempt}): {e}. Retrying in {backoff:.1f}s...",
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)
            else:
                log("error", f"HTTP error after {MAX_RETRIES} attempts: {e}")

        except Exception as e:
            last_exception = e
            if attempt < MAX_RETRIES:
                log(
                    "warn",
                    f"Unexpected error (attempt {attempt}): {e}. Retrying in {backoff:.1f}s...",
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)
            else:
                log("error", f"Unexpected error after {MAX_RETRIES} attempts: {e}")

    raise RuntimeError(
        f"Download failed after {MAX_RETRIES} attempts"
    ) from last_exception


def create_symlink(source: Path, target: Path) -> None:
    """Create a symlink, removing existing if necessary."""
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source)
    log("info", f"Created symlink: {target} -> {source}")


def main() -> int:
    """
    Main entry point.

    Returns:
        0 on success, non-zero on failure
    """
    # Read configuration from environment
    model_repo = get_env_or_default("MODEL_REPO", "Qwen/Qwen2.5-0.5B-Instruct-GGUF")
    model_file = get_env_or_default("MODEL_FILE", "qwen2.5-0.5b-instruct-q4_k_m.gguf")
    hf_token = os.environ.get("HF_TOKEN", "").strip() or None

    log("info", f"Configuration: repo={model_repo}, file={model_file}")

    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Check if model already exists (via symlink or direct file)
    main_model_path = MODELS_DIR / MAIN_MODEL_NAME
    model_file_path = MODELS_DIR / model_file

    if main_model_path.exists():
        log("info", f"Model already exists at {main_model_path}. Skipping download.")
        return 0

    if model_file_path.exists():
        log("info", f"Model file exists at {model_file_path}. Creating symlink.")
        create_symlink(model_file_path, main_model_path)
        return 0

    # Download the model
    try:
        downloaded_path = download_with_retry(
            repo_id=model_repo,
            filename=model_file,
            local_dir=MODELS_DIR,
            token=hf_token,
        )

        # Create symlink to main_model.gguf if different
        if downloaded_path.name != MAIN_MODEL_NAME:
            create_symlink(downloaded_path, main_model_path)

        log("info", "Model download complete!")
        return 0

    except RuntimeError as e:
        log("error", f"Fatal error: {e}")
        return 1
    except Exception as e:
        log("error", f"Unexpected fatal error: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
