#!/usr/bin/env python3
"""
Model Downloader for DenseCore Kubernetes Deployments.

This script handles downloading models from Hugging Face Hub with:
- Existence checking (skip if already present)
- Retry logic with exponential backoff
- Proper symlink creation for the main model path
- Graceful error handling with clear exit codes
"""

import hashlib
import os
import re
import shutil
import tempfile
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
    revision: str,
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
                revision=revision,
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


DEFAULT_REPO = "lmstudio-community/Qwen3.5-0.8B-GGUF"
DEFAULT_FILE = "Qwen3.5-0.8B-Q4_K_M.gguf"
DEFAULT_REVISION = "7925ccdc665d4efdb1034791e6b553e11128e6f8"
DEFAULT_SHA256 = "f5b14da98939b60bbe1019a964eba656407e1e0b64f1fe3003ff6d650e93bfec"


def verify_sha256(path: Path, expected: str) -> None:
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != expected:
        raise RuntimeError(f"SHA-256 mismatch for {path.name}: expected {expected}, got {actual}")


def create_symlink(source: Path, target: Path) -> None:
    """Publish a verified model with an atomic alias replacement."""
    if source == target:
        return
    with tempfile.TemporaryDirectory(prefix=".publish-", dir=target.parent) as staging:
        temporary = Path(staging) / "model-link"
        temporary.symlink_to(source.resolve())
        temporary.replace(target)
    log("info", f"Published verified model: {target} -> {source}")


def main() -> int:
    try:
        model_repo = get_env_or_default("MODEL_REPO", DEFAULT_REPO)
        model_file = get_env_or_default("MODEL_FILE", DEFAULT_FILE)
        is_default = (model_repo, model_file) == (DEFAULT_REPO, DEFAULT_FILE)
        revision = get_env_or_default("MODEL_REVISION", DEFAULT_REVISION if is_default else "")
        expected = get_env_or_default("MODEL_SHA256", DEFAULT_SHA256 if is_default else "").lower()
        if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise RuntimeError("MODEL_REVISION must be a full immutable 40-character commit hash")
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise RuntimeError("MODEL_SHA256 must be the expected 64-character SHA-256")
        filename = Path(model_file)
        if filename.is_absolute() or ".." in filename.parts or model_file in ("", "."):
            raise RuntimeError("MODEL_FILE must be a relative repository filename without '..'")
        hf_token = os.environ.get("HF_TOKEN", "").strip() or None
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        main_model_path = MODELS_DIR / MAIN_MODEL_NAME
        model_file_path = MODELS_DIR / ".artifacts" / expected / filename
        legacy_path = MODELS_DIR / filename
        if not model_file_path.resolve().is_relative_to(MODELS_DIR.resolve()):
            raise RuntimeError("Cached model path escapes the model directory")
        log("info", f"Configuration: repo={model_repo}, file={model_file}, revision={revision}, sha256={expected}")
        # An existing alias is not evidence that the requested revision is cached.
        # Hash the requested file on every init, including a shared PVC cache hit.
        if model_file_path.exists():
            verify_sha256(model_file_path, expected)
        else:
            with tempfile.TemporaryDirectory(prefix=".download-", dir=MODELS_DIR) as staging:
                if legacy_path.exists():
                    if not legacy_path.resolve().is_relative_to(MODELS_DIR.resolve()):
                        raise RuntimeError("Cached model path escapes the model directory")
                    # Preserve old paths used by other pods while migrating a legacy cache.
                    verify_sha256(legacy_path, expected)
                    downloaded_path = Path(staging) / "verified-legacy.gguf"
                    shutil.copyfile(legacy_path, downloaded_path)
                else:
                    downloaded_path = download_with_retry(
                        repo_id=model_repo, filename=model_file, local_dir=Path(staging),
                        token=hf_token, revision=revision,
                    )
                # A Hub client/cache may return a symlink into its staging cache.
                # Materialize it before the staging directory is removed.
                if downloaded_path.is_symlink():
                    regular = Path(staging) / "materialized-model.gguf"
                    shutil.copyfile(downloaded_path, regular)
                    downloaded_path = regular
                verify_sha256(downloaded_path, expected)
                model_file_path.parent.mkdir(parents=True, exist_ok=True)
                downloaded_path.replace(model_file_path)
        create_symlink(model_file_path, main_model_path)
        log("info", "Model checksum verified; model path is ready")
        return 0
    except Exception as error:
        log("error", f"Model not published: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
