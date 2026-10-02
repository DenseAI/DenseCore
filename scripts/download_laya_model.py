#!/usr/bin/env python3
"""Download the supported, self-contained Laya GGUF at an immutable revision."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request


REPOSITORY = "zerodegress/laya-gguf"
REVISION = "ff850ac9c08707e926847711a4d1819dc228ecc7"
FILENAME = "laya-f16.gguf"
SIZE = 848158048
SHA256 = "62db2affc0fe4b9f2538ba61299e5fd8bf876f50f765089a343cd4113eef21a4"
URL = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{FILENAME}"


def verify(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size != SIZE:
        return False
    with path.open("rb") as source:
        digest = hashlib.sha256()
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
        return digest.hexdigest() == SHA256


def download(destination: Path) -> None:
    if verify(destination):
        return
    if destination.exists():
        raise ValueError(f"Refusing to replace an existing file with a different checksum: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as sink:
            temporary = Path(sink.name)
            with urllib.request.urlopen(URL, timeout=60) as source:
                while chunk := source.read(1024 * 1024):
                    sink.write(chunk)
        if not verify(temporary):
            raise ValueError("Downloaded Laya GGUF failed size/SHA-256 validation")
        # A competing download must not replace an unrelated model.
        if destination.exists():
            if not verify(destination):
                raise ValueError(f"Destination appeared with a different checksum: {destination}")
        else:
            destination.hardlink_to(temporary)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/laya") / FILENAME)
    args = parser.parse_args()
    download(args.output)
    print(json.dumps({"path": str(args.output.resolve()), "repository": REPOSITORY,
                      "revision": REVISION, "sha256": SHA256, "bytes": SIZE}, indent=2))


if __name__ == "__main__":
    main()
