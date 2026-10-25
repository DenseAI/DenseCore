#!/usr/bin/env python3
"""Fetch pinned Julia-1 research artifacts without executing upstream code.

This is an artifact preparation tool, not a DenseCore Julia runtime or a quality
qualification. The pinned upstream provenance has quality_gate=false.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request

REPOSITORY = 'SupersonicLabs/Julia-1'
REVISION = 'a85b127321d580d65176c89ced8273f305745d85'
# Sizes/hashes independently read from the immutable HF tree and downloaded files.
FILES = {
    'model.safetensors': (577189056, 'df853bf7fe424420011f3d0c47a05d7341aa9eefa7fb9f203ea4aada4ad95b72'),
    'tokenizer/tokenizer.json': (34363188, '609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f'),
    'tokenizer/tokenizer_config.json': (652, '6b069e57db0ce0794c22547725275f22618809c4ef4249307337e651a0dfef8c'),
    'julia_config.json': (147, 'b9b881646beeac5414eaa78b39d3fe89ff8a2c062731ad718432620f9aee6bf7'),
    'encoder/config.json': (1936, 'c79cda42d42ddf777218254845ce92fb25a10dc35874be40d06abd707894523d'),
    'provenance.json': (11773, 'd3322e07820ad33b38e4bf5b4c4833c6b7e858ca553754fb8cac682ee802fd25'),
    'inference-policy.json': (355, '415468ae1c229186e63a277b2fd2e5ee3bbf4c00655125c0e43e01d5a8406e8a'),
    'julia/typed.py': (2415, 'ed89e66a70fcedac1339347bd8fcca69fd2cd1a31535ad0148dae42c610b544d'),
    'julia/model.py': (7539, 'ef2ba82fe20cdf0db7bb887e9ef075476ed08b985ce9a95be0de3e26246ecc81'),
    'julia/data.py': (7163, 'e3510fa4152ec11fa193046715991f44d7c2f85fd2488a98ef11c9d3db23da4e'),
    'julia/probabilities.py': (595, '2a0197d2e0fa5a3c4b06b93599a85706724df7b0d2cc821a13ed293aef59f206'),
    'julia/inference.py': (4613, '79b4e716365a6e07da4580ead725d5b76d06ef3a824a3ec23738feba64704b36'),
}


def manifest():
    return {
        'schema_version': 1, 'repository': REPOSITORY, 'revision': REVISION,
        'densecore_runtime': 'not_evaluated_by_acquisition', 'independent_parity': 'not_evaluated_by_acquisition',
        'upstream_quality_gate': False, 'gpu_cost_comparison': 'unavailable',
        'files': {name: {'bytes': size, 'sha256': digest,
                         'url': f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}'}
                  for name, (size, digest) in FILES.items()},
    }


def verify(path, size, digest):
    if not path.is_file() or path.stat().st_size != size:
        return False
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest() == digest


def fetch(root, name, spec):
    size, digest = spec
    destination = root / name
    if verify(destination, size, digest):
        return
    if destination.exists():
        raise ValueError(f'Refusing to replace existing file with different checksum: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.part', delete=False) as sink:
            temporary = Path(sink.name)
            url = f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}'
            with urllib.request.urlopen(url, timeout=60) as source:
                while chunk := source.read(1024 * 1024):
                    sink.write(chunk)
        if not verify(temporary, size, digest):
            raise ValueError(f'Julia artifact failed size/SHA-256 validation: {name}')
        # link() publishes atomically without overwriting a competing download.
        try:
            destination.hardlink_to(temporary)
        except FileExistsError:
            if not verify(destination, size, digest):
                raise ValueError(f'Destination appeared with different checksum: {destination}')
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('models/julia-1'))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--manifest-only', action='store_true', help='print pins without downloading')
    mode.add_argument('--verify-only', action='store_true', help='validate all local files without network')
    args = parser.parse_args()
    if not args.manifest_only:
        for name, spec in FILES.items():
            if args.verify_only:
                if not verify(args.output / name, *spec):
                    raise SystemExit(f'Missing or invalid Julia artifact: {args.output / name}')
            else:
                fetch(args.output, name, spec)
    result = manifest()
    result['artifacts_verified'] = not args.manifest_only
    result['path'] = str(args.output.resolve())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
