#!/usr/bin/env python3
"""Validate bound System One raw artifacts, then report encoder economics."""
import argparse
import json
from pathlib import Path
from systemone_economics import digest, load_workload, summarize, validate_artifacts, require


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('manifest', 'workload', 'quality', 'pricing', 'records', 'metadata', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--server-identity', type=Path, help='Same hash-bound HTTP server identity artifact used for the run')
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    _, price = validate_artifacts(manifest, args.workload, args.quality, args.pricing, args.server_identity)
    manifest['_sha256'] = digest(args.manifest)
    rows = [json.loads(line) for line in args.records.read_text().splitlines() if line.strip()]
    metadata = json.loads(args.metadata.read_text())
    require(metadata['records_sha256'] == digest(args.records), 'raw records hash mismatch')
    result = summarize(manifest, load_workload(args.workload), metadata, rows, price)
    result['manifest_sha256'] = manifest['_sha256']
    for key in ('host_id', 'threads', 'concurrency', 'started_at', 'finished_at', 'warmup_requests', 'warmup_rounds', 'memory', 'stage_times', 'records_sha256', 'run_id', 'batch_size', 'resource', 'resource_identity_status', 'observed_host', 'runtime_dependencies_sha256', 'runtime_dependencies_stable', 'runtime_dependencies_origin'):
        result[key] = metadata[key]
    if manifest['boundary'] == 'http_e2e':
        for key in ('server_identity_sha256', 'endpoint'):
            result[key] = metadata[key]
    if manifest['boundary'] == 'encoder_pipeline':
        for key in ('executable_sha256', 'native_library_sha256'):
            result[key] = metadata[key]
    result['bindings'] = {k: manifest[k] for k in ('model_sha256', 'tokenizer_sha256', 'implementation_sha256', 'precision', 'workload_sha256', 'quality_sha256', 'pricing_sha256', 'runtime_dependencies_sha256') + (('executable_sha256', 'native_library_sha256') if manifest['boundary'] == 'encoder_pipeline' else ())}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
