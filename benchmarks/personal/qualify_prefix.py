#!/usr/bin/env python3
"""Require both prefix request/HTTP isolation and bounded snapshot eviction.

One model per invocation; no model/default/performance promotion is implied.
The candidate artifact and model bytes remain fixed under the shared build lock.
"""
import argparse
import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys

from check_mtp_parity import digest, write
from check_prefix_parity import compare
from compare_servers import validate_response


def commands(args):
    harness = Path(__file__).with_name('check_prefix_parity.py')
    common = [sys.executable, str(harness), '--library', str(args.library), '--model', str(args.model),
              '--threads', str(args.threads), '--context', str(args.context), '--churn', '8']
    return {
        'main': [*common, '--server', str(args.server), '--concurrency', '1', '2', '4',
                 '--port', str(args.port), '--out', str(args.out / 'main')],
        'pressure': [*common, '--concurrency', '1', '--near-full-pressure', '--require-eviction',
                     '--out', str(args.out / 'pressure')],
    }


def combined_gate(phases, model_sha, library_sha, server_sha, context):
    failures = []
    for name in ('main', 'pressure'):
        phase = phases.get(name)
        if not phase or phase.get('passed') is not True:
            failures.append(f'{name}: phase missing or failed')
            continue
        for field, expected in (('model_sha256', model_sha), ('library_sha256', library_sha)):
            if phase.get(field) != expected:
                failures.append(f'{name}: {field} differs from candidate')
        config = phase.get('configuration', {})
        expected_widths = [1, 2, 4] if name == 'main' else [1]
        if (config.get('context') != context or config.get('concurrency') != expected_widths or
                config.get('churn') != 8 or bool(config.get('near_full_pressure')) != (name == 'pressure')):
            failures.append(f'{name}: workload configuration differs from required gate')
        native = phase.get('native', {})
        try:
            failures.extend(f'{name}: {failure}' for failure in compare(native, name == 'pressure'))
            for mode, group in native.items():
                plan = group.get('runtime_plan', {})
                if (plan.get('effective_context') != context or
                        plan.get('effective_concurrency') != max(expected_widths) or
                        len(group.get('churn_requests', [])) != 8):
                    failures.append(f'{name}/{mode}: effective capacity or pressure rows differ')
                if name == 'pressure':
                    pressure = group.get('near_full_pressure', {})
                    rows = [row for row in group['requests'] if row['label'] == 'near-full-pressure']
                    count = pressure.get('prompt_tokens', 0)
                    if (not context - 33 <= count <= context - 17 or
                            pressure.get('requested_max_tokens') != 1 or pressure.get('effective_context') != context or
                            len(rows) != 1 or rows[0].get('generated_tokens') != 1):
                        failures.append(f'pressure/{mode}: bounded cold context evidence missing')
        except (KeyError, TypeError, ValueError) as error:
            failures.append(f'{name}: incomplete raw evidence: {error}')
    main = phases.get('main', {})
    if main.get('server_sha256') != server_sha or main.get('http_final_answer_verified') is not True:
        failures.append('main: matching server/final-answer HTTP verification missing')
    for mode in ('off', 'on'):
        rows = main.get('http', {}).get(mode, [])
        if len(rows) != 5:
            failures.append(f'main/HTTP {mode}: five turns missing')
            continue
        for turn, (row, city) in enumerate(zip(rows, ('Paris', 'Tokyo', 'Berlin', 'Rome', 'Madrid')), 1):
            try:
                if row.get('turn') != turn or row.get('error'):
                    raise ValueError('turn missing or errored')
                raw = row['response']
                choice = raw['choices'][0]
                message = choice['message']
                validate_response(message.get('content') or '', message.get('reasoning_content') or '',
                                  choice.get('finish_reason'), raw.get('usage', {}), rf'\b{city}\b')
            except (KeyError, IndexError, TypeError, ValueError) as error:
                failures.append(f'main/HTTP {mode}/turn{turn}: {error}')
    return failures


def verify_files(artifact):
    files = artifact.get('files', {})
    if not files:
        raise ValueError('candidate artifact has no file hashes')
    for path, expected in files.items():
        if digest(Path(path)) != expected:
            raise ValueError(f'candidate file changed: {path}')


def run(args):
    with args.build_lock.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.out.exists() and any(args.out.iterdir()):
            raise ValueError('output directory must be empty; preserve previous evidence')
        args.out.mkdir(parents=True, exist_ok=True)
        artifact = json.loads(args.artifact.read_text())
        verify_files(artifact)
        files = artifact['files']
        library_sha, server_sha = files.get(str(args.library)), files.get(str(args.server))
        if not library_sha or not server_sha:
            raise ValueError('library/server must be listed in the candidate artifact')
        source_revision = subprocess.check_output(
            ['git', '-C', str(args.library.parent), 'rev-parse', 'HEAD'], text=True).strip()
        if source_revision != artifact.get('revision'):
            raise ValueError('checkout revision differs from candidate artifact')
        if digest(args.model) != args.expected_model_sha256:
            raise ValueError('model bytes differ from pinned SHA')
        plan = commands(args)
        harness_files = [Path(__file__), Path(__file__).with_name('check_prefix_parity.py'),
                         Path(__file__).with_name('check_mtp_parity.py'), Path(__file__).with_name('compare_servers.py')]
        harness_hashes = {str(path): digest(path) for path in harness_files}
        report = {'passed': False, 'candidate_artifact': str(args.artifact),
                  'candidate_artifact_sha256': digest(args.artifact), 'revision': source_revision,
                  'model_sha256': args.expected_model_sha256, 'library_sha256': library_sha,
                  'server_sha256': server_sha, 'harness_files': harness_hashes, 'commands': plan,
                  'phase_returncodes': {}, 'phases': {},
                  'scope': 'main request/HTTP correctness AND separate c1 snapshot eviction; no default/performance claim'}
        output = args.out / 'combined.json'
        write(output, report)
        try:
            for name, command in plan.items():
                verify_files(artifact)
                with (args.out / f'{name}.log').open('w') as log:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                report['phase_returncodes'][name] = result.returncode
                summary = args.out / name / 'summary.json'
                if summary.exists():
                    report['phases'][name] = json.loads(summary.read_text())
                write(output, report)
                verify_files(artifact)
                if digest(args.model) != args.expected_model_sha256:
                    raise ValueError('model bytes changed during phase')
            failures = combined_gate(report['phases'], args.expected_model_sha256,
                                     library_sha, server_sha, args.context)
            failures.extend(f'{name}: child exit{code}' for name, code in report['phase_returncodes'].items() if code)
            for path, expected in harness_hashes.items():
                if digest(Path(path)) != expected:
                    failures.append(f'harness changed: {path}')
            if digest(args.artifact) != report['candidate_artifact_sha256']:
                failures.append('candidate artifact changed')
            report.update(passed=not failures, failures=failures)
        except BaseException as error:
            report.update(passed=False, error=str(error))
            write(output, report)
            raise
        write(output, report)
        print(f"{'PASS' if report['passed'] else 'FAIL'}: {output}")
        return int(not report['passed'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('artifact', 'library', 'server', 'model', 'out', 'build-lock'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-model-sha256', required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--context', type=int, default=2048)
    parser.add_argument('--port', type=int, default=18090)
    args = parser.parse_args()
    for name in ('artifact', 'library', 'server', 'model', 'out', 'build_lock'):
        setattr(args, name, getattr(args, name).resolve())
    if args.threads < 1 or args.context < 256:
        parser.error('threads must be positive and context must be at least256')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
