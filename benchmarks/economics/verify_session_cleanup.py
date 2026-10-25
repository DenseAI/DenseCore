#!/usr/bin/env python3
"""Record read-only evidence that explicitly listed session resources are absent."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import urlparse

COLLECTIONS = {'instance': 'instances', 'disk': 'disks', 'address': 'addresses'}
SDK_GUARDS = ('core/should_prompt_to_enable_api', 'compute/allow_partial_error')
NAME = re.compile(r'[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?\Z')
PROJECT = re.compile(r'[a-z][a-z0-9-]{4,28}[a-z0-9]\Z')
SCOPE = re.compile(r'(zones|regions)/[a-z][-a-z0-9]*\Z')


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'duplicate JSON key: {key}')
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError(f'invalid JSON number: {value}')
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant)


def resource_id(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError('resource id must be a positive decimal integer')
    value = str(value)
    if not re.fullmatch(r'[1-9][0-9]*', value) or int(value) > (1 << 64) - 1:
        raise ValueError('resource id must be a positive uint64 decimal integer')
    return value


def resource_uri(project, kind, scope, name):
    if not isinstance(kind, str) or kind not in COLLECTIONS or not isinstance(name, str) or not NAME.fullmatch(name):
        raise ValueError('unsupported resource kind or invalid resource name')
    if not isinstance(scope, str) or not (scope == 'global' or SCOPE.fullmatch(scope)):
        raise ValueError('scope must be zones/ZONE, regions/REGION, or global')
    if ((kind == 'instance' and not scope.startswith('zones/')) or
            (kind == 'disk' and scope == 'global') or
            (kind == 'address' and scope.startswith('zones/'))):
        raise ValueError('resource kind and scope do not match')
    return f'projects/{project}/{scope}/{COLLECTIONS[kind]}/{name}'


def validate_manifest(document):
    if (not isinstance(document, dict) or type(document.get('schema_version')) is not int or
            document['schema_version'] != 1):
        raise ValueError('cleanup manifest schema_version must be 1')
    project = document.get('project')
    if not isinstance(project, str) or not PROJECT.fullmatch(project):
        raise ValueError('project must be an explicit GCP project ID')
    resources = document.get('resources')
    if not isinstance(resources, list) or not resources:
        raise ValueError('explicit nonempty session resource inventory required')
    targets = {}
    for item in resources:
        if not isinstance(item, dict):
            raise ValueError('each session resource must be an object')
        uri = resource_uri(project, item.get('kind'), item.get('scope'), item.get('name'))
        if uri in targets:
            raise ValueError('duplicate session resource URI')
        targets[uri] = resource_id(item.get('id'))
    return project, targets


def inventory_uri(item, project, kind):
    if not isinstance(item, dict) or not isinstance(item.get('selfLink'), str):
        raise ValueError('inventory row lacks a resource selfLink')
    link = urlparse(item['selfLink'])
    if (link.scheme != 'https' or link.netloc not in ('www.googleapis.com', 'compute.googleapis.com') or
            link.query or link.fragment or not link.path.startswith('/compute/v1/projects/')):
        raise ValueError('unexpected inventory selfLink')
    uri = link.path[len('/compute/v1/'):]
    prefix = f'projects/{project}/'
    suffix = f'/{COLLECTIONS[kind]}/{item.get("name", "")}'
    if not uri.startswith(prefix) or not uri.endswith(suffix):
        raise ValueError('inventory project, kind, or resource name mismatch')
    scope = uri[len(prefix):-len(suffix)]
    if uri != resource_uri(project, kind, scope, item.get('name')):
        raise ValueError('invalid inventory resource URI')
    return uri, resource_id(item.get('id'))


def verify(document, runner=subprocess.run, now=None):
    project, targets = validate_manifest(document)
    report = {'schema_version': 1, 'project': project,
              'observed_at_utc': (now or datetime.now(timezone.utc)).isoformat(),
              'passed': False, 'queries': [], 'target_resources': [], 'untracked_resources': [],
              'scope': 'Only manifest resources; absence is not a zero-bill or final billing-settlement guarantee.'}
    env = dict(os.environ, CLOUDSDK_CORE_DISABLE_PROMPTS='0', CLOUDSDK_CORE_USER_OUTPUT_ENABLED='1',
               CLOUDSDK_CORE_SHOULD_PROMPT_TO_ENABLE_API='0', CLOUDSDK_COMPUTE_ALLOW_PARTIAL_ERROR='0',
               CLOUDSDK_CORE_LOG_HTTP='0', CLOUDSDK_CORE_DISABLE_USAGE_REPORTING='1',
               CLOUDSDK_COMPONENT_MANAGER_DISABLE_UPDATE_CHECK='1')

    def query(arguments):
        # API-enable hooks are disabled through the verified SDK guards below.
        # EOF alone is unsafe: SDK prompts may accept their default on EOF.
        command = ['gcloud', *arguments, f'--project={project}', '--format=json', '--verbosity=warning']
        row = {'command': command,
               'command_sha256': digest(json.dumps(command, separators=(',', ':')))}
        report['queries'].append(row)
        try:
            result = runner(command, stdin=subprocess.DEVNULL, capture_output=True,
                            text=True, timeout=60, env=env, check=False)
        except (OSError, subprocess.TimeoutExpired) as error:
            row['error'] = str(error)
            raise ValueError('inventory command failed or timed out') from error
        row.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                   stdout_sha256=digest(result.stdout), stderr_sha256=digest(result.stderr))
        # A successful partial aggregated list may still emit a warning. Such
        # output cannot certify absence, even if a returned JSON list is empty.
        if result.returncode != 0 or result.stderr.strip():
            raise ValueError('inventory command error or warning; absence unverified')
        return strict_json(result.stdout)

    try:
        # These properties guard both normal and batch Compute error hooks.
        # Verify that this installed SDK recognizes and applies them BEFORE any
        # provider inventory call; an unknown/ignored prerequisite fails closed.
        report['sdk_guards'] = {}
        for key in SDK_GUARDS:
            value = query(['config', 'get-value', key])
            disabled = (value is False or (type(value) is int and value == 0) or
                        (isinstance(value, str) and value.lower() in ('0', 'false')))
            report['sdk_guards'][key] = {'value': value, 'disabled': disabled}
            if not disabled:
                raise ValueError(f'installed gcloud safety property is not disabled: {key}')
        project_info = query(['projects', 'describe', project])
        if (not isinstance(project_info, dict) or project_info.get('projectId') != project or
                project_info.get('lifecycleState') != 'ACTIVE'):
            raise ValueError('active project identity was not verified')
        inventory = {}
        inventory_ids = {collection: {} for collection in COLLECTIONS.values()}
        for kind, collection in COLLECTIONS.items():
            items = query(['compute', collection, 'list'])
            if not isinstance(items, list):
                raise ValueError('inventory command did not return a complete JSON list')
            for item in items:
                uri, identifier = inventory_uri(item, project, kind)
                if uri in inventory:
                    raise ValueError('duplicate inventory resource URI')
                if identifier in inventory_ids[collection]:
                    raise ValueError(f'duplicate inventory resource ID within {collection}')
                inventory[uri] = identifier
                inventory_ids[collection][identifier] = uri
        for uri, identifier in sorted(targets.items()):
            observed_uri = inventory_ids[uri.split('/')[-2]].get(identifier)
            row = {'uri': uri, 'expected_id': identifier,
                   'absent': uri not in inventory and observed_uri is None}
            if observed_uri is not None:
                row.update(expected_id_observed_uri=observed_uri, uri_mismatch=observed_uri != uri)
            if uri in inventory:
                row.update(observed_id=inventory[uri], same_id=identifier == inventory[uri])
            report['target_resources'].append(row)
        report['untracked_resources'] = [{'uri': uri, 'id': identifier}
                                         for uri, identifier in sorted(inventory.items()) if uri not in targets]
        report['passed'] = all(item['absent'] for item in report['target_resources'])
        if not report['passed']:
            report['error'] = 'a session resource or same-URI replacement remains present'
    except (KeyError, TypeError, ValueError) as error:
        report['error'] = str(error)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True, help='new evidence file; never overwritten')
    args = parser.parse_args()
    try:
        if args.output.exists() or args.output.is_symlink():
            raise ValueError('output already exists; choose a new evidence path')
        if not args.output.parent.is_dir():
            raise ValueError('output parent directory must already exist')
        source = args.manifest.read_bytes().decode('utf-8')
        document = strict_json(source)
        validate_manifest(document)  # Reject invalid input before any gcloud call.
        report = verify(document)
        report['manifest_sha256'] = digest(source)
        report['verifier_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        with args.output.open('x') as handle:
            json.dump(report, handle, indent=2)
            handle.write('\n')
        print(json.dumps({'passed': report['passed'], 'output': str(args.output),
                          'report_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
                          'error': report.get('error')}))
        return 0 if report['passed'] else 1
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f'Cleanup verification failed: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
