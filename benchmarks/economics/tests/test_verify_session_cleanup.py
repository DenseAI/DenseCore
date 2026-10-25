import importlib.util
import json
import os
import sys
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('cleanup', Path(__file__).parents[1] / 'verify_session_cleanup.py')
cleanup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleanup)


class CleanupVerificationTest(unittest.TestCase):
    def setUp(self):
        self.manifest = {'schema_version': 1, 'project': 'densecore-test', 'resources': [
            {'kind': 'instance', 'scope': 'zones/us-central1-a', 'name': 'session-vm', 'id': '100'},
            {'kind': 'disk', 'scope': 'zones/us-central1-a', 'name': 'session-disk', 'id': '101'},
            {'kind': 'address', 'scope': 'regions/us-central1', 'name': 'session-ip', 'id': '102'},
        ]}
        self.answers = [
            {'projectId': 'densecore-test', 'lifecycleState': 'ACTIVE'}, [], [], []]
        self.calls = []
        self.guard_answers = {}

    def runner(self, command, **kwargs):
        self.calls.append((command, kwargs))
        if command[1:3] == ['config', 'get-value']:
            answer = self.guard_answers.get(command[3], '0')
        else:
            index = sum(call[0][1] != 'config' for call in self.calls) - 1
            answer = self.answers[index]
        if isinstance(answer, BaseException):
            raise answer
        if isinstance(answer, subprocess.CompletedProcess):
            return answer
        return subprocess.CompletedProcess(command, 0, json.dumps(answer), '')

    def row(self, kind, name, identifier, scope='zones/us-central1-a'):
        return {'name': name, 'id': identifier,
                'selfLink': f'https://www.googleapis.com/compute/v1/projects/densecore-test/{scope}/{kind}/{name}'}

    def test_absence_checks_all_collections_without_mutating_commands(self):
        with patch.dict(os.environ, CLOUDSDK_CORE_DISABLE_PROMPTS='1',
                        CLOUDSDK_CORE_USER_OUTPUT_ENABLED='0', CLOUDSDK_CORE_LOG_HTTP='1'):
            result = cleanup.verify(self.manifest, self.runner)
        self.assertTrue(result['passed'])
        self.assertEqual(len(result['target_resources']), 3)
        self.assertEqual(len(self.calls), 6)
        for command, kwargs in self.calls:
            self.assertTrue('describe' in command or 'get-value' in command or command[3] == 'list')
            self.assertIn('--project=densecore-test', command)
            self.assertIn('--format=json', command)
            self.assertIn('--verbosity=warning', command)
            self.assertNotIn('--quiet', command)
            self.assertNotIn('--filter', command)
            self.assertNotIn('--limit', command)
            self.assertEqual(kwargs['stdin'], subprocess.DEVNULL)
            self.assertEqual(kwargs['env']['CLOUDSDK_CORE_DISABLE_PROMPTS'], '0')
            self.assertEqual(kwargs['env']['CLOUDSDK_CORE_SHOULD_PROMPT_TO_ENABLE_API'], '0')
            self.assertEqual(kwargs['env']['CLOUDSDK_COMPUTE_ALLOW_PARTIAL_ERROR'], '0')
            self.assertEqual(kwargs['env']['CLOUDSDK_CORE_USER_OUTPUT_ENABLED'], '1')
            self.assertEqual(kwargs['env']['CLOUDSDK_CORE_LOG_HTTP'], '0')
        for evidence in result['queries']:
            self.assertEqual(cleanup.digest(json.dumps(evidence['command'], separators=(',', ':'))),
                             evidence['command_sha256'])
            self.assertEqual(cleanup.digest(evidence['stdout']), evidence['stdout_sha256'])
            self.assertEqual(cleanup.digest(evidence['stderr']), evidence['stderr_sha256'])

    def test_stopped_vm_unattached_disk_and_unused_address_still_fail(self):
        for answer_index, row in [
            (1, self.row('instances', 'session-vm', '100')),
            (2, self.row('disks', 'session-disk', '101')),
            (3, self.row('addresses', 'session-ip', '102', 'regions/us-central1')),
        ]:
            with self.subTest(answer_index=answer_index):
                self.calls = []
                self.answers = [{'projectId': 'densecore-test', 'lifecycleState': 'ACTIVE'}, [], [], []]
                row.update(status='TERMINATED', users=[])
                self.answers[answer_index] = [row]
                result = cleanup.verify(self.manifest, self.runner)
                self.assertFalse(result['passed'])
                remaining = [target for target in result['target_resources'] if not target['absent']]
                self.assertEqual(len(remaining), 1)
                self.assertTrue(remaining[0]['same_id'])

    def test_same_uri_replacement_is_not_deleted_even_with_different_id(self):
        self.answers[1] = [self.row('instances', 'session-vm', '999')]
        result = cleanup.verify(self.manifest, self.runner)
        self.assertFalse(result['passed'])
        target = next(item for item in result['target_resources'] if item['uri'].endswith('instances/session-vm'))
        self.assertFalse(target['same_id'])
        self.assertEqual(target['observed_id'], '999')

    def test_same_name_in_another_zone_is_reported_separately(self):
        self.answers[1] = [self.row('instances', 'session-vm', '200', 'zones/us-east1-b')]
        result = cleanup.verify(self.manifest, self.runner)
        self.assertTrue(result['passed'])
        self.assertEqual(len(result['untracked_resources']), 1)
        self.assertIn('zones/us-east1-b', result['untracked_resources'][0]['uri'])

    def test_global_addresses_and_regional_disks_have_distinct_scopes(self):
        self.manifest['resources'] = [
            {'kind': 'disk', 'scope': 'regions/us-central1', 'name': 'regional-disk', 'id': '400'},
            {'kind': 'address', 'scope': 'global', 'name': 'global-ip', 'id': '401'}]
        self.answers[2] = [self.row('disks', 'regional-disk', '400', 'regions/us-central1')]
        self.answers[3] = [self.row('addresses', 'global-ip', '401', 'global')]
        result = cleanup.verify(self.manifest, self.runner)
        self.assertFalse(result['passed'])
        self.assertEqual(sum(not target['absent'] for target in result['target_resources']), 2)

    def test_permission_error_warning_timeout_and_malformed_json_are_not_absence(self):
        failures = [subprocess.CompletedProcess([], 1, '', 'PERMISSION_DENIED'),
                    subprocess.CompletedProcess([], 0, '[]', 'WARNING: partial results'),
                    subprocess.CompletedProcess([], 0, '{', ''),
                    subprocess.CompletedProcess([], 0, '{"items":[]}', ''),
                    subprocess.CompletedProcess([], 0, '{"items":[],"nextPageToken":"more"}', ''),
                    subprocess.CompletedProcess([], 0, '[{"warning":{"code":"UNREACHABLE"}}]', ''),
                    subprocess.CompletedProcess([], 0, '[] []', ''),
                    subprocess.CompletedProcess([], 0, '[NaN]', ''),
                    subprocess.CompletedProcess([], 0, '{"duplicate":1,"duplicate":2}', ''),
                    subprocess.TimeoutExpired('gcloud', 60), OSError('gcloud absent')]
        for answer in failures:
            with self.subTest(answer=repr(answer)):
                self.calls = []
                self.answers = [{'projectId': 'densecore-test', 'lifecycleState': 'ACTIVE'}, answer]
                result = cleanup.verify(self.manifest, self.runner)
                self.assertFalse(result['passed'])
                self.assertTrue(result['error'])
                self.assertEqual(len(self.calls), 4)

    def test_unknown_or_enabled_sdk_guard_prevents_provider_calls(self):
        for key in cleanup.SDK_GUARDS:
            for answer in ['1', True, '', None, subprocess.CompletedProcess([], 2, '', 'unknown property')]:
                with self.subTest(key=key, answer=repr(answer)):
                    self.calls = []
                    self.guard_answers = {key: answer}
                    report = cleanup.verify(self.manifest, self.runner)
                    self.assertFalse(report['passed'])
                    self.assertTrue(all(command[1:3] == ['config', 'get-value'] for command, _ in self.calls))
        self.guard_answers = {}

    def test_project_identity_or_lifecycle_mismatch_blocks_before_inventory(self):
        for project in [{'projectId': 'wrong-project', 'lifecycleState': 'ACTIVE'},
                        {'projectId': 'densecore-test', 'lifecycleState': 'DELETE_REQUESTED'}, []]:
            with self.subTest(project=project):
                self.calls = []
                self.answers = [project]
                self.assertFalse(cleanup.verify(self.manifest, self.runner)['passed'])
                self.assertEqual(len(self.calls), 3)

    def test_bad_inventory_rows_and_duplicate_uris_fail_closed(self):
        good = self.row('instances', 'session-vm', '100')
        for items in [[{}], [dict(good, id=0)], [dict(good, id=True)],
                      [dict(good, selfLink=good['selfLink'].replace('densecore-test', 'another-project'))],
                      [dict(good, selfLink=good['selfLink'].replace('https:', 'http:'))],
                      [dict(good, name='different-name')], [good, good]]:
            with self.subTest(items=items):
                self.calls = []
                self.answers = [{'projectId': 'densecore-test', 'lifecycleState': 'ACTIVE'}, items]
                self.assertFalse(cleanup.verify(self.manifest, self.runner)['passed'])
                self.assertEqual(len(self.calls), 4)

    def test_invalid_manifest_never_calls_gcloud(self):
        for document in [dict(self.manifest, schema_version=True), dict(self.manifest, resources=[]), dict(self.manifest, project='--bad'),
                         dict(self.manifest, resources=self.manifest['resources'] * 2),
                         dict(self.manifest, resources=[{'kind': 'disk', 'scope': 'global', 'name': 'disk', 'id': '1'}]),
                         dict(self.manifest, resources=[{'kind': 'instance', 'scope': 'zones/a', 'name': 'vm', 'id': '0'}])]:
            with self.subTest(document=document):
                with self.assertRaises(ValueError): cleanup.verify(document, self.runner)
        self.assertEqual(self.calls, [])

    def test_cli_refuses_overwrite_before_running_queries(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / 'manifest.json'
            output = Path(directory) / 'existing.json'
            manifest.write_text(json.dumps(self.manifest))
            output.write_text('preserved')
            with patch('sys.argv', ['verify', str(manifest), '--output', str(output)]), \
                    patch.object(cleanup, 'verify') as verify:
                with self.assertRaises(SystemExit) as error: cleanup.main()
            self.assertEqual(error.exception.code, 2)
            verify.assert_not_called()
            self.assertEqual(output.read_text(), 'preserved')


class CleanupCliTest(unittest.TestCase):
    def test_fake_gcloud_cli_preserves_success_and_failure_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / 'gcloud'
            fake.write_text(r"""#!/usr/bin/env python3
import json, os, sys
with open(os.environ['FAKE_CALLS'], 'a') as log:
    log.write(json.dumps(sys.argv[1:]) + '\n')
if sys.argv[1:3] == ['config', 'get-value']:
    print(json.dumps('0'))
elif sys.argv[1:3] == ['projects', 'describe']:
    print(json.dumps({'projectId': 'densecore-test', 'lifecycleState': 'ACTIVE'}))
else:
    print('[]')
    if os.environ.get('FAKE_WARNING') == '1':
        print('WARNING: partial results', file=sys.stderr)
""")
            fake.chmod(0o755)
            manifest = root / 'manifest.json'
            # Preserve raw bytes, including CRLF, in the manifest hash.
            raw = json.dumps({'schema_version': 1, 'project': 'densecore-test', 'resources': [
                {'kind': 'instance', 'scope': 'zones/us-central1-a', 'name': 'vm', 'id': '123'}]}, indent=2).replace('\n', '\r\n').encode()
            manifest.write_bytes(raw)
            env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ['PATH'],
                       FAKE_CALLS=str(root / 'calls.jsonl'))
            for warning, expected in [('0', 0), ('1', 1)]:
                with self.subTest(warning=warning):
                    output = root / f'report-{warning}.json'
                    result = subprocess.run([sys.executable, str(Path(cleanup.__file__)), str(manifest),
                                             '--output', str(output)], env=dict(env, FAKE_WARNING=warning),
                                            capture_output=True, text=True, timeout=20)
                    self.assertEqual(result.returncode, expected, result.stderr)
                    report = json.loads(output.read_text())
                    self.assertEqual(report['passed'], expected == 0)
                    self.assertEqual(report['manifest_sha256'], cleanup.hashlib.sha256(raw).hexdigest())
                    self.assertTrue(report['queries'])
                    self.assertIn('report_sha256', json.loads(result.stdout))
            commands = [json.loads(line) for line in (root / 'calls.jsonl').read_text().splitlines()]
            self.assertEqual(len(commands), 10)  # Six success queries + guards/project/first warning failure.
            for command in commands:
                self.assertTrue(command[:2] in (['config', 'get-value'], ['projects', 'describe']) or command[:1] == ['compute'])
                self.assertFalse(set(command) & {'delete', 'create', 'enable', 'update'})
