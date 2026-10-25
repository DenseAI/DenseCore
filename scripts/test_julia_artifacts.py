"""Offline artifact integrity and lossless Julia serialization regressions."""
import hashlib
import io
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import download_julia_model as artifacts
import julia_inputs


class Tokenizer:
    """Character tokenizer makes expected sequence boundaries transparent."""
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return type('Encoded', (), {'ids': [100 + ord(char) for char in text]})()


class JuliaArtifactsTest(unittest.TestCase):
    def test_integrity_reuse_and_user_file_preservation(self):
        payload = b'pinned artifact'
        spec = len(payload), hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(artifacts.urllib.request, 'urlopen', return_value=io.BytesIO(payload)) as fetch:
                artifacts.fetch(root, 'encoder/config.json', spec)
                artifacts.fetch(root, 'encoder/config.json', spec)
                self.assertEqual(fetch.call_count, 1)
            target = root / 'encoder/config.json'
            target.write_bytes(b'user file')
            with self.assertRaisesRegex(ValueError, 'Refusing'):
                artifacts.fetch(root, 'encoder/config.json', spec)
            self.assertEqual(target.read_bytes(), b'user file')

    def test_corrupt_download_not_published(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(artifacts.urllib.request, 'urlopen', return_value=io.BytesIO(b'bad')):
            with self.assertRaisesRegex(ValueError, 'validation'):
                artifacts.fetch(Path(tmp), 'weight', (3, '0' * 64))
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_manifest_does_not_promote_artifact(self):
        result = artifacts.manifest()
        self.assertFalse(result['upstream_quality_gate'])
        self.assertEqual(result['densecore_runtime'], 'not_evaluated_by_acquisition')
        self.assertEqual(result['independent_parity'], 'not_evaluated_by_acquisition')

    def test_exact_boundaries_and_no_padding_count(self):
        row = {'state': 'S', 'question': 'Q', 'options': ['A', 'B']}
        got = julia_inputs.encode_request(Tokenizer(), row, max_length=64, head_length=32)
        encode = lambda text: [100 + ord(char) for char in text]
        expected = [2] + encode('choice question: Q') + [1, 4] + encode(' A') + [4] + encode(' B') + [1] + encode('S') + [1]
        self.assertEqual(got['ids'], expected)
        self.assertEqual([got['ids'][i] for i in got['markers']], [4, 4])
        self.assertEqual(got['valid_input_tokens'], len(expected))
        self.assertFalse(got['truncated'])

    def test_default_uses_checkpoint_policy_512_head_budget(self):
        row = {'state': 'S', 'question': 'Q' * 300, 'options': ['A', 'B']}
        encoded = julia_inputs.encode_request(Tokenizer(), row)
        self.assertGreater(len(encoded['ids']), 256)
        with self.assertRaisesRegex(ValueError, 'head'):
            julia_inputs.encode_request(Tokenizer(), row, head_length=256)

    def test_overflow_and_reserved_marker_rejected(self):
        base = {'state': 'S', 'question': 'Q', 'options': ['A', 'B']}
        for update, pattern in [({'state': 'S' * 100}, 'context'), ({'question': 'Q' * 30}, 'head'), ({'options': ['A' * 48, 'B']}, '48-token'), ({'state': '<mask>'}, 'reserved')]:
            with self.subTest(update=update), self.assertRaisesRegex(ValueError, pattern):
                julia_inputs.encode_request(Tokenizer(), base | update, max_length=64, head_length=32)

    def test_raw_probabilities_remain_available(self):
        result = julia_inputs.probabilities([0, -6, -7, -8])
        self.assertEqual(result['display_probabilities'], [1, 0, 0, 0])
        self.assertLess(result['raw_probabilities'][0], 1)
        self.assertGreater(result['raw_probabilities'][1], 0)
        self.assertAlmostEqual(sum(result['raw_probabilities']), 1)
        self.assertEqual(julia_inputs.probabilities([1, 1])['index'], 0)
        with self.assertRaisesRegex(ValueError, 'finite'):
            julia_inputs.probabilities([math.nan, 0])
        with self.assertRaisesRegex(ValueError, '2..20'):
            julia_inputs.probabilities([0] * 101)


if __name__ == '__main__':
    unittest.main()
