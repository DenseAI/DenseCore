import hashlib
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('download_model', Path(__file__).with_name('download_model.py'))
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)


class DownloadIntegrityTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.payload = b'GGUF model bytes'
        self.env = {'MODEL_REPO': 'owner/model', 'MODEL_FILE': 'model.gguf',
            'MODEL_REVISION': 'a' * 40, 'MODEL_SHA256': hashlib.sha256(self.payload).hexdigest()}
        self.patches = [patch.dict(os.environ, self.env, clear=True), patch.object(downloader, 'MODELS_DIR', self.root)]
        for context in self.patches:
            context.start(); self.addCleanup(context.stop)

    def test_cache_is_verified_before_alias_publication(self):
        (self.root / 'model.gguf').write_bytes(self.payload)
        with patch.object(downloader, 'download_with_retry') as download:
            self.assertEqual(downloader.main(), 0)
            download.assert_not_called()
        self.assertEqual((self.root / downloader.MAIN_MODEL_NAME).read_bytes(), self.payload)
        (self.root / downloader.MAIN_MODEL_NAME).write_bytes(b'corrupted')
        self.assertEqual(downloader.main(), 1)

    def test_wrong_download_is_never_published(self):
        def fetch(**kwargs):
            self.assertEqual(kwargs['revision'], 'a' * 40)
            path = kwargs['local_dir'] / 'model.gguf'
            path.write_bytes(b'wrong'); return path
        with patch.object(downloader, 'download_with_retry', side_effect=fetch):
            self.assertEqual(downloader.main(), 1)
        self.assertFalse((self.root / 'model.gguf').exists())
        self.assertFalse((self.root / downloader.MAIN_MODEL_NAME).exists())

    def test_unrelated_alias_does_not_bypass_requested_download(self):
        old = self.root / 'other.gguf'; old.write_bytes(b'other')
        (self.root / downloader.MAIN_MODEL_NAME).symlink_to(old)
        def fetch(**kwargs):
            path = kwargs['local_dir'] / 'model.gguf'; path.write_bytes(self.payload); return path
        with patch.object(downloader, 'download_with_retry', side_effect=fetch) as download:
            self.assertEqual(downloader.main(), 0)
            download.assert_called_once()
        self.assertEqual((self.root / downloader.MAIN_MODEL_NAME).read_bytes(), self.payload)

    def test_mutable_revision_missing_checksum_and_traversal_fail(self):
        for key, value in [('MODEL_REVISION', 'main'), ('MODEL_SHA256', ''), ('MODEL_FILE', '../escape.gguf')]:
            with self.subTest(key=key), patch.dict(os.environ, {key: value}):
                self.assertEqual(downloader.main(), 1)

    def test_new_revision_cannot_overwrite_previous_verified_artifact(self):
        def fetch(**kwargs):
            path = kwargs['local_dir'] / 'model.gguf'
            path.write_bytes(self.payload)
            return path
        with patch.object(downloader, 'download_with_retry', side_effect=fetch):
            self.assertEqual(downloader.main(), 0)
            previous = (self.root / downloader.MAIN_MODEL_NAME).resolve()
            old_bytes = previous.read_bytes()
            self.payload = b'new revision bytes'
            with patch.dict(os.environ, {'MODEL_REVISION': 'b' * 40,
                    'MODEL_SHA256': hashlib.sha256(self.payload).hexdigest()}):
                self.assertEqual(downloader.main(), 0)
            current = (self.root / downloader.MAIN_MODEL_NAME).resolve()
            self.assertNotEqual(previous, current)
            self.assertEqual(previous.read_bytes(), old_bytes)
            self.assertEqual(current.read_bytes(), self.payload)

    def test_cache_symlink_cannot_escape_models_directory(self):
        with tempfile.TemporaryDirectory() as outside:
            other = Path(outside) / 'model.gguf'; other.write_bytes(self.payload)
            (self.root / 'model.gguf').symlink_to(other)
            self.assertEqual(downloader.main(), 1)


if __name__ == '__main__': unittest.main()
