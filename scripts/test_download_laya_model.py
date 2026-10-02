"""Offline integrity/overwrite regression tests for the model downloader."""
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import download_laya_model as downloader


class DownloadTest(unittest.TestCase):
    def test_verified_download_and_cached_reuse(self):
        payload = b"test model payload"
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(downloader, "SIZE", len(payload)), \
                patch.object(downloader, "SHA256", hashlib.sha256(payload).hexdigest()), \
                patch.object(downloader.urllib.request, "urlopen", return_value=io.BytesIO(payload)) as fetch:
            target = Path(tmp) / "model.gguf"
            downloader.download(target)
            downloader.download(target)
            self.assertEqual(target.read_bytes(), payload)
            self.assertEqual(fetch.call_count, 1)
            self.assertEqual(list(Path(tmp).iterdir()), [target])

    def test_existing_different_file_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(downloader.urllib.request, "urlopen") as fetch:
            target = Path(tmp) / "model.gguf"
            target.write_bytes(b"user-owned file")
            with self.assertRaisesRegex(ValueError, "Refusing to replace"):
                downloader.download(target)
            self.assertEqual(target.read_bytes(), b"user-owned file")
            fetch.assert_not_called()

    def test_corrupt_download_is_not_published(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(downloader.urllib.request, "urlopen", return_value=io.BytesIO(b"bad")):
            target = Path(tmp) / "model.gguf"
            with self.assertRaisesRegex(ValueError, "validation"):
                downloader.download(target)
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
