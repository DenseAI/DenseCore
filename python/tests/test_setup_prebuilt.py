"""Packaging contract tests; no native build or shared-library loading."""
import os
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

from setuptools import Distribution


class PrebuiltWheelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch("setuptools.setup"):
            cls.Build = runpy.run_path(str(Path(__file__).parents[1] / "setup.py"))["CMakeBuild"]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.native = self.root / "native"
        self.native.mkdir()
        self.output = self.root / "wheel" / "densecore" / "libdensecore.test.so"
        self.command = self.Build(Distribution())
        self.command.build_temp = str(self.root / "cmake")
        self.command.get_ext_fullpath = lambda name: str(self.output)
        self.ext = type("Extension", (), {"name": "densecore.libdensecore", "sourcedir": str(self.root)})()

    def test_prebuilt_copies_runtime_and_skips_cmake(self):
        (self.native / "libdensecore.so").write_bytes(b"native-payload")
        (self.native / "libdensecore.so.1").symlink_to("libdensecore.so")
        (self.native / "libggml.so").write_bytes(b"ggml-runtime")
        (self.native / "libhwy_test.so").write_bytes(b"excluded-test-library")
        with patch.dict(os.environ, {"DENSECORE_PREBUILT_LIB_DIR": str(self.native)}), patch("platform.system", return_value="Linux"), patch("subprocess.check_call") as cmake, patch.object(self.command, "_copy_linux_runtime_dependency") as gomp:
            self.command.build_extension(self.ext)
        cmake.assert_not_called()
        gomp.assert_called_once_with(self.native / "libdensecore.so", self.output.parent, "libgomp.so.1")
        self.assertEqual(self.output.read_bytes(), b"native-payload")
        self.assertEqual((self.output.parent / "libdensecore.so.1").read_bytes(), b"native-payload")
        self.assertEqual((self.output.parent / "libggml.so").read_bytes(), b"ggml-runtime")
        self.assertFalse((self.output.parent / "libhwy_test.so").exists())

    def test_explicit_invalid_prebuilt_never_falls_back_to_build(self):
        for path in ["", str(self.root / "missing"), str(self.native)]:
            with self.subTest(path=path), patch.dict(os.environ, {"DENSECORE_PREBUILT_LIB_DIR": path}), patch("subprocess.check_call") as cmake:
                with self.assertRaises(RuntimeError):
                    self.command.build_extension(self.ext)
                cmake.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_default_still_configures_and_builds_with_mimalloc_off(self):
        environment = dict(os.environ)
        environment.pop("DENSECORE_PREBUILT_LIB_DIR", None)
        commands = []
        def fake_cmake(command, **kwargs):
            commands.append(command)
            if len(commands) == 2:
                (Path(self.command.build_temp) / "libdensecore.so").write_bytes(b"fresh-build")
        with patch.dict(os.environ, environment, clear=True), patch("platform.system", return_value="Linux"), patch("subprocess.check_call", side_effect=fake_cmake), patch.object(self.command, "_copy_linux_runtime_dependency"):
            self.command.build_extension(self.ext)
        self.assertEqual(len(commands), 2)
        self.assertEqual(commands[0][0], "cmake")
        self.assertIn("-DDENSECORE_USE_MIMALLOC=OFF", commands[0])
        self.assertEqual(commands[1][:3], ["cmake", "--build", "."])
        self.assertEqual(self.output.read_bytes(), b"fresh-build")


if __name__ == "__main__":
    unittest.main()
