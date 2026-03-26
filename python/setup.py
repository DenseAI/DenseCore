"""
DenseCore setup.py
"""

import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.sdist import sdist

# Project directory (python/)
PROJECT_DIR = Path(__file__).parent.resolve()
# Root directory (DenseCore/)
ROOT_DIR = PROJECT_DIR.parent.resolve()


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    @staticmethod
    def _runtime_rpath() -> str:
        if platform.system() == "Darwin":
            return "@loader_path"
        return "$ORIGIN"

    @staticmethod
    def _copy_file(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    def _copy_matching_runtime_libs(self, cmake_output_dir: Path, extdir: Path) -> None:
        prefixes = (
            "libdensecore",
            "libggml",
            "libspdlog",
            "libhwy",
            "libgomp",
        )
        skip_prefixes = ("libhwy_test",)
        patterns = ("*.so*", "*.dylib", "*.dylib*")

        copied: set[str] = set()
        for pattern in patterns:
            for candidate in sorted(cmake_output_dir.glob(pattern)):
                name = candidate.name
                if not any(name.startswith(prefix) for prefix in prefixes):
                    continue
                if any(name.startswith(prefix) for prefix in skip_prefixes):
                    continue
                if candidate.is_dir():
                    continue
                if name in copied:
                    continue
                self._copy_file(candidate.resolve(), extdir / name)
                copied.add(name)

    def _copy_linux_runtime_dependency(self, built_lib: Path, extdir: Path, soname: str) -> None:
        if platform.system() != "Linux":
            return

        try:
            output = subprocess.check_output(["ldd", str(built_lib)], text=True)
        except Exception:
            return

        pattern = re.compile(rf"\b{re.escape(soname)}\b => (\S+)")
        for line in output.splitlines():
            match = pattern.search(line)
            if not match:
                continue
            dep_path = Path(match.group(1))
            if dep_path.exists():
                self._copy_file(dep_path.resolve(), extdir / soname)
            return

    def build_extension(self, ext):
        # Get the full path where setuptools expects the extension module
        ext_fullpath = Path(self.get_ext_fullpath(ext.name)).resolve()
        extdir = ext_fullpath.parent

        # Ensure the directory exists
        extdir.mkdir(parents=True, exist_ok=True)

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # Set CMake args
        # Note: CMakeLists.txt forces CMAKE_LIBRARY_OUTPUT_DIRECTORY to CMAKE_BINARY_DIR
        # so we don't set it here, and we expect the output in build_temp.
        cmake_args = [
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            "-DDENSECORE_BUILD_TESTS=OFF",
            "-DDENSECORE_USE_MIMALLOC=OFF",
            f"-DCMAKE_BUILD_RPATH={self._runtime_rpath()}",
            f"-DCMAKE_INSTALL_RPATH={self._runtime_rpath()}",
            "-DCMAKE_SKIP_BUILD_RPATH=OFF",
            "-DCMAKE_BUILD_WITH_INSTALL_RPATH=OFF",
        ]

        # Build args
        build_args = []
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            if hasattr(self, "parallel") and self.parallel:
                build_args += [f"-j{self.parallel}"]

        # Detect if we are building from a source distribution (sdist)
        bundled_core_src = PROJECT_DIR / "densecore" / "core_src"
        if bundled_core_src.exists():
            ext.sourcedir = str(bundled_core_src)
        else:
            core_dir = ROOT_DIR / "core"
            if not core_dir.exists():
                raise RuntimeError(f"Cannot find core directory at {core_dir}")
            ext.sourcedir = str(core_dir)

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # Config and Build
        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)

        # Find the built library and copy to the expected location
        # CMake outputs directly to build_temp because of its configuration
        cmake_output_dir = Path(self.build_temp).resolve()

        import platform

        if platform.system() == "Darwin":
            lib_patterns = ["libdensecore.dylib", "libdensecore.so"]
        else:
            lib_patterns = ["libdensecore.so"]

        built_lib = None
        for pattern in lib_patterns:
            candidate = cmake_output_dir / pattern
            if candidate.exists():
                built_lib = candidate
                break

        if built_lib:
            # Copy to the path setuptools expects
            print(f"Copying {built_lib} -> {ext_fullpath}")
            shutil.copy2(built_lib, ext_fullpath)
            self._copy_matching_runtime_libs(cmake_output_dir, extdir)
            self._copy_linux_runtime_dependency(built_lib, extdir, "libgomp.so.1")
        else:
            raise RuntimeError(
                f"CMake build completed but library not found. "
                f"Looked for {lib_patterns} in {cmake_output_dir}"
            )


class CustomSdist(sdist):
    @staticmethod
    def _ignore_core_tree(_dir: str, names: list[str]) -> set[str]:
        ignored = set()
        ignored_exact = {
            ".cache",
            ".git",
            ".github",
            "_deps",
            "CMakeFiles",
            "Testing",
            "bin",
            "build",
            "cmake-build-debug",
            "lib",
            "examples",
            "docs",
            "ci",
        }
        ignored_prefixes = ("build-", "build_")
        ignored_suffixes = (
            ".a",
            ".dll",
            ".dylib",
            ".o",
            ".obj",
            ".pyc",
            ".pyd",
            ".pyo",
            ".so",
        )
        ignored_files = {
            "CMakeCache.txt",
            "CTestTestfile.cmake",
            "Makefile",
            "cmake_install.cmake",
            "compile_commands.json",
        }

        for name in names:
            if name in ignored_exact or name in ignored_files:
                ignored.add(name)
                continue
            if any(name.startswith(prefix) for prefix in ignored_prefixes):
                ignored.add(name)
                continue
            if name.endswith(ignored_suffixes) or name == "__pycache__":
                ignored.add(name)

        return ignored

    def run(self):
        # When creating a source distribution, we need to bundle the C++ source
        # into the package so that it can be built by users.

        # 1. Define source and destination
        core_src = ROOT_DIR / "core"
        dest_dir = PROJECT_DIR / "densecore" / "core_src"

        # 2. Check if we are in the repo
        if core_src.exists():
            print(f"Bundling C++ source from {core_src} to {dest_dir}...")
            if dest_dir.exists():
                shutil.rmtree(dest_dir)

            # Copy, ignoring build artifacts and hidden files
            shutil.copytree(
                core_src,
                dest_dir,
                ignore=self._ignore_core_tree,
            )
            self._sanitize_bundled_core_tree(dest_dir)

            # Also copy root files if needed (LICENSE, README)
            # README is already handled by setup() metadata, LICENSE usually via MANIFEST.in
            # But let's verify LICENSE existence
            license_src = ROOT_DIR / "LICENSE"
            if license_src.exists():
                shutil.copy(license_src, PROJECT_DIR / "LICENSE")

        else:
            print(
                "Warning: ../core not found. Assuming we are already in an sdist or simplified environment."
            )

        # Remove stale manifest metadata so setuptools regenerates SOURCES.txt
        # from the freshly bundled tree instead of following old build artifact entries.
        for stale_path in PROJECT_DIR.glob("*.egg-info"):
            if stale_path.is_dir():
                shutil.rmtree(stale_path)
            elif stale_path.exists():
                stale_path.unlink()

        # 3. Run the standard sdist command
        super().run()

        # 4. Cleanup (optional, but keeps the tree clean)
        if dest_dir.exists():
            print(f"Cleaning up {dest_dir}...")
            shutil.rmtree(dest_dir)
        if (PROJECT_DIR / "LICENSE").exists():
            os.remove(PROJECT_DIR / "LICENSE")

    @staticmethod
    def _sanitize_bundled_core_tree(dest_dir: Path) -> None:
        removable_dirs = {
            "_deps",
            "CMakeFiles",
            "Testing",
            "build",
            "lib",
            "bin",
            "cmake-build-debug",
        }
        removable_files = {
            "CMakeCache.txt",
            "CTestTestfile.cmake",
            "Makefile",
            "cmake_install.cmake",
            "compile_commands.json",
            "densecore_tests[1]_include.cmake",
        }
        removable_suffixes = (".a", ".dll", ".dylib", ".o", ".obj", ".pyc", ".pyd", ".pyo", ".so")

        for path in sorted(dest_dir.rglob("*")):
            name = path.name
            if path.is_dir():
                if name in removable_dirs or name.startswith("build-") or name.startswith("build_"):
                    shutil.rmtree(path, ignore_errors=True)
                continue

            if name in removable_files:
                path.unlink(missing_ok=True)
                continue
            if name.endswith(removable_suffixes) or ".so." in name or ".dylib." in name:
                path.unlink(missing_ok=True)


# Read long description from README
readme_path = PROJECT_DIR / "README.md"
long_description = ""
if readme_path.exists():
    with open(readme_path, encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="densecore",
    version="1.0.0",
    author="DenseCore Team",
    author_email="jake@densecore.ai",
    description="High-Performance CPU Inference Engine for LLMs with HuggingFace Integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Jake-Network/DenseCore",
    packages=find_packages(include=["densecore", "densecore.*"]),
    # Add CMakeExtension
    ext_modules=[CMakeExtension("densecore.libdensecore")],
    cmdclass={
        "build_ext": CMakeBuild,
        "sdist": CustomSdist,
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    package_data={
        "densecore": [
            "py.typed",
            "*.so",
            "*.so.*",
            "*.dylib",
            "*.dylib.*",
        ],
    },
    include_package_data=False,
    zip_safe=False,
)
