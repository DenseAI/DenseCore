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
    RUNTIME_LIBRARY_PREFIXES = (
        "libdensecore",
        "libggml",
        "libspdlog",
        "libhwy",
        "libgomp",
    )

    @staticmethod
    def _runtime_rpath() -> str:
        if platform.system() == "Darwin":
            return "@loader_path"
        return "$ORIGIN"

    @staticmethod
    def _copy_file(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    @classmethod
    def _remove_runtime_libs(cls, directory: Path) -> None:
        if not directory.exists():
            return
        for prefix in cls.RUNTIME_LIBRARY_PREFIXES:
            for candidate in directory.glob(f"{prefix}*"):
                if candidate.is_file() or candidate.is_symlink():
                    candidate.unlink()

    def _copy_matching_runtime_libs(self, cmake_output_dir: Path, extdir: Path) -> None:
        skip_prefixes = ("libhwy_test",)
        patterns = ("*.so*", "*.dylib", "*.dylib*")
        active_targets = {
            candidate.resolve()
            for pattern in ("*.so", "*.dylib")
            for candidate in cmake_output_dir.glob(pattern)
            if any(candidate.name.startswith(prefix) for prefix in self.RUNTIME_LIBRARY_PREFIXES)
        }

        copied: set[str] = set()
        for pattern in patterns:
            for candidate in sorted(cmake_output_dir.glob(pattern)):
                name = candidate.name
                if not any(name.startswith(prefix) for prefix in self.RUNTIME_LIBRARY_PREFIXES):
                    continue
                if any(name.startswith(prefix) for prefix in skip_prefixes):
                    continue
                if candidate.is_dir():
                    continue
                if (
                    not name.endswith((".so", ".dylib"))
                    and candidate.resolve() not in active_targets
                ):
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
        self._remove_runtime_libs(extdir)

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

        cmake_output_dir = Path(self.build_temp).resolve()

        # Config and Build
        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)

        # Find the built library and copy to the expected location
        # CMake outputs directly to build_temp because of its configuration
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
            ".codex",
            ".idea",
            ".omx",
            ".vscode",
            ".git",
            ".github",
            "_deps",
            "CMakeFiles",
            "Testing",
            "bin",
            "build",
            "build_mimalloc",
            "cmake-build-debug",
            "graphify-out",
            "lib",
            "examples",
            "docs",
            "ci",
            "tests",
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
            ".idx",
        )
        ignored_files = {
            "CMakeCache.txt",
            "CTestTestfile.cmake",
            "CTestCostData.txt",
            "LastTest.log",
            "Makefile",
            "cmake_install.cmake",
            "compile_commands.json",
            ".clang-format",
            ".graphifyignore",
            "AGENTS.md",
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

        try:
            super().run()
        finally:
            if dest_dir.exists():
                print(f"Cleaning up {dest_dir}...")
                shutil.rmtree(dest_dir)
            for stale_path in PROJECT_DIR.glob("*.egg-info"):
                if stale_path.is_dir():
                    shutil.rmtree(stale_path)
                elif stale_path.exists():
                    stale_path.unlink()

    @staticmethod
    def _sanitize_bundled_core_tree(dest_dir: Path) -> None:
        removable_dirs = {
            ".cache",
            "_deps",
            "CMakeFiles",
            "Testing",
            "build",
            "build_mimalloc",
            "lib",
            "bin",
            "cmake-build-debug",
            "graphify-out",
            "tests",
        }
        removable_files = {
            "CMakeCache.txt",
            "CTestTestfile.cmake",
            "CTestCostData.txt",
            "LastTest.log",
            "Makefile",
            "cmake_install.cmake",
            "compile_commands.json",
            "densecore_tests[1]_include.cmake",
            ".clang-format",
            ".graphifyignore",
            "AGENTS.md",
        }
        removable_suffixes = (
            ".a",
            ".dll",
            ".dylib",
            ".idx",
            ".o",
            ".obj",
            ".pyc",
            ".pyd",
            ".pyo",
            ".so",
        )

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
    version="0.1.0",
    author="DenseCore Team",
    author_email="jake@densecore.ai",
    description="High-Performance CPU Inference Engine for LLMs with HuggingFace Integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DenseAI/DenseCore",
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
            "prompt_profiles.json",
            "*.so",
            "*.so.*",
            "*.dylib",
            "*.dylib.*",
        ],
    },
    include_package_data=False,
    zip_safe=False,
)
