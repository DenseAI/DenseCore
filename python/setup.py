"""
DenseCore setup.py
"""

import os
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

        if platform.system() == "Windows":
            lib_patterns = ["densecore.dll", "libdensecore.dll"]
        elif platform.system() == "Darwin":
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

        # 3. Run the standard sdist command
        super().run()

        # 4. Cleanup (optional, but keeps the tree clean)
        if dest_dir.exists():
            print(f"Cleaning up {dest_dir}...")
            shutil.rmtree(dest_dir)
        if (PROJECT_DIR / "LICENSE").exists():
            os.remove(PROJECT_DIR / "LICENSE")


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
    packages=find_packages(),
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
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    package_data={
        "densecore": ["py.typed", "*.so", "*.dll"],
    },
    include_package_data=True,
    zip_safe=False,
)
