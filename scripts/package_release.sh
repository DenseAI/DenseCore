#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: scripts/package_release.sh <version> [--binary <build-dir>] [--server <server-bin>] [--licenses <licenses-dir>] [--out <out-dir>] [--source-only]

Creates:
  densecore-<version>-source.tar.gz
  densecore-<version>-source.tar.gz.sha256
  densecore-<version>-linux-amd64.tar.gz
  densecore-<version>-linux-amd64.tar.gz.sha256

The source archive is built from tracked files plus recursive submodule
contents. The binary archive is for the generic portable linux-amd64 artifact
and requires a portable DenseCore build.
USAGE
}

if [ "$#" -lt 1 ]; then
  usage
  exit 1
fi

version=$1
shift

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
root_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
build_dir="$root_dir/build"
server_bin="$root_dir/server/densecore-server-linux-amd64"
licenses_dir="$root_dir/release/licenses"
out_dir="$root_dir/release"
source_only=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --binary)
      build_dir=$2
      shift 2
      ;;
    --server)
      server_bin=$2
      shift 2
      ;;
    --licenses)
      licenses_dir=$2
      shift 2
      ;;
    --out)
      out_dir=$2
      shift 2
      ;;
    --source-only)
      source_only=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

mkdir -p "$out_dir"
tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/densecore-release.XXXXXX")
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

require_file() {
  if [ ! -f "$1" ]; then
    echo "required release file missing: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [ ! -d "$1" ]; then
    echo "required release directory missing: $1" >&2
    exit 1
  fi
}

portable_cache_value() {
  local name=$1
  local cache_file="$build_dir/CMakeCache.txt"
  if [ -f "$cache_file" ]; then
    sed -n "s/^${name}:[^=]*=//p" "$cache_file" | tail -n 1
  fi
}

compiler_version() {
  local compiler=$1
  if [ -n "$compiler" ] && command -v "$compiler" >/dev/null 2>&1; then
    "$compiler" --version 2>/dev/null | sed -n '1p'
  else
    echo "unknown"
  fi
}

json_escape() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

require_file "$root_dir/LICENSE"
require_file "$root_dir/NOTICE"
require_file "$root_dir/THIRD_PARTY_NOTICES.md"
require_file "$root_dir/README.md"
require_file "$root_dir/docs/RELEASE.md"
require_file "$root_dir/server/openapi.yaml"
require_file "$root_dir/scripts/entrypoint.sh"
require_file "$root_dir/scripts/release_smoke.sh"
require_file "$root_dir/scripts/container_model_smoke.sh"
require_file "$root_dir/.gitmodules"
require_dir "$root_dir/core/third_party/ggml"

source_root="$tmp_dir/densecore-${version}"
mkdir -p "$source_root"
git -C "$root_dir" archive HEAD | tar -x -C "$source_root"

git -C "$root_dir" submodule status --recursive | while IFS= read -r line; do
  status=${line:0:1}
  rest=${line:1}
  set -- $rest
  path=${2:-}
  if [ -z "${path:-}" ]; then
    continue
  fi
  if [ "$status" = "-" ]; then
    echo "submodule is not initialized: $path" >&2
    exit 1
  fi
  mkdir -p "$source_root/$path"
  git -C "$root_dir/$path" archive HEAD | tar -x -C "$source_root/$path"
done

source_tar="$out_dir/densecore-${version}-source.tar.gz"
tar -czf "$source_tar" -C "$tmp_dir" "densecore-${version}"
tar -tzf "$source_tar" > "$tmp_dir/source-files.txt"
for required_path in \
  LICENSE \
  NOTICE \
  THIRD_PARTY_NOTICES.md \
  README.md \
  charts/densecore/Chart.yaml \
  core/CMakeLists.txt \
  core/cmake/aarch64-toolchain.cmake \
  core/src/runtime/engine.cpp \
  core/third_party/ggml/LICENSE \
  proto/densecore.proto \
  python/pyproject.toml \
  server/go.mod \
  server/go.sum \
  server/openapi.yaml \
  docs/RELEASE.md \
  scripts/collect_release_licenses.sh \
  scripts/entrypoint.sh \
  scripts/package_release.sh \
  scripts/release_smoke.sh \
  scripts/container_model_smoke.sh; do
  grep -Fqx "densecore-${version}/${required_path}" "$tmp_dir/source-files.txt" || {
    echo "source archive is missing required release file: $required_path" >&2
    exit 1
  }
done
(cd "$out_dir" && sha256sum "$(basename "$source_tar")") > "$source_tar.sha256"

if [ "$source_only" -eq 1 ]; then
  exit 0
fi

portable=$(portable_cache_value DENSECORE_PORTABLE)
ggml_native=$(portable_cache_value GGML_NATIVE)
build_type=$(portable_cache_value CMAKE_BUILD_TYPE)
c_compiler=$(portable_cache_value CMAKE_C_COMPILER)
cxx_compiler=$(portable_cache_value CMAKE_CXX_COMPILER)
cache_sha256=$(sha256sum "$build_dir/CMakeCache.txt" | awk '{print $1}')

if [ "$portable" != "ON" ] && [ "$portable" != "TRUE" ]; then
  echo "binary release requires DENSECORE_PORTABLE=ON in $build_dir/CMakeCache.txt; got '${portable:-unset}'" >&2
  exit 1
fi
if [ "$ggml_native" != "OFF" ] && [ "$ggml_native" != "FALSE" ]; then
  echo "binary release requires GGML_NATIVE=OFF in $build_dir/CMakeCache.txt; got '${ggml_native:-unset}'" >&2
  exit 1
fi

require_file "$server_bin"
require_dir "$licenses_dir"
require_file "$licenses_dir/LICENSE"
require_file "$licenses_dir/NOTICE"
require_file "$licenses_dir/THIRD_PARTY_NOTICES.md"

binary_root="$tmp_dir/binary"
mkdir -p "$binary_root/licenses"

shopt -s nullglob
densecore_libs=("$build_dir"/libdensecore.so*)
ggml_libs=("$build_dir"/libggml*.so*)
if (( ${#densecore_libs[@]} == 0 || ${#ggml_libs[@]} == 0 )); then
  echo "required DenseCore or ggml shared libraries are missing from $build_dir" >&2
  exit 1
fi

cp -P "${densecore_libs[@]}" "${ggml_libs[@]}" "$binary_root/"
cp "$server_bin" "$binary_root/densecore-server.bin"
cat > "$binary_root/densecore-server" <<'LAUNCHER'
#!/bin/sh
set -eu
package_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export LD_LIBRARY_PATH="$package_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$package_dir/densecore-server.bin" "$@"
LAUNCHER
chmod +x "$binary_root/densecore-server" "$binary_root/densecore-server.bin"
cp -R "$licenses_dir"/. "$binary_root/licenses/"
mkdir -p "$binary_root/docs" "$binary_root/server" "$binary_root/scripts"
cp "$root_dir/README.md" "$binary_root/README.md"
cp "$root_dir/docs/RELEASE.md" "$binary_root/docs/RELEASE.md"
cp "$root_dir/server/openapi.yaml" "$binary_root/server/openapi.yaml"
cp "$root_dir/scripts/entrypoint.sh" "$binary_root/scripts/entrypoint.sh"
cp "$root_dir/scripts/release_smoke.sh" "$binary_root/scripts/release_smoke.sh"
cp "$root_dir/scripts/container_model_smoke.sh" "$binary_root/scripts/container_model_smoke.sh"
chmod +x "$binary_root/scripts/entrypoint.sh" "$binary_root/scripts/release_smoke.sh" \
  "$binary_root/scripts/container_model_smoke.sh"

commit=$(git -C "$root_dir" rev-parse HEAD)
short_commit=$(git -C "$root_dir" rev-parse --short HEAD)
tag=$(git -C "$root_dir" describe --tags --exact-match 2>/dev/null || true)
model_hash=""
if [ -n "${DENSECORE_SMOKE_MODEL:-}" ] && [ -f "$DENSECORE_SMOKE_MODEL" ]; then
  model_hash=$(sha256sum "$DENSECORE_SMOKE_MODEL" | awk '{print $1}')
fi

cat > "$binary_root/provenance.json" <<EOF
{
  "version": "$version",
  "git_commit": "$commit",
  "git_short_commit": "$short_commit",
  "git_tag": "$tag",
  "compiler_c": "$(json_escape "$(basename "${c_compiler:-unknown}")")",
  "compiler_cxx": "$(json_escape "$(basename "${cxx_compiler:-unknown}")")",
  "compiler_c_version": "$(json_escape "$(compiler_version "${c_compiler:-}")")",
  "compiler_cxx_version": "$(json_escape "$(compiler_version "${cxx_compiler:-}")")",
  "cmake_build_type": "${build_type:-unknown}",
  "cmake_preset": "${DENSECORE_CMAKE_PRESET:-portable-release}",
  "cmake_cache_sha256": "$cache_sha256",
  "cpu_arch": "$(uname -m)",
  "densecore_build_mode": "portable-release",
  "densecore_portable": true,
  "ggml_native": false,
  "smoke_model_sha256": "$model_hash"
}
EOF

binary_tar="$out_dir/densecore-${version}-linux-amd64.tar.gz"
tar -czf "$binary_tar" -C "$binary_root" .

tar -tzf "$binary_tar" > "$tmp_dir/binary-files.txt"
grep -Eq '(^|/)licenses/THIRD_PARTY_NOTICES\.md$' "$tmp_dir/binary-files.txt" || {
  echo "binary archive is missing collected third-party notices" >&2
  exit 1
}
grep -Eq '(^|/)provenance\.json$' "$tmp_dir/binary-files.txt" || {
  echo "binary archive is missing provenance metadata" >&2
  exit 1
}
grep -Eq '(^|/)README\.md$' "$tmp_dir/binary-files.txt" || {
  echo "binary archive is missing README.md" >&2
  exit 1
}
grep -Eq '(^|/)server/openapi\.yaml$' "$tmp_dir/binary-files.txt" || {
  echo "binary archive is missing OpenAPI spec" >&2
  exit 1
}

(cd "$out_dir" && sha256sum "$(basename "$binary_tar")") > "$binary_tar.sha256"
