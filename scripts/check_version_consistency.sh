#!/usr/bin/env bash
set -euo pipefail

release_version="${1:-0.1.0}"
root_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

expected=$release_version
if [[ "$expected" =~ ^([0-9]+)\.([0-9]+)-rc\.[0-9]+$ ]]; then
  expected="${BASH_REMATCH[1]}.${BASH_REMATCH[2]}.0"
elif [[ "$expected" =~ ^([0-9]+\.[0-9]+\.[0-9]+)-rc\.[0-9]+$ ]]; then
  expected="${BASH_REMATCH[1]}"
fi

check() {
  local label=$1
  local pattern=$2
  local file=$3

  if ! grep -Eq "$pattern" "$root_dir/$file"; then
    echo "version mismatch: ${label} in ${file}; expected ${expected}" >&2
    exit 1
  fi
}

escaped=${expected//./\\.}

check "CMake project" "project\\(DenseCore VERSION ${escaped} " "core/CMakeLists.txt"
check "C API header" "@version ${escaped}$" "core/include/densecore.h"
check "Python project" "^version = \"${escaped}\"$" "python/pyproject.toml"
check "Python setup.py" "version=\"${escaped}\"" "python/setup.py"
check "Python runtime package" "^__version__ = \"${escaped}\"$" "python/densecore/__init__.py"
if ! awk -v expected="version = \"${expected}\"" '
  $0 == "name = \"densecore\"" { getline; if ($0 == expected) found = 1 }
  END { exit(found ? 0 : 1) }
' "$root_dir/python/uv.lock"; then
  echo "version mismatch: Python lockfile in python/uv.lock; expected ${expected}" >&2
  exit 1
fi
check "Go buildinfo" "var Version = \"${escaped}\"" "server/internal/buildinfo/buildinfo.go"
check "OpenAPI document" "^  version: ${escaped}$" "server/openapi.yaml"
check "Helm chart version" "^version: ${escaped}$" "charts/densecore/Chart.yaml"
check "Helm appVersion" "^appVersion: \"${escaped}\"$" "charts/densecore/Chart.yaml"
check "Helm image tag" "tag: \"${escaped}\"" "charts/densecore/values.yaml"
check "Docker Compose image" "image: denseai/densecore:${escaped}" "docker-compose.yml"
check "Downloader image label" "org.opencontainers.image.version=\"${escaped}\"" "docker/downloader/Dockerfile"

for values_file in "$root_dir"/charts/densecore/examples/values-*.yaml; do
  if grep -Eq 'kubernetes\.io/arch:[[:space:]]*arm64' "$values_file"; then
    if grep -Eq 'repository:[[:space:]]*denseai/densecore' "$values_file"; then
      echo "unqualified Arm example points at the amd64-only public image: ${values_file#$root_dir/}" >&2
      exit 1
    fi
    continue
  fi
  if ! grep -Eq "tag: \"${escaped}\"" "$values_file"; then
    echo "version mismatch: Helm example image tag in ${values_file#$root_dir/}; expected ${expected}" >&2
    exit 1
  fi
done

echo "DenseCore release ${release_version} is consistent with package metadata ${expected}"
