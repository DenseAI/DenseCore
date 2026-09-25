#!/bin/sh
set -eu

target=${1:-}
if [ -z "$target" ]; then
    echo "usage: $0 <portable|amd64-v3|arm64>" >&2
    exit 2
fi

case "$target" in
    portable)
        cat <<'EOF'
cpu_target=portable
tag_suffix=
cmake_args=-DDENSECORE_CPU_TARGET=portable -DDENSECORE_PORTABLE=ON -DGGML_NATIVE=OFF
docker_platforms=linux/amd64,linux/arm64
EOF
        ;;
    amd64-v3)
        cat <<'EOF'
cpu_target=amd64-v3
tag_suffix=-amd64-v3
cmake_args=-DDENSECORE_CPU_TARGET=amd64-v3 -DGGML_NATIVE=OFF
docker_platforms=linux/amd64
EOF
        ;;
    arm64)
        cat <<'EOF'
cpu_target=arm64
tag_suffix=-arm64
cmake_args=-DDENSECORE_CPU_TARGET=arm64 -DDENSECORE_ARM_TARGET=generic -DDENSECORE_ARM_CORRECTNESS_FIRST=ON -DGGML_NATIVE=OFF
docker_platforms=linux/arm64
EOF
        ;;
    *)
        echo "invalid DenseCore release CPU target: $target" >&2
        echo "expected one of: portable, amd64-v3, arm64" >&2
        exit 2
        ;;
esac
