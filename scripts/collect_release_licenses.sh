#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
    echo "usage: $0 <output-dir>" >&2
    exit 1
fi

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
OUT_DIR=$1

mkdir -p "$OUT_DIR"
cp "$ROOT_DIR/LICENSE" "$OUT_DIR/LICENSE"
cp "$ROOT_DIR/NOTICE" "$OUT_DIR/NOTICE"
cp "$ROOT_DIR/THIRD_PARTY_NOTICES.md" "$OUT_DIR/THIRD_PARTY_NOTICES.md"

mkdir -p "$OUT_DIR/third_party/ggml"
cp "$ROOT_DIR/core/third_party/ggml/LICENSE" "$OUT_DIR/third_party/ggml/LICENSE"

BUILD_DIR=${DENSECORE_BUILD_DIR:-$ROOT_DIR/build}
for component in simde highway mimalloc spdlog; do
    source_dir="$BUILD_DIR/_deps/${component}-src"
    if [ ! -d "$source_dir" ]; then
        echo "missing fetched dependency source for $component: $source_dir" >&2
        echo "build DenseCore first or set DENSECORE_BUILD_DIR" >&2
        exit 1
    fi

    target="$OUT_DIR/third_party/cpp/$component"
    mkdir -p "$target"
    found=0
    for name in LICENSE LICENSE.txt LICENSE.md LICENSE-BSD3 COPYING COPYING.txt NOTICE NOTICE.txt; do
        if [ -f "$source_dir/$name" ]; then
            cp "$source_dir/$name" "$target/$name"
            found=1
        fi
    done
    if [ "$found" -eq 0 ]; then
        echo "no license file found for fetched dependency: $component" >&2
        exit 1
    fi
done

if command -v go >/dev/null 2>&1 && [ -f "$ROOT_DIR/server/go.mod" ]; then
    GO_PACKAGE=${DENSECORE_GO_PACKAGE:-./cmd/densecore}
    GO_LIST_OUT="$OUT_DIR/go-dependencies.txt"
    # Remove the legacy shipped inventory containing local module-cache paths.
    rm -f "$OUT_DIR/go-dependencies.raw"
    GO_LIST_RAW=$(mktemp "${TMPDIR:-/tmp}/densecore-go-licenses.XXXXXX")
    trap 'rm -f "$GO_LIST_RAW"' EXIT HUP INT TERM
    (
        cd "$ROOT_DIR/server"
        go list -mod=mod -deps -f '{{with .Module}}{{if not .Main}}{{.Path}}|{{.Version}}|{{.Dir}}{{end}}{{end}}' "$GO_PACKAGE"
    ) > "$GO_LIST_RAW"
    sort -u "$GO_LIST_RAW" -o "$GO_LIST_RAW"
    cut -d '|' -f 1,2 "$GO_LIST_RAW" > "$GO_LIST_OUT"
    while IFS='|' read -r module version dir; do
        [ -n "$module" ] || continue
        if [ -z "$dir" ] || [ ! -d "$dir" ]; then
            echo "missing module source directory for $module" >&2
            exit 1
        fi

        target="$OUT_DIR/third_party/go/$module"
        copied=0
        for name in LICENSE LICENSE.txt LICENSE.md COPYING COPYING.txt NOTICE NOTICE.txt; do
            if [ -f "$dir/$name" ]; then
                mkdir -p "$target"
                cp -f "$dir/$name" "$target/$name"
                copied=1
            fi
        done

        if [ "$copied" -eq 0 ]; then
            echo "no license or notice file found for Go module: $module" >&2
            exit 1
        fi
    done < "$GO_LIST_RAW"
fi
