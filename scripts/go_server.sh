#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
	echo "usage: $0 <go args...>" >&2
	exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
server_dir="$repo_root/server"
densecloud_dir="${DENSECLOUD_DIR:-$repo_root/../DenseCloud}"

# DenseSeries modules are private by default.
export GOPRIVATE="${GOPRIVATE:-github.com/DenseAI/*,github.com/DenseCore/*,github.com/denseseries/*}"
export GONOSUMDB="${GONOSUMDB:-github.com/DenseAI/*,github.com/DenseCore/*,github.com/denseseries/*}"

export GOCACHE="${GOCACHE:-${TMPDIR:-/tmp}/densecore-go-build-cache}"
export GOMODCACHE="${GOMODCACHE:-${TMPDIR:-/tmp}/densecore-go-mod-cache}"

mkdir -p "$GOCACHE" "$GOMODCACHE"

workfile=""
cleanup() {
	if [[ -n "$workfile" && -f "$workfile" ]]; then
		rm -f "$workfile"
	fi
}
trap cleanup EXIT

if [[ -f "$densecloud_dir/go.mod" ]]; then
	densecloud_dir="$(cd "$densecloud_dir" && pwd)"
	workfile="$(mktemp "${TMPDIR:-/tmp}/densecore-go-work.XXXXXX")"
	cat >"$workfile" <<EOF
go 1.24.0

use (
	$server_dir
	$densecloud_dir
)
EOF
fi

if [[ -n "$workfile" ]]; then
	(cd "$server_dir" && GOWORK="$workfile" go "$@")
else
	(cd "$server_dir" && go "$@")
fi
