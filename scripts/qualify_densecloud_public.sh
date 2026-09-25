#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/qualify_densecloud_public.sh [options]

Qualifies DenseCore against the public DenseCloud v1.1.0 Go module and
anonymous dense-base 1.1.0 OCI chart from clean caches and isolated config.

Options:
  --output-dir DIR           Persist logs and metadata in DIR.
  --run-kubernetes-smoke     Run the optional Kubernetes/API smoke.
  --helm-values PATH         Values file to use for the optional Kubernetes smoke.
  --release-name NAME        Helm release name for the optional Kubernetes smoke.
  --namespace NAME           Namespace for the optional Kubernetes smoke.
  --chat-model ID            Model id to send in chat smoke requests. Default: densecore-v1
  --timeout SECONDS          Timeout for rollout/smoke waits. Default: 600
  --help                     Show this help text.
EOF
}

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)

densecloud_module="github.com/DenseAI/DenseCloud"
densecloud_tag="v1.1.0"
densebase_repo="oci://ghcr.io/denseai/charts"
densebase_chart="dense-base"
densebase_version="1.1.0"
default_values="charts/densecore/examples/values-cpu-inference.yaml"

output_dir=""
run_kubernetes_smoke=0
helm_values=""
release_name="densecore-qualify"
namespace="densecore-qualify"
chat_model="densecore-v1"
timeout_seconds=600

while [ $# -gt 0 ]; do
  case "$1" in
    --output-dir)
      output_dir=$2
      shift 2
      ;;
    --run-kubernetes-smoke)
      run_kubernetes_smoke=1
      shift
      ;;
    --helm-values)
      helm_values=$2
      shift 2
      ;;
    --release-name)
      release_name=$2
      shift 2
      ;;
    --namespace)
      namespace=$2
      shift 2
      ;;
    --chat-model)
      chat_model=$2
      shift 2
      ;;
    --timeout)
      timeout_seconds=$2
      shift 2
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

fail() {
  echo "qualify_densecloud_public: $*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
}

resolve_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$repo_dir" "$1" ;;
  esac
}

wait_for_http() {
  local url=$1
  local attempts=$2
  local i
  for i in $(seq 1 "$attempts"); do
    if curl -fsS "$url" >/dev/null; then
      return 0
    fi
    sleep 1
  done
  return 1
}

require_cmd tar
require_cmd git
require_cmd go
require_cmd helm
require_cmd cmake
require_cmd curl
require_cmd sha256sum

tmp_root=$(mktemp -d "${TMPDIR:-/tmp}/densecore-public-qualify.XXXXXX")
repo_copy="$tmp_root/repo"
artifact_dir="$tmp_root/artifacts"
logs_dir="$artifact_dir/logs"
result_dir="$artifact_dir/result"
mkdir -p "$repo_copy" "$logs_dir" "$result_dir"

cleanup() {
  if [ -n "${port_forward_pid:-}" ] && kill -0 "$port_forward_pid" 2>/dev/null; then
    kill "$port_forward_pid" 2>/dev/null || true
    wait "$port_forward_pid" 2>/dev/null || true
  fi
  if [ -n "${created_namespace:-}" ] && [ "$created_namespace" = "1" ]; then
    kubectl delete namespace "$namespace" --wait=false >/dev/null 2>&1 || true
  fi
  if [ -n "$output_dir" ]; then
    mkdir -p "$output_dir"
    cp -a "$artifact_dir"/. "$output_dir"/
  fi
  chmod -R u+w "$tmp_root" 2>/dev/null || true
  rm -rf "$tmp_root" 2>/dev/null || true
}
trap cleanup EXIT

if [ -f "$repo_dir/go.work" ] || [ -f "$repo_dir/go.work.sum" ]; then
  fail "go.work/go.work.sum is present; public qualification requires GOWORK=off without a workspace file"
fi
if find "$repo_dir/charts/densecore/charts" -maxdepth 1 -name 'dense-base-*.tgz' -print -quit 2>/dev/null | grep -q .; then
  fail "pre-vendored dense-base archive detected under charts/densecore/charts"
fi

file_list="$tmp_root/file-list.txt"
(
  cd "$repo_dir"
  git ls-files -z --cached --modified --others --exclude-standard
) | while IFS= read -r -d '' path; do
  if [ -e "$repo_dir/$path" ]; then
    printf '%s\0' "$path"
  fi
done >"$file_list"

(
  cd "$repo_dir"
  tar --warning=no-file-changed --warning=no-timestamp --null -T "$file_list" -cf -
) | tar -xf - -C "$repo_copy"

mkdir -p "$repo_copy/charts/densecore/charts"
rm -rf "$repo_copy/charts/densecore/charts"
mkdir -p "$repo_copy/charts/densecore/charts"

export HOME="$tmp_root/home"
export GOPATH="$tmp_root/gopath"
export GOMODCACHE="$tmp_root/gomodcache"
export GOCACHE="$tmp_root/gocache"
export HELM_CONFIG_HOME="$tmp_root/helm/config"
export HELM_CACHE_HOME="$tmp_root/helm/cache"
export HELM_DATA_HOME="$tmp_root/helm/data"
export GOFLAGS=-mod=mod
export GOWORK=off
export GOPROXY=https://proxy.golang.org,direct
mkdir -p "$HOME" "$GOPATH" "$GOMODCACHE" "$GOCACHE" "$HELM_CONFIG_HOME" "$HELM_CACHE_HOME" "$HELM_DATA_HOME"

resolved_gowork=$(go env GOWORK)
if [ "$resolved_gowork" != "off" ] && [ -n "$resolved_gowork" ]; then
  fail "expected go env GOWORK to resolve to off, got: $resolved_gowork"
fi

if grep -Eq '^[[:space:]]*replace([[:space:]]|\()' "$repo_copy/server/go.mod"; then
  fail "server/go.mod contains a replace directive"
fi
if grep -Eq '^[[:space:]]*repository:[[:space:]]*file://' "$repo_copy/charts/densecore/Chart.yaml"; then
  fail "charts/densecore/Chart.yaml uses a file:// dependency"
fi
if ! grep -Eq "^[[:space:]]*version:[[:space:]]*$densebase_version$" "$repo_copy/charts/densecore/Chart.yaml"; then
  fail "charts/densecore/Chart.yaml does not pin ${densebase_chart} ${densebase_version}"
fi
if ! grep -Eq "^[[:space:]]*repository:[[:space:]]*$densebase_repo$" "$repo_copy/charts/densecore/Chart.yaml"; then
  fail "charts/densecore/Chart.yaml does not pin ${densebase_repo}"
fi

densecore_commit=$(git -C "$repo_dir" rev-parse HEAD)
go_version=$(go version | awk '{print $3}')
helm_version=$(helm version --template '{{ .Version }}')

go_list_json="$result_dir/densecloud-module.json"
(
  cd "$repo_copy/server"
  go list -m -json "${densecloud_module}@${densecloud_tag}"
) | tee "$go_list_json" >/dev/null

module_version=$(sed -n 's/^[[:space:]]*"Version": "\(.*\)",$/\1/p' "$go_list_json" | head -n 1)
module_hash=$(sed -n 's/^[[:space:]]*"Hash": "\(.*\)",$/\1/p' "$go_list_json" | head -n 1)
if [ "$module_version" != "$densecloud_tag" ]; then
  fail "public module resolved to ${module_version:-unknown}, expected ${densecloud_tag}"
fi

(
  cd "$repo_copy/server"
  go mod download >"$logs_dir/go-mod-download.log" 2>&1
)

cmake -S "$repo_copy/core" -B "$repo_copy/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DDENSECORE_BUILD_TESTS=OFF \
  >"$logs_dir/cmake-configure.log" 2>&1
cmake --build "$repo_copy/build" -j"$(nproc)" >"$logs_dir/cmake-build.log" 2>&1

(
  cd "$repo_copy/server"
  export CGO_ENABLED=1
  export CGO_CFLAGS="-I$repo_copy/core/include"
  export CGO_LDFLAGS="-L$repo_copy/build -ldensecore -lstdc++ -ldl"
  export LD_LIBRARY_PATH="$repo_copy/build:${LD_LIBRARY_PATH:-}"
  go test ./... >"$logs_dir/go-test.log" 2>&1
)

helm_show_log="$result_dir/helm-show-chart.txt"
helm show chart "${densebase_repo}/${densebase_chart}" --version "$densebase_version" \
  >"$helm_show_log" 2>"$logs_dir/helm-show-chart.stderr"

helm_pull_log="$logs_dir/helm-pull.log"
helm pull "${densebase_repo}/${densebase_chart}" --version "$densebase_version" --destination "$result_dir" \
  >"$helm_pull_log" 2>&1

chart_archive="$result_dir/${densebase_chart}-${densebase_version}.tgz"
[ -f "$chart_archive" ] || fail "expected pulled chart archive at $chart_archive"

chart_sha256=$(sha256sum "$chart_archive" | awk '{print $1}')
chart_digest=$(sed -n 's/^Digest: //p' "$helm_pull_log" | head -n 1)
resolved_chart_version=$(sed -n 's/^version: //p' "$helm_show_log" | head -n 1)
resolved_chart_name=$(sed -n 's/^name: //p' "$helm_show_log" | head -n 1)
if [ "$resolved_chart_name" != "$densebase_chart" ] || [ "$resolved_chart_version" != "$densebase_version" ]; then
  fail "public Helm chart resolved to ${resolved_chart_name:-unknown} ${resolved_chart_version:-unknown}"
fi

(
  cd "$repo_copy"
  helm dependency update charts/densecore >"$logs_dir/helm-dependency-update.log" 2>&1
  helm lint charts/densecore >"$logs_dir/helm-lint.log" 2>&1
  helm template densecore charts/densecore -f "$default_values" >"$result_dir/helm-template.yaml" 2>"$logs_dir/helm-template.stderr"
)

if ! grep -Eq "^[[:space:]]*repository:[[:space:]]*$densebase_repo$" "$repo_copy/charts/densecore/Chart.lock"; then
  fail "Chart.lock did not resolve the canonical OCI repository"
fi
if ! grep -Eq "^[[:space:]]*version:[[:space:]]*$densebase_version$" "$repo_copy/charts/densecore/Chart.lock"; then
  fail "Chart.lock did not resolve ${densebase_chart} ${densebase_version}"
fi

smoke_status="NOT RUN"
smoke_notes="Kubernetes/API smoke not requested"
image_ref="NOT RUN"
image_digest="NOT RUN"
model_path_value="NOT RUN"
model_sha256="NOT RUN"

if [ "$run_kubernetes_smoke" = "1" ]; then
  require_cmd kubectl
  smoke_values=${helm_values:-$default_values}
  smoke_values=$(resolve_path "$smoke_values")
  [ -f "$smoke_values" ] || fail "Helm values file not found: $smoke_values"
  kubectl cluster-info >"$logs_dir/kubectl-cluster-info.log" 2>&1

  created_namespace=0
  if ! kubectl get namespace "$namespace" >/dev/null 2>&1; then
    kubectl create namespace "$namespace" >"$logs_dir/kubectl-create-namespace.log" 2>&1
    created_namespace=1
  fi

  (
    cd "$repo_copy"
    helm upgrade --install "$release_name" charts/densecore \
      --namespace "$namespace" \
      -f "$smoke_values" \
      >"$logs_dir/helm-upgrade-install.log" 2>&1
  )

  deployment_name=$(kubectl get deployment -n "$namespace" -l "app.kubernetes.io/instance=$release_name" -o jsonpath='{.items[0].metadata.name}')
  [ -n "$deployment_name" ] || fail "failed to resolve DenseCore deployment name in namespace $namespace"

  service_name=$(
    kubectl get service -n "$namespace" -l "app.kubernetes.io/instance=$release_name" \
      -o jsonpath='{range .items[*]}{.metadata.name}{" "}{range .spec.ports[*]}{.port}{" "}{end}{"\n"}{end}' \
      | awk '$0 ~ /(^| )8080( |$)/ { print $1; exit }'
  )
  [ -n "$service_name" ] || fail "failed to resolve DenseCore HTTP service name in namespace $namespace"

  kubectl rollout status "deployment/$deployment_name" -n "$namespace" --timeout="${timeout_seconds}s" \
    >"$logs_dir/kubectl-rollout-status.log" 2>&1

  pods_json="$result_dir/kubectl-pods.json"
  kubectl get pods -n "$namespace" -l "app.kubernetes.io/instance=$release_name" -o json >"$pods_json"
  if grep -Eq '"restartCount":[1-9]' "$pods_json"; then
    fail "pod restart count was non-zero during Kubernetes smoke"
  fi

  pod_name=$(kubectl get pods -n "$namespace" -l "app.kubernetes.io/instance=$release_name" -o jsonpath='{.items[0].metadata.name}')
  [ -n "$pod_name" ] || fail "failed to resolve DenseCore pod name in namespace $namespace"
  image_ref=$(kubectl get deployment "$deployment_name" -n "$namespace" -o jsonpath='{.spec.template.spec.containers[0].image}')
  image_digest=$(kubectl get pod "$pod_name" -n "$namespace" -o jsonpath='{.status.containerStatuses[0].imageID}')
  model_path_value=$(
    kubectl get deployment "$deployment_name" -n "$namespace" \
      -o jsonpath='{range .spec.template.spec.containers[0].env[*]}{.name}={.value}{"\n"}{end}' \
      | sed -n 's/^MAIN_MODEL_PATH=//p' \
      | head -n 1
  )
  [ -n "$model_path_value" ] || fail "MAIN_MODEL_PATH is not set on the deployed DenseCore container"
  model_sha256=$(
    kubectl exec -n "$namespace" "$pod_name" -- sha256sum "$model_path_value" \
      | awk 'NR==1 { print $1 }'
  )
  [ -n "$model_sha256" ] || fail "failed to compute model SHA-256 from the running DenseCore pod"

  port_forward_log="$logs_dir/kubectl-port-forward.log"
  kubectl port-forward -n "$namespace" "service/$service_name" 18080:8080 >"$port_forward_log" 2>&1 &
  port_forward_pid=$!
  if ! wait_for_http "http://127.0.0.1:18080/health/startup" "$timeout_seconds"; then
    fail "timed out waiting for /health/startup on port-forwarded DenseCore service"
  fi
  curl -fsS "http://127.0.0.1:18080/health/live" >"$result_dir/health-live.json"
  curl -fsS "http://127.0.0.1:18080/health/ready" >"$result_dir/health-ready.json"
  curl -fsS "http://127.0.0.1:18080/health/startup" >"$result_dir/health-startup.json"
  curl -fsS "http://127.0.0.1:18080/v1/models" >"$result_dir/models.json"
  curl -fsS "http://127.0.0.1:18080/metrics" >"$result_dir/metrics.txt"

  if ! grep -q 'densecloud_http_requests_total' "$result_dir/metrics.txt"; then
    fail "expected DenseCloud HTTP metrics in /metrics output"
  fi
  if ! grep -q 'densecore_pending_requests' "$result_dir/metrics.txt"; then
    fail "expected DenseCore metrics in /metrics output"
  fi

  curl -fsS -H 'Content-Type: application/json' \
    -d "{\"model\":\"$chat_model\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one short sentence.\"}],\"max_tokens\":16}" \
    "http://127.0.0.1:18080/v1/chat/completions" \
    >"$result_dir/chat-nonstream.json"
  grep -q '"choices"' "$result_dir/chat-nonstream.json" || fail "non-streaming chat smoke did not return choices"

  curl -fsS -H 'Content-Type: application/json' \
    -d "{\"model\":\"$chat_model\",\"messages\":[{\"role\":\"user\",\"content\":\"Count to two.\"}],\"max_tokens\":16,\"stream\":true}" \
    "http://127.0.0.1:18080/v1/chat/completions" \
    >"$result_dir/chat-stream.sse"
  grep -q 'data:' "$result_dir/chat-stream.sse" || fail "streaming chat smoke did not emit SSE data"
  grep -q '\[DONE\]' "$result_dir/chat-stream.sse" || fail "streaming chat smoke did not terminate with [DONE]"

  smoke_status="PASS"
  smoke_notes="Helm install, health endpoints, metrics, and chat smoke passed"
fi

cat >"$result_dir/qualification-metadata.txt" <<EOF
densecore_commit=$densecore_commit
densecloud_tag=$densecloud_tag
densecloud_module=$densecloud_module
densecloud_module_version=$module_version
densecloud_module_origin_hash=$module_hash
densebase_repository=$densebase_repo
densebase_chart=$densebase_chart
densebase_chart_version=$densebase_version
densebase_chart_digest=${chart_digest:-unknown}
densebase_chart_sha256=$chart_sha256
go_version=$go_version
helm_version=$helm_version
chart_lock_digest=$(sed -n 's/^digest: //p' "$repo_copy/charts/densecore/Chart.lock" | head -n 1)
release_name=$release_name
namespace=$namespace
image_ref=$image_ref
image_digest=$image_digest
model_path=$model_path_value
model_sha256=$model_sha256
kubernetes_smoke=$smoke_status
kubernetes_smoke_notes=$smoke_notes
EOF

printf 'qualification artifacts: %s\n' "${output_dir:-$artifact_dir}"
