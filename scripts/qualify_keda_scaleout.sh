#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/qualify_keda_scaleout.sh <setup|control|autoscale|collect|cleanup|all> [options]

Runs a DenseCore KEDA scale-out qualification harness with explicit setup,
fixed-replica control, autoscaled run, telemetry collection, and owned-namespace
cleanup. The harness fails closed when the selected PromQL is masked or when
the target cluster cannot satisfy the requested preconditions.

Commands:
  setup        Install pinned KEDA and Prometheus releases and verify CRDs/pods.
  control      Deploy DenseCore with a fixed single replica and run the load lane.
  autoscale    Deploy DenseCore with KEDA min=1/max=2 and run the same load lane.
  collect      Collect 5s telemetry from an already deployed release.
  cleanup      Remove the owned DenseCore qualification release and namespace only.
  all          Run setup, control, autoscale, collect summary, and cleanup.

Required options for control/autoscale/all:
  --helm-values PATH        Base DenseCore Helm values file containing image,
                            model, resources, and product runtime settings.
  --metric-query PROMQL     Real product-owned PromQL query. Example:
                            sum(densecore_pending_requests{namespace="densecore-keda"})
  --metric-threshold NUM    Threshold used for the KEDA trigger.

General options:
  --output-dir DIR          Persist logs and artifacts in DIR.
  --release-name NAME       DenseCore Helm release name. Default: densecore-keda
  --namespace NAME          DenseCore namespace. Default: densecore-keda
  --keda-release NAME       KEDA Helm release name. Default: <release>-keda
  --prom-release NAME       Prometheus Helm release name. Default: <release>-prom
  --chat-model ID           Model id for /v1/chat/completions. Default: densecore-v1
  --keda-chart-version VER  KEDA Helm chart version. Default: 2.18.0
  --prom-chart-version VER  kube-prometheus-stack chart version. Default: 78.5.0
  --prometheus-service NAME Prometheus service name override. Default:
                            auto-discover by release labels
  --keda-namespace NAME     KEDA namespace. Default: <namespace>-keda
  --monitoring-namespace N  Monitoring namespace. Default: <namespace>-monitoring
  --poll-interval SEC       Telemetry/KEDA poll interval. Default: 15
  --cooldown-period SEC     KEDA cooldown period. Default: 180
  --timeout SEC             Rollout and endpoint timeout. Default: 900
  --request-count N         Total chat requests per run. Default: 8
  --stream-count N          Streaming requests per run. Default: 2
  --concurrency N           Concurrent client workers. Default: 4
  --max-output-tokens N     Max completion tokens. Default: 64
  --collect-seconds N       Extra post-load collection window. Default: 180
  --image-digest DIGEST     Required for real control/autoscale runs.
  --model-name NAME         Required for real control/autoscale runs.
  --model-quantization Q    Required for real control/autoscale runs.
  --model-sha256 SHA256     Required for real control/autoscale runs.
  --context-length N        Required for real control/autoscale runs.
  --threads N               Real-run override; otherwise harvested from values.
  --densecore-max-num-seqs N
                            Real-run override; otherwise harvested from values.
  --densecore-max-prefill-seqs N
                            Real-run override; otherwise harvested from values.
  --cleanup-observability   Also remove owned KEDA and monitoring releases and
                            namespaces after exact ownership verification.
  --dry-run                 Validate arguments, paths, and generated files only.
  --help                    Show this help text.
EOF
}

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)

command_name=${1:-}
if [ -z "$command_name" ]; then
  usage >&2
  exit 1
fi
if [ "$command_name" = "--help" ] || [ "$command_name" = "-h" ]; then
  usage
  exit 0
fi
shift || true

release_name="densecore-keda"
namespace="densecore-keda"
keda_release_name=""
prom_release_name=""
chat_model="densecore-v1"
keda_namespace=""
monitoring_namespace=""
prometheus_service=""
keda_chart_version="2.18.0"
prom_chart_version="78.5.0"
poll_interval=15
cooldown_period=180
timeout_seconds=900
request_count=8
stream_count=2
concurrency=4
max_output_tokens=64
collect_seconds=180
helm_values=""
metric_query=""
metric_threshold=""
metric_name="densecore_pending_requests"
per_pod_proof_metric="densecore_total_requests"
output_dir=""
dry_run=0
image_digest=""
model_name=""
model_quantization=""
model_sha256=""
context_length=""
threads_override=""
max_num_seqs_override=""
max_prefill_seqs_override=""
cleanup_observability=0
tmp_root=""
repo_copy=""

ownership_label_key="denseai.com/qualify-keda-owned"
ownership_owner_key="denseai.com/qualify-keda-owner-id"
ownership_release_key="denseai.com/qualify-keda-release"
ownership_role_key="denseai.com/qualify-keda-role"
ownership_id="${USER:-unknown}@$(hostname -s 2>/dev/null || printf unknown)"

fail() {
  echo "qualify_keda_scaleout: $*" >&2
  exit 1
}

log() {
  printf '[qualify_keda_scaleout] %s\n' "$*" >&2
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

run_logged() {
  local logfile=$1
  shift
  if [ "$dry_run" -eq 1 ]; then
    log "dry-run: $* > $logfile"
    return 0
  fi
  "$@" >"$logfile" 2>&1
}

json_escape() {
  jq -Rn --arg value "$1" '$value'
}

now_iso() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

now_ms() {
  date +%s%3N
}

ensure_output_dir() {
  if [ -z "$output_dir" ]; then
    output_dir="$repo_dir/release/keda-scaleout/$(date -u +%Y%m%dT%H%M%SZ)"
  fi
  mkdir -p "$output_dir"
  logs_dir="$output_dir/logs"
  results_dir="$output_dir/results"
  runtime_dir="$output_dir/runtime"
  mkdir -p "$logs_dir" "$results_dir" "$runtime_dir"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --output-dir)
      output_dir=$2
      shift 2
      ;;
    --helm-values)
      helm_values=$2
      shift 2
      ;;
    --metric-query)
      metric_query=$2
      shift 2
      ;;
    --metric-threshold)
      metric_threshold=$2
      shift 2
      ;;
    --metric-name)
      metric_name=$2
      shift 2
      ;;
    --per-pod-proof-metric)
      per_pod_proof_metric=$2
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
    --keda-release)
      keda_release_name=$2
      shift 2
      ;;
    --prom-release)
      prom_release_name=$2
      shift 2
      ;;
    --chat-model)
      chat_model=$2
      shift 2
      ;;
    --keda-chart-version)
      keda_chart_version=$2
      shift 2
      ;;
    --prom-chart-version)
      prom_chart_version=$2
      shift 2
      ;;
    --prometheus-service)
      prometheus_service=$2
      shift 2
      ;;
    --keda-namespace)
      keda_namespace=$2
      shift 2
      ;;
    --monitoring-namespace)
      monitoring_namespace=$2
      shift 2
      ;;
    --poll-interval)
      poll_interval=$2
      shift 2
      ;;
    --cooldown-period)
      cooldown_period=$2
      shift 2
      ;;
    --timeout)
      timeout_seconds=$2
      shift 2
      ;;
    --request-count)
      request_count=$2
      shift 2
      ;;
    --stream-count)
      stream_count=$2
      shift 2
      ;;
    --concurrency)
      concurrency=$2
      shift 2
      ;;
    --max-output-tokens)
      max_output_tokens=$2
      shift 2
      ;;
    --collect-seconds)
      collect_seconds=$2
      shift 2
      ;;
    --image-digest)
      image_digest=$2
      shift 2
      ;;
    --model-name)
      model_name=$2
      shift 2
      ;;
    --model-quantization)
      model_quantization=$2
      shift 2
      ;;
    --model-sha256)
      model_sha256=$2
      shift 2
      ;;
    --context-length)
      context_length=$2
      shift 2
      ;;
    --threads)
      threads_override=$2
      shift 2
      ;;
    --densecore-max-num-seqs)
      max_num_seqs_override=$2
      shift 2
      ;;
    --densecore-max-prefill-seqs)
      max_prefill_seqs_override=$2
      shift 2
      ;;
    --cleanup-observability)
      cleanup_observability=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      fail "unknown argument: $1"
      ;;
  esac
done

case "$command_name" in
  setup|control|autoscale|collect|cleanup|all) ;;
  *)
    fail "unknown command: $command_name"
    ;;
esac

if [ -n "$helm_values" ]; then
  helm_values=$(resolve_path "$helm_values")
fi
if [ -z "$keda_release_name" ]; then
  keda_release_name="${release_name}-keda"
fi
if [ -z "$prom_release_name" ]; then
  prom_release_name="${release_name}-prom"
fi
if [ -z "$keda_namespace" ]; then
  keda_namespace="${namespace}-keda"
fi
if [ -z "$monitoring_namespace" ]; then
  monitoring_namespace="${namespace}-monitoring"
fi

validate_args() {
  if printf '%s' "$metric_query" | grep -Eq '(^|[[:space:]])or[[:space:]]+vector\(0\)'; then
    fail "metric query must not hide scrape failures with 'or vector(0)'"
  fi
  case "$command_name" in
    control|autoscale|all)
      [ -n "$helm_values" ] || fail "--helm-values is required for $command_name"
      [ -f "$helm_values" ] || fail "helm values file not found: $helm_values"
      [ -n "$metric_query" ] || fail "--metric-query is required for $command_name"
      [ -n "$metric_threshold" ] || fail "--metric-threshold is required for $command_name"
      ;;
    collect)
      [ -n "$metric_query" ] || fail "--metric-query is required for collect"
      ;;
  esac
  [ "$stream_count" -le "$request_count" ] || fail "--stream-count cannot exceed --request-count"
  [ "$namespace" != "$keda_namespace" ] || fail "--namespace and --keda-namespace must differ"
  [ "$namespace" != "$monitoring_namespace" ] || fail "--namespace and --monitoring-namespace must differ"
  [ "$keda_namespace" != "$monitoring_namespace" ] || fail "--keda-namespace and --monitoring-namespace must differ"
  [ "$release_name" != "$keda_release_name" ] || fail "--release-name and --keda-release must differ"
  [ "$release_name" != "$prom_release_name" ] || fail "--release-name and --prom-release must differ"
}

require_base_cmds() {
  require_cmd git
  require_cmd helm
  require_cmd kubectl
  require_cmd curl
  require_cmd jq
  require_cmd sha256sum
}

stop_service_port_forward() {
  if [ -n "${service_port_forward_pid:-}" ] && kill -0 "$service_port_forward_pid" 2>/dev/null; then
    kill "$service_port_forward_pid" 2>/dev/null || true
    wait "$service_port_forward_pid" 2>/dev/null || true
  fi
  service_port_forward_pid=""
}

stop_prometheus_port_forward() {
  if [ -n "${prom_port_forward_pid:-}" ] && kill -0 "$prom_port_forward_pid" 2>/dev/null; then
    kill "$prom_port_forward_pid" 2>/dev/null || true
    wait "$prom_port_forward_pid" 2>/dev/null || true
  fi
  prom_port_forward_pid=""
}

cleanup_tmp() {
  stop_service_port_forward
  stop_prometheus_port_forward
  if [ -n "${collector_pid:-}" ] && kill -0 "$collector_pid" 2>/dev/null; then
    kill "$collector_pid" 2>/dev/null || true
    wait "$collector_pid" 2>/dev/null || true
  fi
  if [ -n "$tmp_root" ] && [ -d "$tmp_root" ]; then
    rm -rf "$tmp_root"
  fi
}
trap cleanup_tmp EXIT

ensure_repo_copy() {
  if [ -n "$repo_copy" ] && [ -d "$repo_copy" ]; then
    return 0
  fi
  tmp_root=$(mktemp -d "${TMPDIR:-/tmp}/densecore-keda-qualify.XXXXXX")
  repo_copy="$tmp_root/repo"
  mkdir -p "$repo_copy/charts"
  cp -a "$repo_dir/charts/densecore" "$repo_copy/charts/"
  mkdir -p "$repo_copy/charts/densecore/charts"
}

cluster_preflight() {
  if [ "$dry_run" -eq 1 ]; then
    log "dry-run: argument/path generation only; skipping live cluster preflight"
    return 0
  fi
  kubectl cluster-info >"$logs_dir/kubectl-cluster-info.log" 2>&1 || fail "kubectl cluster-info failed"
  kubectl version -o json >"$results_dir/kubectl-version.json" 2>"$logs_dir/kubectl-version.stderr" || true
  kubectl get nodes -o json >"$results_dir/nodes.json" 2>"$logs_dir/kubectl-nodes.stderr" || true
}

namespace_exists() {
  kubectl get namespace "$1" >/dev/null 2>&1
}

namespace_json_field() {
  local ns=$1
  local section=$2
  local key=$3
  kubectl get namespace "$ns" -o json | jq -r --arg section "$section" --arg key "$key" '.metadata[$section][$key] // ""'
}

require_owned_namespace() {
  local ns=$1
  local role=$2
  local release=$3
  local owned owner release_owner role_owner
  namespace_exists "$ns" || fail "namespace $ns does not exist"
  owned=$(namespace_json_field "$ns" labels "$ownership_label_key")
  owner=$(namespace_json_field "$ns" annotations "$ownership_owner_key")
  release_owner=$(namespace_json_field "$ns" annotations "$ownership_release_key")
  role_owner=$(namespace_json_field "$ns" annotations "$ownership_role_key")
  [ "$owned" = "true" ] || fail "namespace $ns is not marked as harness-owned"
  [ "$owner" = "$ownership_id" ] || fail "namespace $ns is owned by $owner, not $ownership_id"
  [ "$release_owner" = "$release" ] || fail "namespace $ns release marker is $release_owner, expected $release"
  [ "$role_owner" = "$role" ] || fail "namespace $ns role marker is $role_owner, expected $role"
}

ensure_owned_namespace() {
  local ns=$1
  local role=$2
  local release=$3
  if [ "$dry_run" -eq 1 ]; then
    log "dry-run: argument/path generation only; would ensure owned namespace $ns for role=$role release=$release"
    return 0
  fi
  if namespace_exists "$ns"; then
    require_owned_namespace "$ns" "$role" "$release"
    return 0
  fi
  kubectl create namespace "$ns" >"$logs_dir/create-namespace-${ns}.log" 2>&1
  kubectl label namespace "$ns" "${ownership_label_key}=true" --overwrite >>"$logs_dir/create-namespace-${ns}.log" 2>&1
  kubectl annotate namespace "$ns" \
    "${ownership_owner_key}=${ownership_id}" \
    "${ownership_release_key}=${release}" \
    "${ownership_role_key}=${role}" \
    --overwrite >>"$logs_dir/create-namespace-${ns}.log" 2>&1
}

helm_release_exists() {
  local release=$1
  local ns=$2
  helm status "$release" -n "$ns" >/dev/null 2>&1
}

verify_reusable_release() {
  local release=$1
  local ns=$2
  local chart_prefix=$3
  local chart_version=$4
  local status_json chart_string status
  status_json=$(helm list -n "$ns" -o json | jq -c --arg release "$release" '.[] | select(.name == $release)' | head -n 1)
  [ -n "$status_json" ] || fail "expected reusable release $release in namespace $ns"
  chart_string=$(printf '%s' "$status_json" | jq -r '.chart')
  status=$(printf '%s' "$status_json" | jq -r '.status')
  [ "$status" = "deployed" ] || fail "release $release in namespace $ns is not deployed"
  [ "$chart_string" = "${chart_prefix}-${chart_version}" ] || fail "release $release in namespace $ns is on $chart_string, expected ${chart_prefix}-${chart_version}"
}

wait_pods_ready() {
  local ns=$1
  [ "$dry_run" -eq 1 ] && return 0
  kubectl wait --namespace "$ns" --for=condition=Ready pod --all --timeout="${timeout_seconds}s" \
    >"$logs_dir/wait-pods-${ns}.log" 2>&1
}

discover_prometheus_service() {
  local service_json names count
  if [ -n "$prometheus_service" ]; then
    printf '%s\n' "$prometheus_service"
    return 0
  fi
  service_json=$(kubectl get svc -n "$monitoring_namespace" -l "app.kubernetes.io/instance=${prom_release_name}" -o json)
  names=$(printf '%s' "$service_json" | jq -r '
    [.items[]
      | select(
          (
            any((.spec.ports // [])[]?; (.port == 9090) or ((.name // "") == "web"))
          )
        )
      | .metadata.name
    ]')
  count=$(printf '%s' "$names" | jq 'length')
  [ "$count" -eq 1 ] || fail "expected exactly one Prometheus service in namespace $monitoring_namespace for release $prom_release_name, found $count"
  names=$(printf '%s' "$names" | jq -r '.[0]')
  [ -n "$names" ] || fail "failed to resolve Prometheus service name in namespace $monitoring_namespace"
  printf '%s\n' "$names"
}

extract_chart_dependency_version() {
  awk '
    $1 == "-" && $2 == "name:" && $3 == "dense-base" { in_dep=1; next }
    in_dep && $1 == "version:" { print $2; exit }
    in_dep && $1 == "-" && $2 == "name:" { in_dep=0 }
  ' "$repo_dir/charts/densecore/Chart.yaml"
}

extract_chart_dependency_repo() {
  awk '
    $1 == "-" && $2 == "name:" && $3 == "dense-base" { in_dep=1; next }
    in_dep && $1 == "repository:" { print $2; exit }
    in_dep && $1 == "-" && $2 == "name:" { in_dep=0 }
  ' "$repo_dir/charts/densecore/Chart.yaml"
}

extract_yaml_scalar() {
  local file=$1
  local key=$2
  sed -n "s/^[[:space:]]*${key}:[[:space:]]*\"\\{0,1\\}\\([^\"#]*\\)\"\\{0,1\\}[[:space:]]*$/\\1/p" "$file" | head -n 1
}

extract_extra_env_value() {
  local file=$1
  local target=$2
  awk -v target="$target" '
    $1 == "-" && $2 == "name:" && $3 == target { want=1; next }
    want && $1 == "value:" {
      sub(/^[[:space:]]*value:[[:space:]]*/, "", $0)
      gsub(/"/, "", $0)
      print $0
      exit
    }
  ' "$file"
}

extract_resource_value() {
  local file=$1
  local section=$2
  local field=$3
  awk -v section="$section" -v field="$field" '
    $1 == section ":" { in_section=1; next }
    in_section && $1 == field ":" {
      gsub(/"/, "", $2)
      print $2
      exit
    }
    in_section && /^[^[:space:]]/ { in_section=0 }
  ' "$file"
}

detect_cluster_type() {
  local context_name=$1
  case "$context_name" in
    kind-*) printf 'kind' ;;
    gke_*) printf 'gke' ;;
    minikube) printf 'minikube' ;;
    *) printf 'unknown' ;;
  esac
}

write_manifest() {
  local phase=$1
  local densecore_commit densecloud_version chart_version chart_repo context_name cluster_type helm_version
  local image_repo image_tag model_source model_claim model_path model_filename
  local req_cpu req_mem lim_cpu lim_mem threads max_num_seqs max_prefill_seqs
  local node_count node_names node_machine_types node_cpu node_memory kube_server_version kube_client_version

  densecore_commit=$(git -C "$repo_dir" rev-parse HEAD)
  densecloud_version=$(extract_chart_dependency_version)
  [ -n "$densecloud_version" ] || fail "could not resolve DenseCloud/dense-base dependency version from charts/densecore/Chart.yaml"
  chart_repo=$(extract_chart_dependency_repo)
  [ -n "$chart_repo" ] || fail "could not resolve dense-base repository from charts/densecore/Chart.yaml"
  chart_version=$(sed -n 's/^version:[[:space:]]*//p' "$repo_dir/charts/densecore/Chart.yaml" | head -n 1)
  context_name=$(kubectl config current-context 2>/dev/null || printf 'NOT_AVAILABLE')
  cluster_type=$(detect_cluster_type "$context_name")
  helm_version=$(helm version --template '{{ .Version }}' 2>/dev/null || printf 'NOT_AVAILABLE')

  image_repo="NOT_AVAILABLE"
  image_tag="NOT_AVAILABLE"
  model_source="NOT_AVAILABLE"
  model_claim="NOT_AVAILABLE"
  model_path="NOT_AVAILABLE"
  model_filename="NOT_AVAILABLE"
  req_cpu="NOT_AVAILABLE"
  req_mem="NOT_AVAILABLE"
  lim_cpu="NOT_AVAILABLE"
  lim_mem="NOT_AVAILABLE"
  threads="NOT_AVAILABLE"
  max_num_seqs="NOT_AVAILABLE"
  max_prefill_seqs="NOT_AVAILABLE"

  if [ -n "$helm_values" ] && [ -f "$helm_values" ]; then
    image_repo=$(extract_yaml_scalar "$helm_values" repository)
    image_tag=$(extract_yaml_scalar "$helm_values" tag)
    model_source=$(extract_yaml_scalar "$helm_values" source)
    model_claim=$(extract_yaml_scalar "$helm_values" existingClaim)
    model_path=$(extract_yaml_scalar "$helm_values" path)
    model_filename=$(extract_yaml_scalar "$helm_values" filename)
    req_cpu=$(extract_resource_value "$helm_values" requests cpu)
    req_mem=$(extract_resource_value "$helm_values" requests memory)
    lim_cpu=$(extract_resource_value "$helm_values" limits cpu)
    lim_mem=$(extract_resource_value "$helm_values" limits memory)
    threads=$(extract_extra_env_value "$helm_values" THREADS)
    max_num_seqs=$(extract_extra_env_value "$helm_values" DENSECORE_MAX_NUM_SEQS)
    max_prefill_seqs=$(extract_extra_env_value "$helm_values" DENSECORE_MAX_PREFILL_SEQS)
  fi

  [ -n "$threads_override" ] && threads=$threads_override
  [ -n "$max_num_seqs_override" ] && max_num_seqs=$max_num_seqs_override
  [ -n "$max_prefill_seqs_override" ] && max_prefill_seqs=$max_prefill_seqs_override

  node_count=$(jq -r '.items | length' "$results_dir/nodes.json" 2>/dev/null || printf '0')
  node_names=$(jq -rc '[.items[].metadata.name]' "$results_dir/nodes.json" 2>/dev/null || printf '[]')
  node_machine_types=$(jq -rc '[.items[] | (.metadata.labels["node.kubernetes.io/instance-type"] // .metadata.labels["beta.kubernetes.io/instance-type"] // "NOT_AVAILABLE")]' "$results_dir/nodes.json" 2>/dev/null || printf '[]')
  node_cpu=$(jq -rc '[.items[].status.capacity.cpu]' "$results_dir/nodes.json" 2>/dev/null || printf '[]')
  node_memory=$(jq -rc '[.items[].status.capacity.memory]' "$results_dir/nodes.json" 2>/dev/null || printf '[]')
  kube_server_version=$(jq -r '.serverVersion.gitVersion // "NOT_AVAILABLE"' "$results_dir/kubectl-version.json" 2>/dev/null || printf 'NOT_AVAILABLE')
  kube_client_version=$(jq -r '.clientVersion.gitVersion // "NOT_AVAILABLE"' "$results_dir/kubectl-version.json" 2>/dev/null || printf 'NOT_AVAILABLE')

  cat >"$results_dir/${phase}-manifest.json" <<EOF
{
  "timestamp_utc": $(json_escape "$(now_iso)"),
  "phase": $(json_escape "$phase"),
  "densecore_commit": $(json_escape "$densecore_commit"),
  "densecloud_version": $(json_escape "$densecloud_version"),
  "densebase_chart_repository": $(json_escape "$chart_repo"),
  "densebase_chart_version": $(json_escape "$densecloud_version"),
  "densecore_chart_version": $(json_escape "${chart_version:-NOT_AVAILABLE}"),
  "release_name": $(json_escape "$release_name"),
  "namespace": $(json_escape "$namespace"),
  "keda_release_name": $(json_escape "$keda_release_name"),
  "keda_namespace": $(json_escape "$keda_namespace"),
  "prom_release_name": $(json_escape "$prom_release_name"),
  "monitoring_namespace": $(json_escape "$monitoring_namespace"),
  "prometheus_service": $(json_escape "${prometheus_service:-AUTO_DISCOVER_BY_LABEL}"),
  "ownership_id": $(json_escape "$ownership_id"),
  "chat_model": $(json_escape "$chat_model"),
  "keda_chart_version": $(json_escape "$keda_chart_version"),
  "prometheus_chart_version": $(json_escape "$prom_chart_version"),
  "kubernetes_context": $(json_escape "$context_name"),
  "cluster_type": $(json_escape "$cluster_type"),
  "kubernetes_server_version": $(json_escape "$kube_server_version"),
  "kubernetes_client_version": $(json_escape "$kube_client_version"),
  "node_count": $node_count,
  "node_names": $node_names,
  "node_machine_types": $node_machine_types,
  "node_cpu_capacity": $node_cpu,
  "node_memory_capacity": $node_memory,
  "helm_version": $(json_escape "$helm_version"),
  "base_values_file": $(json_escape "${helm_values:-UNSET}"),
  "image_digest": $(json_escape "${image_digest:-NOT_AVAILABLE}"),
  "image_repository_hint": $(json_escape "${image_repo:-NOT_AVAILABLE}"),
  "image_tag_hint": $(json_escape "${image_tag:-NOT_AVAILABLE}"),
  "model_name": $(json_escape "${model_name:-NOT_AVAILABLE}"),
  "model_quantization": $(json_escape "${model_quantization:-NOT_AVAILABLE}"),
  "model_sha256": $(json_escape "${model_sha256:-NOT_AVAILABLE}"),
  "model_source_hint": $(json_escape "${model_source:-NOT_AVAILABLE}"),
  "model_claim_hint": $(json_escape "${model_claim:-NOT_AVAILABLE}"),
  "model_path_hint": $(json_escape "${model_path:-NOT_AVAILABLE}"),
  "model_filename_hint": $(json_escape "${model_filename:-NOT_AVAILABLE}"),
  "context_length": $(json_escape "${context_length:-NOT_AVAILABLE}"),
  "resource_requests_cpu_hint": $(json_escape "${req_cpu:-NOT_AVAILABLE}"),
  "resource_requests_memory_hint": $(json_escape "${req_mem:-NOT_AVAILABLE}"),
  "resource_limits_cpu_hint": $(json_escape "${lim_cpu:-NOT_AVAILABLE}"),
  "resource_limits_memory_hint": $(json_escape "${lim_mem:-NOT_AVAILABLE}"),
  "threads": $(json_escape "${threads:-NOT_AVAILABLE}"),
  "densecore_max_num_seqs": $(json_escape "${max_num_seqs:-NOT_AVAILABLE}"),
  "densecore_max_prefill_seqs": $(json_escape "${max_prefill_seqs:-NOT_AVAILABLE}"),
  "metric_name": $(json_escape "$metric_name"),
  "metric_query": $(json_escape "$metric_query"),
  "metric_threshold": $(json_escape "${metric_threshold:-NOT_AVAILABLE}"),
  "poll_interval_seconds": $poll_interval,
  "cooldown_period_seconds": $cooldown_period,
  "request_count": $request_count,
  "stream_count": $stream_count,
  "concurrency": $concurrency,
  "max_output_tokens": $max_output_tokens
}
EOF
}

require_real_run_metadata() {
  [ "$dry_run" -eq 1 ] && return 0
  [ -n "$image_digest" ] || fail "real runs require --image-digest"
  [ -n "$model_name" ] || fail "real runs require --model-name"
  [ -n "$model_quantization" ] || fail "real runs require --model-quantization"
  [ -n "$model_sha256" ] || fail "real runs require --model-sha256"
  [ -n "$context_length" ] || fail "real runs require --context-length"
  [ -n "$threads_override" ] || [ -n "$(extract_extra_env_value "$helm_values" THREADS)" ] || fail "real runs require THREADS via values or --threads"
  [ -n "$max_num_seqs_override" ] || [ -n "$(extract_extra_env_value "$helm_values" DENSECORE_MAX_NUM_SEQS)" ] || fail "real runs require DENSECORE_MAX_NUM_SEQS via values or --densecore-max-num-seqs"
  [ -n "$max_prefill_seqs_override" ] || [ -n "$(extract_extra_env_value "$helm_values" DENSECORE_MAX_PREFILL_SEQS)" ] || fail "real runs require DENSECORE_MAX_PREFILL_SEQS via values or --densecore-max-prefill-seqs"
}

helm_prepare_dependencies() {
  ensure_repo_copy
  run_logged "$logs_dir/helm-dependency-update.log" helm dependency update "$repo_copy/charts/densecore"
}

setup_observability() {
  ensure_output_dir
  cluster_preflight
  write_manifest setup
  ensure_owned_namespace "$keda_namespace" "keda" "$keda_release_name"
  ensure_owned_namespace "$monitoring_namespace" "monitoring" "$prom_release_name"
  run_logged "$logs_dir/helm-repo-add-keda.log" helm repo add kedacore https://kedacore.github.io/charts
  run_logged "$logs_dir/helm-repo-add-prometheus.log" helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
  run_logged "$logs_dir/helm-repo-update.log" helm repo update

  cat >"$runtime_dir/kube-prometheus-values.yaml" <<EOF
grafana:
  enabled: false
alertmanager:
  enabled: false
prometheus:
  prometheusSpec:
    retention: 24h
defaultRules:
  create: false
kubeEtcd:
  enabled: false
kubeControllerManager:
  enabled: false
kubeScheduler:
  enabled: false
kubeProxy:
  enabled: false
EOF

  if helm_release_exists "$keda_release_name" "$keda_namespace"; then
    verify_reusable_release "$keda_release_name" "$keda_namespace" "keda" "$keda_chart_version"
  else
    run_logged "$logs_dir/keda-install.log" \
      helm install "$keda_release_name" kedacore/keda \
        --namespace "$keda_namespace" \
        --version "$keda_chart_version" \
        --wait \
        --timeout "${timeout_seconds}s"
  fi

  if helm_release_exists "$prom_release_name" "$monitoring_namespace"; then
    verify_reusable_release "$prom_release_name" "$monitoring_namespace" "kube-prometheus-stack" "$prom_chart_version"
  else
    run_logged "$logs_dir/prometheus-install.log" \
      helm install "$prom_release_name" prometheus-community/kube-prometheus-stack \
        --namespace "$monitoring_namespace" \
        --version "$prom_chart_version" \
        -f "$runtime_dir/kube-prometheus-values.yaml" \
        --wait \
        --timeout "${timeout_seconds}s"
  fi

  if [ "$dry_run" -eq 0 ]; then
    wait_pods_ready "$keda_namespace"
    wait_pods_ready "$monitoring_namespace"
    prometheus_service=$(discover_prometheus_service)
    kubectl get deployment -n "$keda_namespace" >"$results_dir/keda-deployments.txt"
    kubectl get pods -n "$keda_namespace" >"$results_dir/keda-pods.txt"
    kubectl get pods -n "$monitoring_namespace" >"$results_dir/monitoring-pods.txt"
    kubectl get crd scaledobjects.keda.sh >"$results_dir/keda-scaledobject-crd.txt"
    kubectl get crd servicemonitors.monitoring.coreos.com >"$results_dir/servicemonitor-crd.txt"
    kubectl get apiservice >"$results_dir/apiservices.txt"
  fi
}

generate_runtime_values() {
  local mode=$1
  local out=$2
  local prom_service_name=$prometheus_service
  if [ -z "$prom_service_name" ]; then
    prom_service_name="PROMETHEUS_SERVICE_AUTO_DISCOVER"
  fi
  local prom_addr="http://${prom_service_name}.${monitoring_namespace}.svc.cluster.local:9090"
  cat >"$out" <<EOF
autoscaling:
  enabled: false

dense-base:
  replicaCount: 1
  serviceMonitor:
    enabled: true
    labels:
      release: ${prom_release_name}
    path: /metrics
  keda:
    enabled: $([ "$mode" = "autoscale" ] && printf 'true' || printf 'false')
    minReplicaCount: 1
    maxReplicaCount: 2
    pollingInterval: $poll_interval
    cooldownPeriod: $cooldown_period
    advanced:
      restoreToOriginalReplicaCount: false
    triggers:
      custom:
        - type: prometheus
          metadata:
            serverAddress: "$prom_addr"
            metricName: "$metric_name"
            threshold: "$metric_threshold"
            query: >-
              $metric_query
EOF
}

label_selector() {
  printf 'app.kubernetes.io/instance=%s,app.kubernetes.io/name=densecore' "$release_name"
}

discover_one() {
  local kind=$1
  kubectl get "$kind" -n "$namespace" -l "$(label_selector)" -o json | jq -r '.items | if length == 1 then .[0].metadata.name else empty end'
}

wait_for_rollout() {
  local deployment_name=$1
  if [ "$dry_run" -eq 1 ]; then
    log "dry-run: argument/path generation only; skipping rollout wait for deployment/$deployment_name"
    return 0
  fi
  kubectl rollout status "deployment/$deployment_name" -n "$namespace" --timeout="${timeout_seconds}s" >"$logs_dir/rollout-${deployment_name}.log" 2>&1
}

start_service_port_forward() {
  local service_name=$1
  local local_port=$2
  local remote_port=${3:-8080}
  if [ "$dry_run" -eq 1 ]; then
    log "dry-run: argument/path generation only; skipping service port-forward"
    return 0
  fi
  stop_service_port_forward
  kubectl port-forward -n "$namespace" "service/$service_name" "${local_port}:${remote_port}" >"$logs_dir/port-forward-${service_name}.log" 2>&1 &
  service_port_forward_pid=$!
  sleep 3
  kill -0 "$service_port_forward_pid" 2>/dev/null || fail "service port-forward exited early"
}

start_prometheus_port_forward() {
  if [ "$dry_run" -eq 1 ]; then
    log "dry-run: argument/path generation only; skipping Prometheus port-forward"
    return 0
  fi
  if [ -n "${prom_port_forward_pid:-}" ] && kill -0 "$prom_port_forward_pid" 2>/dev/null; then
    return 0
  fi
  prometheus_service=$(discover_prometheus_service)
  kubectl port-forward -n "$monitoring_namespace" "service/$prometheus_service" 19090:9090 >"$logs_dir/port-forward-prometheus.log" 2>&1 &
  prom_port_forward_pid=$!
  sleep 3
  kill -0 "$prom_port_forward_pid" 2>/dev/null || fail "Prometheus port-forward exited early"
}

wait_for_http() {
  local url=$1
  local remaining=$2
  local code
  while [ "$remaining" -gt 0 ]; do
    code=$(curl -sS -o /dev/null -w '%{http_code}' "$url" || true)
    if [ "$code" = "200" ]; then
      return 0
    fi
    sleep 1
    remaining=$((remaining - 1))
  done
  return 1
}

prometheus_query() {
  local query=$1
  curl -fsS --get "http://127.0.0.1:19090/api/v1/query" --data-urlencode "query=$query"
}

verify_metric_query() {
  local out="$results_dir/promql-check.json"
  local settle_timeout_seconds=90
  local waited=0
  if [ "$dry_run" -eq 1 ]; then
    log "dry-run: argument/path generation only; skipping live Prometheus query verification"
    return 0
  fi
  while [ "$waited" -le "$settle_timeout_seconds" ]; do
    prometheus_query "$metric_query" >"$out"
    if jq -e '.status == "success" and (.data.result | length) > 0' "$out" >/dev/null; then
      return 0
    fi
    sleep 5
    waited=$((waited + 5))
  done
  fail "Prometheus query returned no data within ${settle_timeout_seconds}s: $metric_query"
}

sample_json() {
  local phase=$1
  local deployment_name=$2
  local service_name=$3
  local scaledobject_name=$4
  local hpa_name=$5
  local prom_json metric_value pods_json

  prom_json=$(prometheus_query "$metric_query")
  metric_value=$(printf '%s' "$prom_json" | jq -r '.data.result[0].value[1] // empty')
  [ -n "$metric_value" ] || fail "Prometheus query returned no scalar value for $metric_query"
  pods_json=$(kubectl get pods -n "$namespace" -l "$(label_selector)" -o json)

  jq -n \
    --arg timestamp "$(now_iso)" \
    --arg phase "$phase" \
    --arg deployment "$deployment_name" \
    --arg service "$service_name" \
    --arg scaledobject "$scaledobject_name" \
    --arg hpa "$hpa_name" \
    --arg metric_query "$metric_query" \
    --arg metric_value "$metric_value" \
    --argjson deploy "$(kubectl get deployment "$deployment_name" -n "$namespace" -o json)" \
    --argjson so "$(kubectl get scaledobject "$scaledobject_name" -n "$namespace" -o json 2>/dev/null || printf '{}')" \
    --argjson hpa_json "$(kubectl get hpa "$hpa_name" -n "$namespace" -o json 2>/dev/null || printf '{}')" \
    --argjson pods "$pods_json" \
    '{
      timestamp_utc: $timestamp,
      phase: $phase,
      deployment: $deployment,
      service: $service,
      scaledobject: $scaledobject,
      hpa: $hpa,
      metric_query: $metric_query,
      metric_value: ($metric_value | tonumber),
      requested_replicas: ($deploy.spec.replicas // 0),
      current_replicas: ($deploy.status.replicas // 0),
      ready_replicas: ($deploy.status.readyReplicas // 0),
      available_replicas: ($deploy.status.availableReplicas // 0),
      updated_replicas: ($deploy.status.updatedReplicas // 0),
      hpa_current_replicas: ($hpa_json.status.currentReplicas // 0),
      hpa_desired_replicas: ($hpa_json.status.desiredReplicas // 0),
      scaledobject_ready: ($so.status.conditions // [] | map(select(.type == "Ready")) | .[0].status // "Unknown"),
      scaledobject_active: ($so.status.conditions // [] | map(select(.type == "Active")) | .[0].status // "Unknown"),
      pods: ($pods.items | map({
        name: .metadata.name,
        phase: .status.phase,
        pod_ip: .status.podIP,
        image_id: (.status.containerStatuses[0].imageID // ""),
        restarts: ([.status.containerStatuses[]?.restartCount] | add // 0),
        created_at: .metadata.creationTimestamp,
        start_time: .status.startTime,
        scheduled_transition: (.status.conditions // [] | map(select(.type == "PodScheduled")) | .[0].lastTransitionTime // ""),
        ready: (([.status.containerStatuses[]? | select(.ready == true)] | length) > 0),
        ready_transition: (.status.conditions // [] | map(select(.type == "Ready")) | .[0].lastTransitionTime // ""),
        startup_transition: (.status.conditions // [] | map(select(.type == "ContainersReady")) | .[0].lastTransitionTime // "")
      }))
    }'
}

sample_csv_header() {
  printf 'timestamp_utc,phase,metric_value,requested_replicas,current_replicas,ready_replicas,available_replicas,hpa_current_replicas,hpa_desired_replicas,scaledobject_ready,scaledobject_active,pod_names\n'
}

json_to_csv_row() {
  jq -r '[
    .timestamp_utc,
    .phase,
    .metric_value,
    .requested_replicas,
    .current_replicas,
    .ready_replicas,
    .available_replicas,
    .hpa_current_replicas,
    .hpa_desired_replicas,
    .scaledobject_ready,
    .scaledobject_active,
    (.pods | map(.name) | join(";"))
  ] | @csv'
}

collect_telemetry() {
  local phase=$1
  local deployment_name=$2
  local service_name=$3
  local scaledobject_name=$4
  local hpa_name=$5
  local duration=$6
  local jsonl="$results_dir/${phase}-telemetry.jsonl"
  local csv="$results_dir/${phase}-telemetry.csv"
  local loops elapsed=0

  if [ "$dry_run" -eq 1 ]; then
    log "dry-run: argument/path generation only; skipping telemetry collection for phase=$phase duration=${duration}s"
    return 0
  fi

  sample_csv_header >"$csv"
  loops=$((duration / 5))
  if [ "$loops" -lt 1 ]; then
    loops=1
  fi

  while [ "$elapsed" -lt "$loops" ]; do
    sample_json "$phase" "$deployment_name" "$service_name" "$scaledobject_name" "$hpa_name" | tee -a "$jsonl" | json_to_csv_row >>"$csv"
    sleep 5
    elapsed=$((elapsed + 1))
  done
}

start_background_collector() {
  local phase=$1
  local deployment_name=$2
  local service_name=$3
  local scaledobject_name=$4
  local hpa_name=$5
  local duration=$6
  if [ "$dry_run" -eq 1 ]; then
    log "dry-run: argument/path generation only; skipping background collector"
    return 0
  fi
  (
    collect_telemetry "$phase" "$deployment_name" "$service_name" "$scaledobject_name" "$hpa_name" "$duration"
  ) &
  collector_pid=$!
}

stop_background_collector() {
  if [ -n "${collector_pid:-}" ] && kill -0 "$collector_pid" 2>/dev/null; then
    kill "$collector_pid" 2>/dev/null || true
    wait "$collector_pid" 2>/dev/null || true
  fi
}

capture_pod_metric_snapshot() {
  local phase=$1
  local metric=$2
  local outfile="$results_dir/${phase}-per-pod-${metric}.json"
  if [ "$dry_run" -eq 1 ]; then
    log "dry-run: argument/path generation only; skipping per-pod metric snapshot for $metric"
    return 0
  fi
  kubectl get pods -n "$namespace" -l "$(label_selector)" -o json \
    | jq -r '.items[].metadata.name' \
    | while read -r pod_name; do
        local value
        value=$(kubectl get --raw "/api/v1/namespaces/${namespace}/pods/${pod_name}:8080/proxy/metrics" | awk -v m="$metric" '$1 == m {print $2; exit}')
        jq -nc --arg pod "$pod_name" --arg metric "$metric" --arg value "${value:-0}" '{pod: $pod, metric: $metric, value: ($value | tonumber? // 0)}'
      done | jq -s '.' >"$outfile"
}

request_payload() {
  local prompt=$1
  local stream_flag=$2
  jq -nc \
    --arg model "$chat_model" \
    --arg prompt "$prompt" \
    --argjson stream "$stream_flag" \
    --argjson max_tokens "$max_output_tokens" \
    '{
      model: $model,
      stream: $stream,
      max_tokens: $max_tokens,
      messages: [
        {role: "system", content: "You are DenseCore qualification smoke."},
        {role: "user", content: $prompt}
      ]
    }'
}

emit_request_log() {
  local file=$1
  shift
  jq -nc \
    --arg timestamp "$(now_iso)" \
    --arg request_id "$1" \
    --arg phase "$2" \
    --arg mode "$3" \
    --arg url "$4" \
    --arg http_status "$5" \
    --arg start_ms "$6" \
    --arg end_ms "$7" \
    --arg ttft_ms "$8" \
    --arg prompt_tokens "$9" \
    --arg output_tokens "${10}" \
    --arg terminal_status "${11}" \
    --arg error_text "${12}" \
    '{
      timestamp_utc: $timestamp,
      request_id: $request_id,
      phase: $phase,
      mode: $mode,
      url: $url,
      http_status: ($http_status | tonumber? // 0),
      start_ms: ($start_ms | tonumber),
      end_ms: ($end_ms | tonumber),
      total_latency_ms: (($end_ms | tonumber) - ($start_ms | tonumber)),
      ttft_ms: ($ttft_ms | tonumber? // null),
      prompt_tokens: ($prompt_tokens | tonumber? // null),
      output_tokens: ($output_tokens | tonumber? // null),
      terminal_status: $terminal_status,
      error_text: $error_text
    }' >>"$file"
}

run_nonstream_request() {
  local phase=$1
  local request_id=$2
  local url=$3
  local prompt=$4
  local body_file="$runtime_dir/${phase}-${request_id}.json"
  local response_file="$runtime_dir/${phase}-${request_id}.response.json"
  local start_ms end_ms http_status prompt_tokens output_tokens error_text
  request_payload "$prompt" false >"$body_file"
  start_ms=$(now_ms)
  http_status=$(curl -sS -o "$response_file" -w '%{http_code}' -H 'Content-Type: application/json' --data @"$body_file" "$url" || true)
  end_ms=$(now_ms)
  prompt_tokens=$(jq -r '.usage.prompt_tokens // empty' "$response_file" 2>/dev/null || true)
  output_tokens=$(jq -r '.usage.completion_tokens // empty' "$response_file" 2>/dev/null || true)
  error_text=$(jq -r '.error.message // empty' "$response_file" 2>/dev/null || true)
  emit_request_log "$results_dir/${phase}-requests.jsonl" \
    "$request_id" "$phase" "nonstream" "$url" "$http_status" "$start_ms" "$end_ms" "" \
    "${prompt_tokens:-}" "${output_tokens:-}" "$( [ "$http_status" = "200" ] && printf success || printf http_error )" "${error_text:-}"
}

run_stream_request() {
  local phase=$1
  local request_id=$2
  local url=$3
  local prompt=$4
  local body_file="$runtime_dir/${phase}-${request_id}.json"
  local stream_file="$runtime_dir/${phase}-${request_id}.stream.txt"
  local header_file="$runtime_dir/${phase}-${request_id}.headers.txt"
  local pipe_file="$runtime_dir/${phase}-${request_id}.fifo"
  local curl_err_file="$runtime_dir/${phase}-${request_id}.curl.err"
  local first_event_file="$runtime_dir/${phase}-${request_id}.ttft"
  local terminal_file="$runtime_dir/${phase}-${request_id}.terminal"
  local start_ms end_ms first_event_ms="" http_status terminal_status="missing_done" error_text="" curl_status=0
  request_payload "$prompt" true >"$body_file"
  start_ms=$(now_ms)
  : >"$stream_file"
  : >"$header_file"
  rm -f "$pipe_file"
  mkfifo "$pipe_file"
  set +e
  stdbuf -oL curl -sS -N -D "$header_file" -H 'Content-Type: application/json' --data @"$body_file" "$url" >"$pipe_file" 2>"$curl_err_file" &
  local curl_pid=$!
  while IFS= read -r line; do
    printf '%s\n' "$line" >>"$stream_file"
    if [ ! -s "$first_event_file" ] && [[ "$line" == data:* ]] && [[ "$line" != "data: [DONE]" ]]; then
      now_ms >"$first_event_file"
    fi
    if [ "$line" = "data: [DONE]" ]; then
      printf 'done\n' >"$terminal_file"
    fi
  done <"$pipe_file"
  wait "$curl_pid"
  curl_status=$?
  set -e
  rm -f "$pipe_file"
  end_ms=$(now_ms)
  http_status=$(awk 'toupper($1) ~ /^HTTP\// {code=$2} END {print code+0}' "$header_file")
  if [ -s "$first_event_file" ]; then
    first_event_ms=$(cat "$first_event_file")
  fi
  if [ -s "$terminal_file" ]; then
    terminal_status=$(tr -d '\r\n' <"$terminal_file")
  fi
  if [ "$curl_status" -ne 0 ]; then
    error_text=$(tr '\n' ' ' <"$curl_err_file" | sed 's/[[:space:]]\+/ /g; s/^ //; s/ $//')
    if [ -z "$error_text" ]; then
      error_text="curl_exit_${curl_status}"
    fi
  fi
  emit_request_log "$results_dir/${phase}-requests.jsonl" \
    "$request_id" "$phase" "stream" "$url" "${http_status:-0}" "$start_ms" "$end_ms" "${first_event_ms:-}" "" "" "$terminal_status" "$error_text"
}

run_load() {
  local phase=$1
  local local_port=$2
  local url="http://127.0.0.1:${local_port}/v1/chat/completions"
  local i active=0
  local prompt_base="Summarize why queue-aware autoscaling matters for CPU inference."
  local background_pids=()
  : >"$results_dir/${phase}-requests.jsonl"

  for i in $(seq 1 "$request_count"); do
    if [ "$i" -le "$stream_count" ]; then
      run_stream_request "$phase" "req-${i}" "$url" "${prompt_base} Streaming request ${i}." &
    else
      run_nonstream_request "$phase" "req-${i}" "$url" "${prompt_base} Non-stream request ${i}." &
    fi
    background_pids+=($!)
    active=$((active + 1))
    if [ "$active" -ge "$concurrency" ]; then
      wait "${background_pids[0]}"
      background_pids=("${background_pids[@]:1}")
      active=$((active - 1))
    fi
  done

  for pid in "${background_pids[@]}"; do
    wait "$pid"
  done
}

summarize_requests() {
  local phase=$1
  local infile="$results_dir/${phase}-requests.jsonl"
  local outfile="$results_dir/${phase}-request-summary.json"
  if [ "$dry_run" -eq 1 ]; then
    log "dry-run: argument/path generation only; skipping request summary"
    return 0
  fi
  jq -s '
    {
      request_count: length,
      success_count: map(select(.http_status == 200)) | length,
      responses_429: map(select(.http_status == 429)) | length,
      responses_5xx: map(select(.http_status >= 500)) | length,
      error_count: map(select(.http_status >= 400)) | length,
      terminal_done_count: map(select(.mode == "stream" and .terminal_status == "done")) | length,
      batch_start_ms: (map(.start_ms) | min // null),
      batch_end_ms: (map(.end_ms) | max // null),
      batch_duration_ms: ((map(.end_ms) | max // 0) - (map(.start_ms) | min // 0)),
      latency_ms_p50: (map(.total_latency_ms) | sort | .[(length / 2 | floor)] // null),
      latency_ms_p95: (map(.total_latency_ms) | sort | .[((length * 95 / 100) | floor)] // null),
      ttft_ms_p50: (map(select(.ttft_ms != null) | .ttft_ms) | sort | .[(length / 2 | floor)] // null),
      ttft_ms_p95: (map(select(.ttft_ms != null) | .ttft_ms) | sort | .[((length * 95 / 100) | floor)] // null)
    }' "$infile" >"$outfile"
}

capture_phase_artifacts() {
  local phase=$1
  local deployment_name=$2
  local service_name=$3
  local scaledobject_name=$4
  local hpa_name=$5
  [ "$dry_run" -eq 0 ] || return 0
  kubectl get deployment "$deployment_name" -n "$namespace" -o yaml >"$results_dir/${phase}-deployment.yaml"
  kubectl get service "$service_name" -n "$namespace" -o yaml >"$results_dir/${phase}-service.yaml"
  kubectl get pods -n "$namespace" -l "$(label_selector)" -o yaml >"$results_dir/${phase}-pods.yaml"
  kubectl get pods -n "$namespace" -l "$(label_selector)" -o json >"$results_dir/${phase}-pods.json"
  kubectl get events -n "$namespace" --sort-by=.lastTimestamp >"$results_dir/${phase}-events.txt" 2>"$logs_dir/${phase}-events.stderr" || true
  if [ -n "$scaledobject_name" ] && [ "$scaledobject_name" != "none" ]; then
    kubectl get scaledobject "$scaledobject_name" -n "$namespace" -o yaml >"$results_dir/${phase}-scaledobject.yaml"
  fi
  if [ -n "$hpa_name" ] && [ "$hpa_name" != "none" ]; then
    kubectl get hpa "$hpa_name" -n "$namespace" -o yaml >"$results_dir/${phase}-hpa.yaml"
  fi
  jq -n \
    --slurpfile pods "$results_dir/${phase}-pods.json" \
    --slurpfile version "$results_dir/kubectl-version.json" \
    '{
      kubernetes_server_version: ($version[0].serverVersion.gitVersion // "NOT_AVAILABLE"),
      pod_images: (($pods[0].items // []) | map({
        pod: .metadata.name,
        image: (.spec.containers[0].image // ""),
        image_id: (.status.containerStatuses[0].imageID // ""),
        created_at: .metadata.creationTimestamp,
        scheduled_at: ((.status.conditions // [] | map(select(.type == "PodScheduled")) | .[0].lastTransitionTime) // ""),
        started_at: (.status.startTime // ""),
        ready_at: ((.status.conditions // [] | map(select(.type == "Ready")) | .[0].lastTransitionTime) // "")
      }))
    }' >"$results_dir/${phase}-runtime-metadata.json"
}

build_phase_assertions() {
  local phase=$1
  local out="$results_dir/${phase}-assertions.json"
  jq -n \
    --arg phase "$phase" \
    --slurpfile req "$results_dir/${phase}-request-summary.json" \
    --slurpfile tele "$results_dir/${phase}-telemetry.jsonl" \
    --slurpfile tail "$results_dir/${phase}-tail-telemetry.jsonl" \
    --slurpfile pre "$results_dir/${phase}-pre-per-pod-${per_pod_proof_metric}.json" \
    --slurpfile post "$results_dir/${phase}-post-per-pod-${per_pod_proof_metric}.json" \
    '
    def req: ($req[0] // {});
    def tele: $tele;
    def tail: $tail;
    def samples: (tele + tail);
    def pre: ($pre[0] // []);
    def post: ($post[0] // []);
    def post_delta_by_pod:
      [post[] as $p |
        {
          pod: $p.pod,
          delta: (($p.value // 0) - ((pre[] | select(.pod == $p.pod) | .value) // 0))
        }
      ];
    {
      phase: $phase,
      request_count: (req.request_count // 0),
      success_count: (req.success_count // 0),
      responses_5xx: (req.responses_5xx // 0),
      error_count: (req.error_count // 0),
      terminal_done_count: (req.terminal_done_count // 0),
      max_metric_value: ([samples[].metric_value] | map(select(. != null)) | max // null),
      max_ready_replicas: ([samples[].ready_replicas] | map(select(. != null)) | max // 0),
      max_current_replicas: ([samples[].current_replicas] | map(select(. != null)) | max // 0),
      final_ready_replicas: (tail[-1].ready_replicas // tele[-1].ready_replicas // 0),
      final_current_replicas: (tail[-1].current_replicas // tele[-1].current_replicas // 0),
      final_hpa_desired_replicas: (tail[-1].hpa_desired_replicas // tele[-1].hpa_desired_replicas // 0),
      backlog_present_with_two_ready: any(samples[]; (.ready_replicas // 0) >= 2 and (.metric_value // 0) > 0),
      second_pod_request_deltas: post_delta_by_pod,
      second_pod_handled_work: any(post_delta_by_pod[]; .delta > 0 and ((pre | length) == 0 or .pod != pre[0].pod))
    }' >"$out"
}

assert_phase_results() {
  local phase=$1
  local assertion_file="$results_dir/${phase}-assertions.json"
  [ "$dry_run" -eq 0 ] || return 0
  build_phase_assertions "$phase"
  jq -e '.success_count > 0 and .success_count == .request_count and .responses_5xx == 0 and .error_count == 0' "$assertion_file" >/dev/null || fail "$phase requests did not complete cleanly"
  if [ "$phase" = "autoscale" ]; then
    jq -e '.max_ready_replicas >= 2 or .max_current_replicas >= 2' "$assertion_file" >/dev/null || fail "autoscale phase never scaled to a second replica"
    jq -e '.final_ready_replicas <= 1 and .final_current_replicas <= 1 and .final_hpa_desired_replicas <= 1' "$assertion_file" >/dev/null || fail "autoscale phase did not scale back to one replica"
    jq -e '.backlog_present_with_two_ready == true' "$assertion_file" >/dev/null || fail "autoscale phase did not show remaining workload after the second pod became ready"
    jq -e '.second_pod_handled_work == true' "$assertion_file" >/dev/null || fail "autoscale phase did not prove that a second pod handled real requests"
    if [ -f "$results_dir/control-request-summary.json" ]; then
      jq -n \
        --slurpfile control "$results_dir/control-request-summary.json" \
        --slurpfile auto "$results_dir/autoscale-request-summary.json" \
        '{
          control_batch_duration_ms: ($control[0].batch_duration_ms // null),
          autoscale_batch_duration_ms: ($auto[0].batch_duration_ms // null),
          control_latency_ms_p95: ($control[0].latency_ms_p95 // null),
          autoscale_latency_ms_p95: ($auto[0].latency_ms_p95 // null),
          improved_vs_control:
            (
              (($auto[0].batch_duration_ms // 1e18) < ($control[0].batch_duration_ms // 1e18))
              or
              (($auto[0].latency_ms_p95 // 1e18) < ($control[0].latency_ms_p95 // 1e18))
            )
        }' >"$results_dir/autoscale-vs-control.json"
      jq -e '.improved_vs_control == true' "$results_dir/autoscale-vs-control.json" >/dev/null || fail "autoscale phase did not show a measurable improvement versus control in the available request evidence"
    fi
  fi
}

deploy_and_run_phase() {
  local phase=$1
  local runtime_values="$runtime_dir/${phase}-overrides.yaml"
  local deployment_name service_name scaledobject_name hpa_name

  ensure_output_dir
  cluster_preflight
  if [ "$dry_run" -eq 0 ]; then
    prometheus_service=$(discover_prometheus_service)
  fi
  write_manifest "$phase"
  ensure_owned_namespace "$namespace" "densecore" "$release_name"
  helm_prepare_dependencies
  generate_runtime_values "$phase" "$runtime_values"
  [ "$dry_run" -eq 1 ] || cp -f "$helm_values" "$runtime_dir/${phase}-base-values.yaml"

  stop_service_port_forward
  run_logged "$logs_dir/helm-upgrade-${phase}.log" \
    helm upgrade --install "$release_name" "$repo_copy/charts/densecore" \
      --namespace "$namespace" \
      --create-namespace \
      -f "$helm_values" \
      -f "$runtime_values" \
      --wait \
      --timeout "${timeout_seconds}s"

  if [ "$dry_run" -eq 1 ]; then
    return 0
  fi

  deployment_name=$(discover_one deployment)
  [ -n "$deployment_name" ] || fail "could not discover a single DenseCore deployment in namespace $namespace"
  service_name=$(discover_one service)
  [ -n "$service_name" ] || fail "could not discover a single DenseCore service in namespace $namespace"
  scaledobject_name=$(kubectl get scaledobject -n "$namespace" -l "$(label_selector)" -o json | jq -r '.items[0].metadata.name // "none"')
  hpa_name=$(kubectl get hpa -n "$namespace" -l "$(label_selector)" -o json | jq -r '.items[0].metadata.name // "none"')

  wait_for_rollout "$deployment_name"
  start_service_port_forward "$service_name" 18080 8080
  start_prometheus_port_forward
  wait_for_http "http://127.0.0.1:18080/health/startup" "$timeout_seconds" || fail "startup probe did not reach 200"
  wait_for_http "http://127.0.0.1:18080/health/live" "$timeout_seconds" || fail "liveness probe did not reach 200"
  wait_for_http "http://127.0.0.1:18080/health/ready" "$timeout_seconds" || fail "readiness probe did not reach 200"
  curl -fsS "http://127.0.0.1:18080/metrics" >"$results_dir/${phase}-service-metrics.txt"
  grep -q 'densecore_pending_requests' "$results_dir/${phase}-service-metrics.txt" || fail "DenseCore pending request metric was not exported"
  grep -q 'densecloud_http_requests_total' "$results_dir/${phase}-service-metrics.txt" || fail "DenseCloud request metrics were not exported"

  verify_metric_query
  capture_pod_metric_snapshot "${phase}-pre" "$per_pod_proof_metric"
  start_background_collector "$phase" "$deployment_name" "$service_name" "$scaledobject_name" "$hpa_name" $((collect_seconds + timeout_seconds))
  run_load "$phase" 18080
  summarize_requests "$phase"
  stop_background_collector
  capture_pod_metric_snapshot "${phase}-post" "$per_pod_proof_metric"
  collect_telemetry "${phase}-tail" "$deployment_name" "$service_name" "$scaledobject_name" "$hpa_name" "$collect_seconds"
  capture_phase_artifacts "$phase" "$deployment_name" "$service_name" "$scaledobject_name" "$hpa_name"
  assert_phase_results "$phase"
  stop_service_port_forward
}

run_cleanup() {
  ensure_output_dir
  if [ "$dry_run" -eq 1 ]; then
    log "dry-run: argument/path generation only; would verify ownership, uninstall release $release_name, and delete namespace $namespace"
    if [ "$cleanup_observability" -eq 1 ]; then
      log "dry-run: argument/path generation only; would also uninstall owned releases $keda_release_name/$prom_release_name and delete namespaces $keda_namespace/$monitoring_namespace"
    fi
    return 0
  fi
  require_owned_namespace "$namespace" "densecore" "$release_name"
  stop_service_port_forward
  stop_prometheus_port_forward
  helm uninstall "$release_name" -n "$namespace" >"$logs_dir/cleanup-helm-uninstall.log" 2>&1 || true
  kubectl delete namespace "$namespace" --wait=false >"$logs_dir/cleanup-namespace.log" 2>&1 || true
  if [ "$cleanup_observability" -eq 1 ]; then
    require_owned_namespace "$keda_namespace" "keda" "$keda_release_name"
    require_owned_namespace "$monitoring_namespace" "monitoring" "$prom_release_name"
    if helm_release_exists "$keda_release_name" "$keda_namespace"; then
      helm uninstall "$keda_release_name" -n "$keda_namespace" >"$logs_dir/cleanup-keda-helm-uninstall.log" 2>&1 || true
    fi
    if helm_release_exists "$prom_release_name" "$monitoring_namespace"; then
      helm uninstall "$prom_release_name" -n "$monitoring_namespace" >"$logs_dir/cleanup-prom-helm-uninstall.log" 2>&1 || true
    fi
    kubectl delete namespace "$keda_namespace" --wait=false >"$logs_dir/cleanup-keda-namespace.log" 2>&1 || true
    kubectl delete namespace "$monitoring_namespace" --wait=false >"$logs_dir/cleanup-monitoring-namespace.log" 2>&1 || true
  fi
}

main() {
  validate_args
  require_base_cmds
  case "$command_name" in
    control|autoscale|all) require_real_run_metadata ;;
  esac

  case "$command_name" in
    setup)
      ensure_output_dir
      setup_observability
      ;;
    control)
      ensure_output_dir
      deploy_and_run_phase control
      ;;
    autoscale)
      ensure_output_dir
      deploy_and_run_phase autoscale
      ;;
    collect)
      ensure_output_dir
      require_owned_namespace "$namespace" "densecore" "$release_name"
      start_prometheus_port_forward
      local_deployment=$(discover_one deployment)
      local_service=$(discover_one service)
      local_scaledobject=$(kubectl get scaledobject -n "$namespace" -l "$(label_selector)" -o json | jq -r '.items[0].metadata.name // "none"')
      local_hpa=$(kubectl get hpa -n "$namespace" -l "$(label_selector)" -o json | jq -r '.items[0].metadata.name // "none"')
      [ -n "$local_deployment" ] || fail "collect could not find a DenseCore deployment"
      [ -n "$local_service" ] || fail "collect could not find a DenseCore service"
      verify_metric_query
      collect_telemetry collect "$local_deployment" "$local_service" "$local_scaledobject" "$local_hpa" "$collect_seconds"
      ;;
    cleanup)
      ensure_output_dir
      run_cleanup
      ;;
    all)
      ensure_output_dir
      setup_observability
      deploy_and_run_phase control
      deploy_and_run_phase autoscale
      run_cleanup
      ;;
  esac
}

main "$@"
