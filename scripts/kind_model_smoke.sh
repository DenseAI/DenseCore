#!/usr/bin/env bash
set -euo pipefail
# Uses a dedicated namespace in a dedicated kind cluster; never changes an existing cluster.
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cluster="densecore-smoke-$$"
namespace=densecore-smoke
artifacts=${DENSECORE_SMOKE_ARTIFACTS:-$(mktemp -d /tmp/densecore-kind-smoke.XXXXXX)}
mkdir -p "$artifacts"
# Keep kind from replacing the user's current context or modifying its kubeconfig.
export KUBECONFIG="$artifacts/kubeconfig"
cleanup() {
  local status=$?
  kubectl --context "kind-$cluster" -n "$namespace" get pods -o yaml > "$artifacts/pods.yaml" 2>&1 || true
  kubectl --context "kind-$cluster" -n "$namespace" logs -l app.kubernetes.io/instance=smoke --all-containers=true > "$artifacts/pods.log" 2>&1 || true
  [[ -z "${forward_pid:-}" ]] || kill "$forward_pid" 2>/dev/null || true
  if ! kind delete cluster --name "$cluster" > "$artifacts/cleanup.log" 2>&1; then
    echo "kind cluster cleanup failed; inspect $artifacts/cleanup.log" >&2
    status=1
  fi
  # This generated file contains ephemeral client credentials; keep it out of uploaded evidence.
  rm -f "$KUBECONFIG"
  echo "kind smoke artifacts: $artifacts"
  return "$status"
}
trap cleanup EXIT
for command in docker kind helm kubectl python3; do command -v "$command" >/dev/null; done
kind create cluster --name "$cluster" --wait 120s > "$artifacts/create.log" 2>&1
kind load docker-image densecore:local densecore-downloader:local --name "$cluster"
kubectl --context "kind-$cluster" create namespace "$namespace"
# Existing corrupt cache must fail the real init image before any server starts.
cat <<'YAML' | kubectl --context "kind-$cluster" -n "$namespace" apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: checksum-failure
spec:
  restartPolicy: Never
  securityContext: {runAsUser: 1000, fsGroup: 1000}
  initContainers:
    - name: seed-corrupt-cache
      image: densecore-downloader:local
      imagePullPolicy: Never
      command: [python3, -c, "from pathlib import Path; Path('/models/Qwen3.5-0.8B-Q4_K_M.gguf').write_bytes(b'wrong model')"]
      volumeMounts: [{name: model, mountPath: /models}]
    - name: verify-checksum
      image: densecore-downloader:local
      imagePullPolicy: Never
      volumeMounts: [{name: model, mountPath: /models}]
  containers:
    - name: must-not-start
      image: densecore-downloader:local
      imagePullPolicy: Never
      command: [python3, -c, "raise SystemExit('server must never start')"]
  volumes: [{name: model, emptyDir: {}}]
YAML
for _ in $(seq 1 60); do
  phase=$(kubectl --context "kind-$cluster" -n "$namespace" get pod checksum-failure -o jsonpath='{.status.phase}')
  [[ "$phase" == Failed ]] && break
  sleep 1
done
kubectl --context "kind-$cluster" -n "$namespace" get pod checksum-failure -o json > "$artifacts/checksum-failure.json"
kubectl --context "kind-$cluster" -n "$namespace" logs checksum-failure -c verify-checksum > "$artifacts/checksum-failure.log"
python3 - "$artifacts/checksum-failure.json" <<'PYTEST'
import json, sys
pod = json.load(open(sys.argv[1]))
checks = pod['status']['initContainerStatuses']
assert checks[1]['state']['terminated']['exitCode'] == 1, checks
assert not pod['status'].get('containerStatuses') or all(not c['state'].get('running') for c in pod['status']['containerStatuses'])
PYTEST
helm dependency build "$root/charts/densecore" > "$artifacts/chart-dependencies.log" 2>&1
helm template smoke "$root/charts/densecore" -f "$root/charts/densecore/examples/values-local-pod.yaml" > "$artifacts/rendered.yaml"
helm --kube-context "kind-$cluster" upgrade --install smoke "$root/charts/densecore" -n "$namespace" \
  -f "$root/charts/densecore/examples/values-local-pod.yaml" --wait --timeout 10m
pod=$(kubectl --context "kind-$cluster" -n "$namespace" get pods -l app.kubernetes.io/instance=smoke -o jsonpath='{.items[0].metadata.name}')
kubectl --context "kind-$cluster" -n "$namespace" get pod "$pod" -o json > "$artifacts/pod-before.json"
kubectl --context "kind-$cluster" -n "$namespace" port-forward "pod/$pod" 18081:8080 > "$artifacts/port-forward.log" 2>&1 &
forward_pid=$!
for _ in $(seq 1 60); do curl -fsS http://127.0.0.1:18081/health/ready > /dev/null && break; sleep 1; done
curl -fsS http://127.0.0.1:18081/health/ready > "$artifacts/ready.json"
# Deleting the pod delivers SIGTERM while an actual accepted request is generating.
python3 "$root/scripts/active_drain_smoke.py" --url http://127.0.0.1:18081 --output "$artifacts/drain.json" --timeout 30 --max-tokens "${DENSECORE_SMOKE_DRAIN_MAX_TOKENS:-32}" -- \
  kubectl --context "kind-$cluster" -n "$namespace" delete pod "$pod" --wait=false
kubectl --context "kind-$cluster" -n "$namespace" wait --for=delete "pod/$pod" --timeout=45s
# Resource limits must be present in the installed pod, not only in the values file.
python3 - "$artifacts/pod-before.json" <<'PY'
import json, sys
pod = json.load(open(sys.argv[1]))
server = pod['spec']['containers'][0]
assert server['resources']['limits'] == {'cpu': '2', 'memory': '4Gi'}, server['resources']
status = pod['status']['initContainerStatuses'][0]
assert status['state']['terminated']['exitCode'] == 0, status
PY
echo 'kind model/downloader/limits/active-drain smoke passed'
