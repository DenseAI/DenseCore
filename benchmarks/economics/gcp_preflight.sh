#!/usr/bin/env bash
set -euo pipefail

project="${GOOGLE_CLOUD_PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}"
if [[ -z "$project" || "$project" == "(unset)" ]]; then
  echo "ERROR: set GOOGLE_CLOUD_PROJECT or gcloud core/project before a benchmark." >&2
  exit 1
fi

echo "project=$project"
echo
echo "Running instances (these already incur compute charges):"
instances="$(gcloud compute instances list --project "$project" \
  --format='value(name,status,zone,machineType.basename())')"
running_instances="$(printf '%s\n' "$instances" | awk '$2 == "RUNNING"')"
if [[ -n "$running_instances" ]]; then
  printf 'NAME\tSTATUS\tZONE\tMACHINE_TYPE\n%s\n' "$running_instances"
else
  echo "(none)"
fi

if [[ "${ALLOW_RUNNING_INSTANCES:-0}" != "1" && -n "$running_instances" ]]; then
  echo "ERROR: existing running instances found. Inspect them before any economics run." >&2
  echo "Set ALLOW_RUNNING_INSTANCES=1 only after confirming they are intentional." >&2
  exit 2
fi

echo
echo "Retained disks (continue billing when unattached):"
gcloud compute disks list --project "$project" --format='table(name,zone,status,sizeGb,type.basename(),users)'
echo
echo "Reserved addresses (may continue billing while unused):"
gcloud compute addresses list --project "$project" --format='table(name,region,status,address,users)'
echo
echo "Preflight passed. This script did not create, stop, or delete any resource."
