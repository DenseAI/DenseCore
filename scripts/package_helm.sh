#!/bin/bash
set -e

# =============================================================================
# DenseCore Helm Packager
# Usage: ./scripts/package_helm.sh [REPO_URL]
# Example: ./scripts/package_helm.sh https://densecore.github.io/charts
# =============================================================================

CHART_DIR="charts/densecore"
OUTPUT_DIR="charts/repo"
REPO_URL=$1

echo "========================================================"
echo "DenseCore Helm Packager"
echo "Chart: ${CHART_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "========================================================"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# 1. Lint
echo ""
echo "[1/3] Linting Chart..."
helm lint "${CHART_DIR}"

# 2. Package
echo ""
echo "[2/3] Packaging Chart..."
helm package "${CHART_DIR}" --destination "${OUTPUT_DIR}"

# 3. Index (Optional)
if [ -n "${REPO_URL}" ]; then
  echo ""
  echo "[3/3] generating index.yaml with URL: ${REPO_URL}..."
  helm repo index "${OUTPUT_DIR}" --url "${REPO_URL}"
else
  echo ""
  echo "[3/3] generating local index.yaml..."
  helm repo index "${OUTPUT_DIR}"
fi

echo ""
echo "========================================================"
echo "SUCCESS! Chart packaged in ${OUTPUT_DIR}"
echo "========================================================"
