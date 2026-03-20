#!/bin/bash
set -e

# =============================================================================
# DenseCore Build & Push Script
# Usage: ./scripts/build_and_push.sh [VERSION]
# Example: ./scripts/build_and_push.sh 0.3.0
# =============================================================================

VERSION=${1:-latest}
DOCKER_ORG="densecore"
REPO_MAIN="${DOCKER_ORG}/densecore"
REPO_DOWNLOADER="${DOCKER_ORG}/downloader"

echo "========================================================"
echo "DenseCore Release Builder"
echo "Version: ${VERSION}"
echo "Organization: ${DOCKER_ORG}"
echo "========================================================"

# Check if docker is running
if ! docker info > /dev/null 2>&1; then
  echo "Error: Docker is not running."
  exit 1
fi

# 1. Build Main Application Image
echo ""
echo "[1/4] Building Main Image (${REPO_MAIN})..."
docker build \
  -t "${REPO_MAIN}:${VERSION}" \
  -t "${REPO_MAIN}:latest" \
  .

# 2. Build Downloader Image
echo ""
echo "[2/4] Building Downloader Image (${REPO_DOWNLOADER})..."
# Context is docker/downloader/ to keep downloader artifacts separate from Helm/K8s config.
docker build \
  -f docker/downloader/Dockerfile \
  -t "${REPO_DOWNLOADER}:${VERSION}" \
  -t "${REPO_DOWNLOADER}:latest" \
  docker/downloader/

# 3. Push Main Image
echo ""
echo "[3/4] Pushing Main Image..."
docker push "${REPO_MAIN}:${VERSION}"
if [ "${VERSION}" != "latest" ]; then
  docker push "${REPO_MAIN}:latest"
fi

# 4. Push Downloader Image
echo ""
echo "[4/4] Pushing Downloader Image..."
docker push "${REPO_DOWNLOADER}:${VERSION}"
if [ "${VERSION}" != "latest" ]; then
  docker push "${REPO_DOWNLOADER}:latest"
fi

echo ""
echo "========================================================"
echo "SUCCESS! Images published:"
echo "  - ${REPO_MAIN}:${VERSION}"
echo "  - ${REPO_DOWNLOADER}:${VERSION}"
echo "========================================================"
