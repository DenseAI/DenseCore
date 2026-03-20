# GitHub Actions Workflows for DenseCore

This directory contains CI/CD workflows for automated testing, building, and deployment.

## Workflows

### 🔨 CI Workflow (`ci.yml`)

**Triggers:** Push to `main`/`develop`, Pull Requests

**Jobs:**
- **cpp-build**: Build C++ core library on Ubuntu and macOS (Release/Debug)
- **go-build**: Build Go server and run tests
- **python-build**: Test Python SDK on Python 3.8-3.12
- **lint**: Code quality checks (ruff, black, clang-format)
- **docker-build**: Test Docker image build
- **security**: Trivy security scanning

### 🐳 Docker Workflow (`docker.yml`)

**Triggers:** Push to `main`, version tags, manual dispatch

**Features:**
- Multi-architecture builds (amd64, arm64)
- Automatic tagging (latest, semantic versioning, git sha)
- Push to Docker Hub
- Vulnerability scanning with Trivy
- Updates Docker Hub description from `DOCKER_DEPLOYMENT.md`

**Required Secrets:**
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

### 🚀 Release Workflow (`release.yml`)

**Triggers:** Version tags (`v*`), manual dispatch

**Jobs:**
- **create-release**: Generate changelog and create GitHub release
- **build-artifacts**: Build binaries for Linux and macOS

**Required Secrets:**
- None (uses `GITHUB_TOKEN` automatically)

### 📦 PyPI Workflow (`pypi-publish.yml`)

**Triggers:** Version tags (`v*`), manual dispatch

**Features:**
- Multi-platform wheels via `cibuildwheel`:
  - Linux x86_64 (manylinux)
  - Linux ARM64 (AWS Graviton, manylinux_aarch64)
  - macOS x86_64 (Intel)
  - macOS ARM64 (Apple Silicon)
- Python 3.10, 3.11, 3.12
- Trusted Publishing (no API token needed)
- Manual target selection:
  - `testpypi`
  - `pypi`
  - `both`

**Required Secrets:**
- None (uses PyPI Trusted Publishing)

### ⎈ Helm Workflow (`helm-publish.yml`)

**Triggers:** Version tags (`v*`), manual dispatch

**Features:**
- Automatic version extraction from git tag
- Helm lint and template validation
- Publish to GitHub Container Registry (OCI)

**Usage:**
```bash
helm install densecore oci://ghcr.io/jake-network/charts/densecore --version 0.3.0
```

**Required Secrets:**
- None (uses `GITHUB_TOKEN` automatically)

### ✅ Code Quality Workflow (`lint.yml`)

**Triggers:** Pull Requests, Push to `main`/`develop`

**Linters:**
- **Python**: ruff, black, isort, mypy
- **Go**: golangci-lint
- **C++**: clang-format, cppcheck
- **Markdown**:  markdownlint
- **YAML**: yamllint
- **Dockerfile**: hadolint

## Setting Up Secrets

### Required Secrets

Go to **Settings → Secrets and variables → Actions** in your GitHub repository:

1. **DOCKERHUB_USERNAME**: Your Docker Hub username
2. **DOCKERHUB_TOKEN**: Docker Hub access token (create at https://hub.docker.com/settings/security)
### Optional Secrets

- **GITHUB_TOKEN**: Automatically provided by GitHub Actions

## Workflow Status Badges

Add to your `README.md`:

```markdown
[![CI](https://github.com/Jake-Network/DenseCore/actions/workflows/ci.yml/badge.svg)](https://github.com/Jake-Network/DenseCore/actions/workflows/ci.yml)
[![Docker](https://github.com/Jake-Network/DenseCore/actions/workflows/docker.yml/badge.svg)](https://github.com/Jake-Network/DenseCore/actions/workflows/docker.yml)
[![Release](https://github.com/Jake-Network/DenseCore/actions/workflows/release.yml/badge.svg)](https://github.com/Jake-Network/DenseCore/actions/workflows/release.yml)
[![Code Quality](https://github.com/Jake-Network/DenseCore/actions/workflows/lint.yml/badge.svg)](https://github.com/Jake-Network/DenseCore/actions/workflows/lint.yml)
```

## Local Testing

### Run Linters Locally

**Python:**
```bash
cd python
ruff check .
black --check .
mypy densecore/
```

**Go:**
```bash
cd server
golangci-lint run
```

**C++:**
```bash
find core -name "*.cpp" -o -name "*.h" | xargs clang-format -i
cppcheck --enable=all core/
```

**Docker:**
```bash
docker run --rm -i hadolint/hadolint < Dockerfile
```

## Customization

### Modifying CI Matrix

Edit `.github/workflows/ci.yml`:

```yaml
strategy:
  matrix:
    os: [ubuntu-latest, macos-latest, windows-latest]  # Add Windows
    python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']
```

### Changing Docker Platforms

Edit `.github/workflows/docker.yml`:

```yaml
platforms: linux/amd64,linux/arm64,linux/arm/v7  # Add ARM v7
```

### Adding New Linters

1. Add linter to  `lint.yml`
2. Create configuration file (if needed)
3. Update `CONTRIBUTING.md` with linter usage

## Troubleshooting

### CI Failing on Dependency Installation

- Check if dependencies are correctly specified in workflow files
- Ensure `apt-get update` is run before `apt-get install` on Ubuntu

### Docker Build Timeouts

- Increase timeout in workflow file
- Optimize Dockerfile for better layer caching
- Use `cache-from` and `cache-to` with GitHub Actions cache

### Security Scan False Positives

- Add exceptions in Trivy configuration
- Update vulnerable dependencies
- Use `continue-on-error: true` for non-blocking warnings

## Contributing

When modifying workflows:
1. Test locally when possible
2. Create PR and verify all checks pass
3. Document changes in this README
4. Update `CONTRIBUTING.md` if workflow changes affect contributors

---

For more information on GitHub Actions, see [GitHub Actions Documentation](https://docs.github.com/en/actions).
