# Contributing to DenseCore

Thank you for your interest in contributing to DenseCore! We welcome contributions from everyone, whether you're fixing bugs, adding features, improving documentation, or suggesting ideas.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Release Documentation](#release-documentation)
- [Community](#community)

---

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. We follow the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/).

**In short:**
- ✅ Be respectful and constructive
- ✅ Welcome newcomers and help them learn
- ✅ Focus on what's best for the community
- ❌ No harassment, trolling, or discrimination

---

## DenseCore Design Philosophy

DenseCore's mission is a **memory-centric inference runtime for heterogeneous AI execution** with **CPU-first design**, **graceful fallback**, and **production-capable serving surfaces**.

When contributing, please keep these principles in mind:

- **CPU-First, Heterogeneous by Design**: Optimize CPU and locality-sensitive paths first. Additional backends are welcome when they preserve fallback correctness.
- **Cloud-Native Serving Surfaces**: Keep the server observable, configurable, and operationally disciplined.
- **Quality and Honesty**: Tests, documentation, explicit feature maturity, and reproducible benchmark claims are expected.

---

## Code Formatting

Format only the files touched by your change while iterating. Repository-wide
`make format` applies C++ clang-format, Python Ruff, and Go golangci-lint fixes;
review its diff before including changes. Check the relevant language with
`make format-check-cpp`, `make format-check-python`, or `make format-check-go`.
The Go targets install the pinned golangci-lint tool if it is missing.

## Getting Started

### Prerequisites

Before you begin, ensure you have:

- **C++:** CMake 3.14+, a C++17 compiler, and Make or Ninja
- **Go:** 1.25.13+ (for server development)
- **Python:** 3.10+ (with pip)
- **Git:** For version control

**System Requirements:**
- Start with Linux amd64 or WSL2 for the v0.1 server path; see [release scope](docs/RELEASE.md) for other platforms.
- 8GB+ RAM (for running tests)
- 10GB+ free disk space

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub first
git clone https://github.com/YOUR_USERNAME/DenseCore.git
cd DenseCore

# Add upstream remote
git remote add upstream https://github.com/DenseAI/DenseCore.git
```

### 2. Initialize Submodules

```bash
git submodule update --init --recursive  # Important: GGML is a submodule!
```

### 3. Build C++ Core

```bash
# Build libdensecore.so
make lib

# Verify build
ls -lh build/libdensecore.so
```

> [!NOTE]
> **Performance Build:** By default, the build uses `-march=native` to optimize for your specific CPU (AVX2/AVX-512). This maximizes performance but the resulting binary may not run on other machines with older CPUs.

**Troubleshooting:**
```bash
# Clean build if needed
make clean
make lib

# Build with debug symbols
cmake -S core -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

Run commands below from the repository root unless a subshell explicitly changes directories.
See the [contributor map](docs/CONTRIBUTOR_MAP.md) for code boundaries, editor setup, and targeted tests.

### 4. Set Up Python Environment (Optional)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e './python[dev]'
python -c "import densecore; print('Success!')"
```

### 5. Build Go Server (Optional)

```bash
# Download dependencies without rewriting module metadata
(cd server && go mod download)

# Build server
GOFLAGS=-mod=mod make server

# Set this to an existing supported GGUF file (see README for model guidance).
export MODEL_PATH=/absolute/path/to/model.gguf
LD_LIBRARY_PATH="$PWD/build${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  ./bin/densecore-server serve --model "$MODEL_PATH"
```

The server requires a startup model through `--model` or `MAIN_MODEL_PATH`.
After it reports readiness, send the [README quick-start request](README.md).
The command above uses the Linux shared-library loader; macOS uses `DYLD_LIBRARY_PATH`.

### 6. Validate the Runtime Image with Docker

The final image is a distroless server image: it has no shell or development toolchain.
Build and run it to check packaging, using an existing model directory:

```bash
docker build -t densecore-local .
export MODEL_DIR=/absolute/path/to/models
docker run --rm -p 127.0.0.1:8080:8080 \
  -v "$MODEL_DIR:/models:ro" \
  densecore-local --model /models/model.gguf
```

The image already has `densecore-server serve` as its entrypoint. The mounted model
must be readable by the image's non-root user. Use the host setup above for editing
and tests; Docker build stages are packaging infrastructure, not a Python development image.

---

## How to Contribute

### Reporting Bugs

Found a bug? Please [open an issue](https://github.com/DenseAI/DenseCore/issues/new) with:

1. **Clear title** describing the problem
2. **Steps to reproduce:** Include the exact build and server commands, request
   payload or SDK script, model identifier/checksum, and relevant configuration.
3. **Expected behavior** vs. **actual behavior**
4. **Environment:**
   - OS: (e.g., Ubuntu 22.04)
   - Python version: (e.g., 3.10.5)
   - DenseCore version: (e.g., v0.1.0)
5. **Error messages/stack traces** (full output)

### Suggesting Features

Have an idea? [Open an issue](https://github.com/DenseAI/DenseCore/issues/new) with:

1. **Use case:** What problem does this solve?
2. **Proposed solution:** How should it work?
3. **Alternatives:** What have you considered?
4. **Examples:** Code samples of proposed API

### Contributing Code

1. **Check existing issues** to avoid duplicate work
2. **Discuss major changes** in an issue before coding
3. **Keep PRs focused** - one feature/fix per PR
4. **Write tests** for new functionality
5. **Update documentation** if changing public APIs

---

## Pull Request Guidelines

### Before Submitting

**Checklist:**
- [ ] Code follows style guidelines (see below)
- [ ] C++ tests pass after configuring with `DENSECORE_BUILD_TESTS=ON` (see [Testing](#testing))
- [ ] Go tests pass (`cd server && go test -mod=mod ./...`)
- [ ] New features have tests
- [ ] Documentation is updated
- [ ] `docs/RELEASE.md` is updated for public scope/API changes
- [ ] Commit messages are clear
- [ ] No merge conflicts with `main`

### PR Process

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/add-cool-feature
   ```

2. **Make your changes:**
   ```bash
   # Edit files
   git add path/to/changed-file
   git commit -m "Explain why this change is needed"
   ```

3. **Keep branch updated:**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

4. **Push and create PR:**
   ```bash
   git push origin feature/add-cool-feature
   ```
   Then open a PR on GitHub.

5. **Respond to review feedback:**
   - Make requested changes
   - Push updates to the same branch
   - PR will update automatically

### Commit Message Format

Use the Lore protocol in `AGENTS.md`: an intent line explaining why the change is
needed, followed by context and useful git-native trailers. For example:

```text
Preserve terminal callbacks when a request is cancelled

Keep cancellation and completion on the same finalization path.

Confidence: high
Scope-risk: narrow
Tested: RequestLifecycleTest.*
Not-tested: Real-model load under concurrent shutdown
```

Include `Constraint:`, `Rejected:`, or `Directive:` when they capture decisions a
future contributor should preserve. Record only verification actually performed.

---

## Code Style

### C++ Style

Follow [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) with exceptions:

- **Naming:**
  - Classes: `PascalCase` (e.g., `InferenceEngine`)
  - Functions: `PascalCase` (e.g., `BuildTransformerGraph()`)
  - Variables: `snake_case` (e.g., `n_tokens`)
  - Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_BATCH_SIZE`)

- **Formatting:**
  ```cpp
  // Use clang-format
  clang-format -i core/src/*.cpp
  ```

- **Comments:**
  ```cpp
  /**
   * @brief Brief description
   * @param name Parameter description
   * @return Return value description
   */
  int MyFunction(const std::string& name);
  ```

### Python Style

Follow [PEP 8](https://pep8.org/):

```bash
# Auto-format with Ruff
ruff format python/

# Lint with ruff
ruff check python/

# Type check with mypy
mypy python/densecore/
```

**Key points:**
- Line length: 100 characters
- Type hints required for public APIs
- Docstrings for all public functions

**Example:**
```python
from typing import List, Optional

def generate(
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.8,
) -> str:
    """Generate text completion.

    Args:
        prompt: Input text prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 = deterministic)

    Returns:
        Generated text string

    Raises:
        InferenceError: If generation fails
    """
    ...
```

### Go Style

Follow [Effective Go](https://go.dev/doc/effective_go):

```bash
# Auto-format
make format

# Lint
make lint-go
```

---

## Testing

### Running Tests

**C++ tests (opt-in; `make lib` alone does not enable them):**
```bash
cmake -S core -B build-tests -DCMAKE_BUILD_TYPE=Release \
  -DDENSECORE_BUILD_TESTS=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build-tests --target densecore_tests -j 4
ctest --test-dir build-tests --output-on-failure
```

Configuration fetches CMake dependencies, including GoogleTest. Use a Makefiles or
Ninja generator to produce `build-tests/compile_commands.json`. See the
[contributor map](docs/CONTRIBUTOR_MAP.md) for smaller test selections; run those
while iterating and the relevant full suites before submitting.

**Python tests (after the editable install above):**
```bash
(cd python && python -m pytest)
(cd python && python -m pytest tests/test_chat_template.py)
```

**Go tests (requires `make lib` for the CGO engine):**
```bash
(cd server && go test -mod=mod ./...)
(cd server && go test -mod=mod ./internal/api -run '^TestSSEStreamWriter' -count=1)
```

`make test` sends an HTTP request to a running server; it is not the unit-test suite.

### Writing Tests

Use the nearest existing test as the template:

- Python configuration and mocks: `python/tests/test_densecore.py`.
- C++ GoogleTest fixtures and request completion: `core/tests/runtime/test_request_lifecycle.cpp`.
- HTTP handlers and streaming mocks: `server/internal/api/handlers_test.go`.

Add a focused case that fails before your fix. C++ sources must be listed in
`TEST_SOURCES` in `core/CMakeLists.txt` if you introduce a new test file. Mock-based
unit tests do not replace real-model API validation for inference changes.

---

## Documentation

### Updating Docs

When making changes that affect users:

1. **Update relevant `.md` files** in `docs/` and `python/README.md`
2. **Update API contract docs** in `docs/API_REFERENCE.md` and `server/openapi.yaml`
3. **Update `CHANGELOG.md`**
4. **Update `docs/RELEASE.md`** when release scope or qualification changes
5. **Update Python docstrings**
6. **Add examples** for new features

### Documentation Structure

```
DenseCore/
├── README.md                 # Main entry point
├── docs/RELEASE.md           # Release scope and qualification gates
├── CONTRIBUTING.md          # This file
├── docs/
│   ├── README.md           # Docs index
│   ├── ARCHITECTURE.md     # System design
│   ├── API_REFERENCE.md    # Complete API
│   ├── DEPLOYMENT.md       # Docker/K8s
│   ├── PERFORMANCE_TUNING.md
│   └── ...
└── python/README.md         # Python SDK guide
```

### Writing Good Docs

**Do:**
- ✅ Provide working code examples
- ✅ Explain *why*, not just *what*
- ✅ Use clear, simple language
- ✅ Include troubleshooting tips
**Don't:**
- ❌ Assume prior knowledge
- ❌ Use jargon without explanation
- ❌ Provide incomplete examples
- ❌ Forget to update after code changes

---

## Release Documentation

- For public release naming, qualification gates, and surface-level scope, update
  and follow [`docs/RELEASE.md`](docs/RELEASE.md).
- Keep server-first claims and non-core surface status (Python/Helm) consistent
  across README, docs index, and API reference docs.

---

## Architecture Overview

Understanding DenseCore's structure helps with contributions:

```
┌─────────────────────────────────────────┐
│ User Layer (Python/Go/CLI)              │
│  - Python SDK (ctypes)                  │
│  - Go REST Server (CGO)                 │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│ Inference Engine (C++)                  │
│  - Request Scheduler                    │
│  - Inference Worker                     │
│  - KV Cache Manager                     │
│  - Model Loader                         │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│ Compute Backend (GGML)                  │
│  - Tensor Operations                    │
│  - SIMD Kernels (AVX2/AVX-512)          │
│  - Quantization (INT4/INT8)             │
└─────────────────────────────────────────┘
```

**Key files:**

- `core/src/runtime/engine.cpp` - Model load, KV/scheduler sizing, C API entry points
- `core/src/runtime/worker.cpp` - Execution loop, graph pool, batching, prefill chunking
- `core/src/runtime/memory/kv_cache.cpp` - Paged KV cache management
- `python/densecore/__init__.py` - Python API
- `server/cmd/densecore/main.go` - Go CLI and server entry point

See the [contributor map](docs/CONTRIBUTOR_MAP.md) for ownership boundaries and tests,
and [ARCHITECTURE.md](docs/ARCHITECTURE.md) for system context.

---

## Community

### Communication Channels

- **GitHub Issues:** Bug reports, feature requests
- **GitHub Discussions:** Q&A, ideas, general discussion

### Getting Help

**Before asking:**
1. Check [documentation](docs/README.md)
2. Search [existing issues](https://github.com/DenseAI/DenseCore/issues)
3. Read [troubleshooting guides](python/README.md#troubleshooting)

**When asking:**
- Provide context and code samples
- Include error messages
- Specify your environment

---

## Recognition

Contributors are recognized in:
- [CHANGELOG.md](CHANGELOG.md) for each release
- GitHub's contributor graph
- Special mention for significant contributions

---

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).

## Questions?

Still have questions? Feel free to:
- Open an issue asking for clarification
- Start a discussion on GitHub
- Reach out to maintainers

**Thank you for contributing to DenseCore! 🚀**
