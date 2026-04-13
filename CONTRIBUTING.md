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

**Before committing, run:**

```bash
make format  # Auto-fix formatting for C++, Python, Go
```

This runs:
- `clang-format` for C++
- `ruff format` + `ruff check --fix` for Python
- `golangci-lint fmt` + `golangci-lint run --fix` for Go

**CI uses the same `make` targets for formatting and lint checks.** Run `make format` before pushing.

## Getting Started

### Prerequisites

Before you begin, ensure you have:

- **C++:** CMake 3.14+, GCC 9+/Clang 10+ (C++17 support)
- **Go:** 1.22+ (for server development)
- **Python:** 3.8+ (with pip)
- **Git:** For version control

**System Requirements:**
- Linux/macOS/Windows with WSL
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
git remote add upstream https://github.com/DenseCore/DenseCore.git
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
cmake -DCMAKE_BUILD_TYPE=Debug core/
make
```

### 4. Set Up Python Environment

```bash
cd python

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

**Verify installation:**
```python
python -c "import densecore; print('Success!')"
```

### 5. Build Go Server (Optional)

```bash
cd server

# Download dependencies
go mod download
go mod tidy

# Build server
make server

# Run server
./bin/densecore-server serve
```

### 6. Development with Docker

To avoid complex local setup, you can use our development Docker image:

```bash
# Build the dev environment
docker build -t densecore-dev -f Dockerfile .

# Run and mount your code
docker run -it --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  densecore-dev bash

# Inside the container, you have all tools (go, cmake, python) pre-installed:
# $ make lib
# $ make server
```

---

## How to Contribute

### Reporting Bugs

Found a bug? Please [open an issue](https://github.com/DenseCore/DenseCore/issues/new) with:

1. **Clear title** describing the problem
2. **Steps to reproduce:**
   ```python
   # Example code that triggers the bug
   import densecore
   model = densecore.from_pretrained("model-repo")
   model.generate("...")  # Crashes here
   ```
3. **Expected behavior** vs. **actual behavior**
4. **Environment:**
   - OS: (e.g., Ubuntu 22.04)
   - Python version: (e.g., 3.10.5)
   - DenseCore version: (e.g., 1.0.0)
5. **Error messages/stack traces** (full output)

### Suggesting Features

Have an idea? [Open an issue](https://github.com/DenseCore/DenseCore/issues/new) with:

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
- [ ] Tests pass (`make test`)
- [ ] New features have tests
- [ ] Documentation is updated
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
   git add .
   git commit -m "feat: add cool feature"
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

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style (formatting, no logic change)
- `refactor`: Code restructuring (no feature/bug change)
- `perf`: Performance improvement
- `test`: Adding/updating tests
- `chore`: Maintenance (dependencies, build, etc.)

**Examples:**
```bash
git commit -m "feat(python): add batch embedding support"
git commit -m "fix(core): resolve memory leak in KV cache"
git commit -m "docs(readme): update installation instructions"
```

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
# Auto-format with black
black python/

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

**C++ Tests:**
```bash
cd core/build
# Run all tests via CTest
ctest --output-on-failure
```

**Python Tests:**
```bash
cd python
pytest                    # All tests
pytest tests/test_api.py  # Specific file
pytest -v                 # Verbose
pytest -k "test_generate" # Specific test
```

**Go Tests:**
```bash
cd server
go test ./...
go test -v internal/handlers  # Verbose
```

### Writing Tests

**Python example:**
```python
# tests/test_generation.py
import pytest
from densecore import DenseCore

def test_basic_generation():
    """Test basic text generation works."""
    model = DenseCore("test_model.gguf")
    response = model.generate("Hello", max_tokens=10)

    assert isinstance(response, str)
    assert len(response) > 0

@pytest.mark.asyncio
async def test_async_streaming():
    """Test async streaming generation."""
    model = DenseCore("test_model.gguf")
    tokens = []

    async for token in model.stream_async("Hello"):
        tokens.append(token)

    assert len(tokens) > 0
```

**C++ example:**
```cpp
// core/tests/test_kv_cache.cpp
#include "kv_cache.h"
#include <cassert>

void test_allocate_blocks() {
    PagedKVCache cache(1000);  // 1000 blocks

    auto blocks = cache.AllocateBlocks(123, 32);  // seq_id=123, 32 tokens
    assert(blocks.size() == 2);  // 32 tokens / 16 per block = 2 blocks

    cache.ReleaseSequence(123);
    assert(cache.GetUsedBlocks() == 0);
}

int main() {
    test_allocate_blocks();
    return 0;
}
```

---

## Documentation

### Updating Docs

When making changes that affect users:

1. **Update relevant .md files** in `docs/`
2. **Update Python docstrings**
3. **Add examples** for new features
4. **Update CHANGELOG.md**

### Documentation Structure

```
DenseCore/
├── README.md                 # Main entry point
├── CONTRIBUTING.md          # This file
├── docs/
│   ├── README.md           # Docs index
│   ├── ARCHITECTURE.md     # System design
│   ├── API_REFERENCE.md    # Complete API
│   ├── DEPLOYMENT.md       # Docker/K8s
│   ├── MODEL_OPTIMIZATION.md
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

## Doxygen Documentation

DenseCore uses Doxygen to generate C++ API documentation from code comments.

### Generating C++ API Docs

```bash
# Install Doxygen
sudo apt-get install doxygen graphviz  # Ubuntu/Debian
brew install doxygen graphviz          # macOS

# Generate documentation
cd /path/to/DenseCore
doxygen Doxyfile

# View generated docs
open docs/html/index.html  # macOS
xdg-open docs/html/index.html  # Linux
```

This generates comprehensive HTML documentation for all C++ APIs in `docs/html/`.

### Writing Doxygen Comments

Use Javadoc-style comments in header files:

```cpp
/**
 * @brief Generate text completion
 *
 * This function submits a text generation request to the inference engine
 * and invokes the callback for each generated token.
 *
 * @param handle Handle to the DenseCore engine
 * @param prompt Input text prompt (null-terminated UTF-8)
 * @param max_tokens Maximum number of tokens to generate
 * @param callback Function to call for each token
 * @param user_data User pointer passed to callback
 * @return Request ID (positive) on success, negative error code on failure
 *
 * @note The callback may be invoked from a different thread
 * @see SubmitRequestIds for pre-tokenized input
 *
 * @code
 * void my_callback(const char* token, int finished, void* data) {
 *     printf("%s", token);
 * }
 *
 * int req_id = SubmitRequest(engine, "Hello", 100, my_callback, NULL);
 * @endcode
 */
int SubmitRequest(DenseCoreHandle handle, const char* prompt,
                  int max_tokens, TokenCallback callback, void* user_data);
```

**Supported tags:**
- `@brief` - Short description
- `@param` - Parameter description
- `@return` - Return value description
- `@note` - Additional notes
- `@see` - Cross-reference to related functions
- `@code/@endcode` - Code examples

### Doxygen Configuration

The `Doxyfile` configures:
- **Input:** `core/include/` (public headers)
- **Output:** `docs/html/` (HTML documentation)
- **Features:** Call graphs, dependency diagrams, search
- **Style:** Modern, responsive theme

To modify settings, edit `Doxyfile` and regenerate.

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
- `core/src/inference.cpp` - Main inference loop
- `core/src/worker.cpp` - Request scheduling
- `core/src/kv_cache.cpp` - Memory management
- `python/densecore/__init__.py` - Python API
- `server/main.go` - HTTP server

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

---

## Community

### Communication Channels

- **GitHub Issues:** Bug reports, feature requests
- **GitHub Discussions:** Q&A, ideas, general discussion
- **Discord:** Real-time chat (coming soon)
- **Twitter:** [@densecore](https://twitter.com/densecore) - Updates

### Getting Help

**Before asking:**
1. Check [documentation](docs/README.md)
2. Search [existing issues](https://github.com/DenseCore/DenseCore/issues)
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
