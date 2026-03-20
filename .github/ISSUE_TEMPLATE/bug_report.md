---
name: Bug Report
about: Create a report to help us improve DenseCore
title: '[BUG] '
labels: bug
assignees: ''
---

## Bug Description

<!-- A clear and concise description of what the bug is -->

## Environment

### Hardware
- **CPU**: [e.g., Intel i7-10870H, Apple M3 Max, AWS Graviton3]
- **CPU Features**: [e.g., AVX2, AVX-512, NEON, AMX]
- **RAM**: [e.g., 32GB]
- **Platform**: [e.g., x86_64, arm64, Apple Silicon]

### Software
- **OS**: [e.g., Ubuntu 22.04, macOS 14.0, Windows 11 WSL2]
- **DenseCore Version**: [e.g., 2.0.0]
- **Python Version** (if applicable): [e.g., 3.11.5]
- **Go Version** (if applicable): [e.g., 1.24]
- **Docker Version** (if applicable): [e.g., 24.0.0]

### Model (if applicable)
- **Model Name**: [e.g., Qwen/Qwen3-0.6B-GGUF]
- **Quantization**: [e.g., Q4_K_M, Q8_0]
- **Model Size**: [e.g., 600MB]

## To Reproduce

Steps to reproduce the behavior:

1.
2.
3.
4.

### Minimal Code Example

```python
# Minimum code to reproduce the bug
import densecore

# ...
```

## Expected Behavior

<!-- A clear and concise description of what you expected to happen -->

## Actual Behavior

<!-- A clear and concise description of what actually happened -->

## Logs and Stack Traces

<!-- Paste relevant logs and stack traces here -->

```
Paste logs here
```

## Component Affected

<!-- Check all that apply -->

- [ ] C++ Core (`core/`)
- [ ] Python SDK (`python/`)
- [ ] Go Server (`server/`)
- [ ] CLI (`cli/`)
- [ ] Docker/Kubernetes
- [ ] Documentation

## Severity Assessment

<!-- How severe is this bug for your use case? -->

- [ ] 🔴 **Critical**: System crash, data loss, security issue
- [ ] 🟠 **High**: Feature broken, no workaround
- [ ] 🟡 **Medium**: Feature broken, workaround exists
- [ ] 🟢 **Low**: Minor inconvenience

## Screenshots

<!-- If applicable, add screenshots to help explain your problem -->

## Possible Solution

<!-- Optional: suggest a fix or reason for the bug -->

## Additional Context

<!-- Add any other context about the problem here -->
