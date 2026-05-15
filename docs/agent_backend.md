# DenseCore Agent Backend

DenseCore's OpenAI-compatible Go server accepts agent-style `/v1/chat/completions` requests with `tools`, `tool_choice`, streaming, and optional `cache_control`.

Supported parser families in v0.1:
- `qwen_xml`: Qwen/Qwen-coder XML-ish `<tool_call>` blocks and JSON-in-block calls.
- `hermes_json`: Hermes/OpenAI JSON `tool_calls` output.
- `generic_json`: conservative JSON fallback.

Reasoning blocks wrapped in `<think>...</think>` are separated from visible assistant content. Normal OpenAI-compatible responses do not expose debug cache fields unless server-side debug logging/metrics are enabled.

Known limitation: the Go server now computes verified prompt-cache keys and safe reuse decisions, but the current `domain.Engine` interface does not expose a C++ KV/SSM restore hook. Hybrid SSM models therefore fail closed unless a matching SSM snapshot handoff is added.

