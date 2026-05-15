# Tool Calling

Example:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-35b-a3b",
    "messages": [{"role": "user", "content": "Read src/main.cpp"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "read_file",
        "description": "Read a file",
        "parameters": {
          "type": "object",
          "properties": {"path": {"type": "string"}},
          "required": ["path"]
        }
      }
    }],
    "tool_choice": "auto"
  }'
```

`tool_choice` supports `auto`, `none`, `required`, and forced function selection:

```json
{"type":"function","function":{"name":"read_file"}}
```

Invalid tool names, undeclared tools, invalid JSON arguments, and missing required arguments are rejected instead of emitted as broken `tool_calls`.

Streaming tool calls are emitted as OpenAI-style `delta.tool_calls` chunks and finish with `finish_reason: "tool_calls"` before `data: [DONE]`.

