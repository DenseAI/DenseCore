# Agent Prompt Cache

Environment:

```bash
export DENSECORE_AGENT_PROMPT_CACHE=auto
export DENSECORE_AGENT_PROMPT_CACHE_MAX_BYTES=4294967296
export DENSECORE_AGENT_PROMPT_CACHE_MAX_SESSIONS=1024
export DENSECORE_AGENT_PROMPT_CACHE_TTL_SECONDS=3600
export DENSECORE_AGENT_PROMPT_CACHE_DEBUG=0
```

Modes:
- `off`: disable cache decisions.
- `auto`: enable only for supported and safe runtime identities.
- `on`: enable for technically supported identities, still failing closed on unsafe reuse.

Cache keys include model identity, tokenizer/template identity, rendered tool schema hash, system prompt hash, LoRA adapter, KV dtype, rope/runtime graph identity, sliding-window policy, SSM policy, parser family, and token IDs.

Safety rules:
- Token identity is verified before reuse.
- Tool schema, template, LoRA, model fingerprint, KV dtype, graph family, and SSM policy changes miss.
- Qwen3.5/Qwen3.6 hybrid SSM reuse requires a matching SSM snapshot boundary.
- Gemma-style sliding/shared-KV graph families are disabled until runtime restore semantics are exposed.

Metrics are exposed under the configured metrics namespace, for example `densecore_prompt_cache_hit_total`, `densecore_prompt_cache_miss_total`, `densecore_prompt_cache_tokens_reused_total`, and `densecore_ssm_snapshot_miss_total`.

