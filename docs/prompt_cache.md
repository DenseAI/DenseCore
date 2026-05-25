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

## Kubernetes cache affinity

DenseCore KV prefix reuse is intentionally pod-local. In a Kubernetes deployment,
round-robin routing across replicas will turn many repeated prompts into cold
prefills because the warmed KV blocks live only in the pod that processed the
earlier request.

DenseCore now exposes a routing contract for ingress controllers and API
gateways:

- Request header: `X-DenseCore-Cache-Affinity`
- Response header: `X-DenseCore-Cache-Affinity-Key`
- Response source header: `X-DenseCore-Cache-Affinity-Source`
- JSON request field: `cache_control.affinity_key`

Use a stable, non-secret value such as a tenant/document digest, conversation
ID, or explicit cache ID:

```json
{
  "model": "qwen",
  "messages": [{"role": "user", "content": "Use this document..."}],
  "cache_control": {
    "conversation_id": "conv-42",
    "cache_id": "sha256:document-prefix",
    "affinity_key": "tenant-a:document-prefix"
  }
}
```

DenseCore hashes the selected value before returning it in response headers, so
the header is stable but opaque. Selection order is:

1. `X-DenseCore-Cache-Affinity`
2. `cache_control.affinity_key`
3. `cache_control.cache_id`
4. `cache_control.conversation_id`
5. a best-effort prompt fingerprint fallback

For Kubernetes, configure the gateway to hash on
`X-DenseCore-Cache-Affinity` before it forwards the request. Clients can either
send that header directly, or send `cache_control.*` and replay the returned
`X-DenseCore-Cache-Affinity-Key` on follow-up requests. The fallback fingerprint
is useful for observability, but production multi-turn or file-context serving
should provide an explicit affinity key because an ingress cannot route on a
response header for the current request.

Prometheus counters:

- `densecore_cache_affinity_key_total`
- `densecore_cache_affinity_explicit_total`
- `densecore_cache_affinity_fallback_total`
