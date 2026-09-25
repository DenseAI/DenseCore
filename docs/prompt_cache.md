# Prompt Cache and Affinity

DenseCore's Go server computes prompt-cache identities and routing affinity for
repeated prefixes. The cache contract is process-local and safety-first: a reuse
decision must miss when runtime state cannot be restored exactly.

## Configuration

```bash
export DENSECORE_AGENT_PROMPT_CACHE=auto
export DENSECORE_AGENT_PROMPT_CACHE_MAX_BYTES=4294967296
export DENSECORE_AGENT_PROMPT_CACHE_MAX_SESSIONS=1024
export DENSECORE_AGENT_PROMPT_CACHE_TTL_SECONDS=3600
export DENSECORE_AGENT_PROMPT_CACHE_DEBUG=0
```

Modes:

- `off`: disable cache decisions
- `auto`: enable only for supported and safe runtime identities
- `on`: request cache use for technically supported identities; unsafe reuse still
  fails closed

Cache identity includes the model fingerprint, tokenizer and template, rendered
tool schema, system prompt, LoRA adapter, KV dtype, RoPE/graph identity,
sliding-window policy, SSM policy, parser family, and token IDs.

## Safety Boundary

- Token identity is verified before reuse.
- Changes to tools, template, LoRA, model, KV, graph family, or state policy miss.
- Qwen3.5 hybrid SSM prefix reuse is currently disabled pending qualification.
- Qwen3.6 hybrid SSM reuse is opt-in and requires a matching SSM snapshot boundary.
- Qwen3.8 hybrid SSM prefix reuse is disabled until real-server C4/C4A
  qualification proves snapshot restore correctness for the skipped-MTP trunk.
- Gemma-style sliding/shared-KV reuse remains disabled where restore semantics are
  not exposed.

The Go server can make and observe safe cache decisions, but not every C++ model
family exposes the restore hook needed to reuse KV/SSM state. `on` does not bypass
that limitation.

Runtime optimization status reports effective model eligibility, not just the
global toggle values. Snapshot restore is active only for an eligible recurrent
model with prefix reuse and snapshot restore both enabled. The Go admission
policy holds requests through completion only for that snapshot lifecycle;
otherwise it allows overlapping requests while still serializing native request
submission. Disabling prefix reuse therefore does not require a second snapshot
toggle to allow continuous batching.

## Kubernetes Affinity

Warm KV blocks live in the process that handled the earlier request. Round-robin
routing across replicas therefore turns repeat prompts into cold prefills.

Request and response contract:

- request: `X-DenseCore-Cache-Affinity`
- response: `X-DenseCore-Cache-Affinity-Key`
- response source: `X-DenseCore-Cache-Affinity-Source`
- JSON field: `cache_control.affinity_key`

```json
{
  "messages": [{"role": "user", "content": "Use this document..."}],
  "cache_control": {
    "conversation_id": "conv-42",
    "cache_id": "sha256:document-prefix",
    "affinity_key": "tenant-a:document-prefix"
  }
}
```

Selection order:

1. `X-DenseCore-Cache-Affinity`
2. `cache_control.affinity_key`
3. `cache_control.cache_id`
4. `cache_control.conversation_id`
5. best-effort prompt fingerprint

DenseCore hashes the selected value before returning it. Use a stable, non-secret
tenant/document/conversation identifier. Configure the ingress or gateway to hash
the request header before forwarding the request. A response header can guide the
next request, not reroute the current one.

## Metrics

Metrics use the configured namespace. Default names include:

- `densecore_prompt_cache_hit_total`
- `densecore_prompt_cache_miss_total`
- `densecore_prompt_cache_tokens_reused_total`
- `densecore_ssm_snapshot_miss_total`
- `densecore_cache_affinity_key_total`
- `densecore_cache_affinity_explicit_total`
- `densecore_cache_affinity_fallback_total`

Validate emitted names on the deployed binary before installing alerts.
