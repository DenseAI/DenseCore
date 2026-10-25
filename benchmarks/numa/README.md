# Physical NUMA comparison harness

`physical_preflight.sh` contains the reviewed physical-memory probe from the
historical private Tier-0 harness. It verifies bound-page placement and measures
local/remote STREAM bandwidth and pointer-chase latency. A declared distance
matrix alone does not pass. It requires two NUMA nodes, `numactl`, a C compiler,
OpenMP and libnuma headers. It creates no cloud resources or DenseCore builds.

`run_abba.py` runs sticky OFF/ON and a preselected best llama NUMA mode on 16 or
48 physical cores. It selects one allowed logical CPU per physical core,
balances the selected cores across nodes, pins every process to that set, verifies
the model bytes once, records binary hashes/commands/configuration, and preserves
five ABBA blocks per comparison. The default maximum output is 1024 tokens.
`llama_mtp: true` adds `--spec-type draft-mtp --spec-draft-n-max 2` (or the
predeclared `draft_n_max`). Use the same GGUF file for both engines.

Example configuration (fill in actual paths, checksum and final-answer test):

```json
{
  "model": "/data/models/model.gguf",
  "sha256": "<verified SHA-256>",
  "dense_command": ["/data/DenseCore/bin/densecore-server", "serve", "--grpc=false"],
  "llama_command": ["/data/llama.cpp/build/bin/llama-server"],
  "llama_numa": "distribute",
  "llama_mtp": false,
  "context": 8192,
  "dense_build_cache": "/data/DenseCore/build/CMakeCache.txt",
  "max_tokens": 1024,
  "prompts": ["Explain binary search and provide a complete Python function named binary_search."],
  "expected_pattern": "def binary_search",
  "dense_env": {"LD_LIBRARY_PATH": "/data/DenseCore/build"},
  "llama_env": {"LD_LIBRARY_PATH": "/data/llama.cpp/build/bin"}
}
```

```bash
python3 benchmarks/numa/run_abba.py session.json --threads 16 --out /data/results/numa16
python3 benchmarks/numa/run_abba.py session.json --threads 48 --out /data/results/numa48
```

Two existing dispatch policies must be controlled together. The native graph
path in `core/src/llm/moe/native.cpp` requires the operational
`DENSECORE_EXPERIMENTAL_MOE_NUMA_STICKY=1` opt-in; it is default OFF. The separate
small-decode backend in `core/src/backend/cpu_backend_moe_small_decode.inl` has a
maintained default ON and honors `DENSECORE_DEBUG_DISABLE_MOE_NUMA_STICKY`.
This is not a newly qualified automatic policy. The harness sets both controls:
OFF uses opt-in `0` plus diagnostic-disable `1`; ON uses `1` plus `0`.
Each case starts a fresh process because the small-decode decision is cached.

The OFF control requires an explicitly configured
`-DDENSECORE_ENABLE_DEBUG_ENV=ON` build. The harness records the supplied CMake
cache hash and rejects an OFF/debug-disabled build before hashing the model.
Configuration strings alone never prove a comparison: every dense sample must
also contain native runtime state and dispatch counters. OFF must report
`disabled_by_debug`, zero sticky operations and positive legacy dispatch; ON
must report `enabled` and actual sticky operations. Missing telemetry, a
single-node/degenerate placement or `disabled_unqualified_model` fails closed.
The native QWEN35 family guard (including the unqualified 122B case) is preserved;
setting the controls cannot bypass it. A path with no sufficient instrumentation
cannot be scored merely because an environment variable was set.

DenseCore and llama must load their own ggml libraries. `dense_env` and
`llama_env` separate loader paths; the baseline defaults to its executable
folder and strips inherited/common DenseCore preload/library settings. Actual
loaded library hashes and process maps are retained with each sample.

Choose the best llama NUMA mode in a separate unscored sweep before freezing the
configuration. Complete final answers, real stream termination, usage and timing
are checked; reasoning-only or truncated output fails. The samples record TTFT,
full completion time and aggregate completed-output throughput. Token identities,
layer logits, MoE/non-MoE breakdown, peak RSS and kernel/fallback coverage require
separate native artifacts. Matching final text is not proof of token parity.
Do not promote a default or publish a +15% claim from this harness alone.

No NUMA or 122B performance result is produced by adding this code. Actual
Qwen3.6-35B and 122B support/promotion remain separate model-specific gates.
