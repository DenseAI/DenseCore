# Model artifacts and qualification

A pin identifies bytes. It is separate from implementation, real-model quality,
performance and automatic feature activation. The current v0.2 work does not
promote MTP or NUMA defaults from fixture tests. Historical v0.1 measurements are
linked from [Benchmarks](BENCHMARKS.md).

| Purpose | Repository and full revision | File | SHA-256 | v0.2 qualification |
|---|---|---|---|---|
| Default CLI, downloader and quick start | `lmstudio-community/Qwen3.5-0.8B-GGUF@7925ccdc665d4efdb1034791e6b553e11128e6f8` | `Qwen3.5-0.8B-Q4_K_M.gguf` | `f5b14da98939b60bbe1019a964eba656407e1e0b64f1fe3003ff6d650e93bfec` | Final integrated image/pod qualification pending |
| Qwen3.5-2B baseline | `unsloth/Qwen3.5-2B-GGUF@f6d5376be1edb4d416d56da11e5397a961aca8ae` | `Qwen3.5-2B-Q4_K_M.gguf` | `aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223` | Local bytes match the public pin; integrated final-answer qualification pending |
| Qwen3 small dense baseline | `Qwen/Qwen3-0.6B-GGUF@23749fefcc72300e3a2ad315e1317431b06b590a` | `Qwen3-0.6B-Q8_0.gguf` | `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031` | Day 0 local baseline; integrated final-answer qualification pending |
| Embedding baseline | `Qwen/Qwen3-Embedding-0.6B-GGUF@370f27d7550e0def9b39c1f16d3fbaa13aa67728` | `Qwen3-Embedding-0.6B-Q8_0.gguf` | `06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439` | Day 0 explicit-LAST baseline; AUTO qualification pending |
| Qwen3.5-0.8B MTP target | `unsloth/Qwen3.5-0.8B-MTP-GGUF@cf8a611f6ed2c2060046219a19f12cd3d5ecd67c` | `Qwen3.5-0.8B-Q4_K_M.gguf` | `ac7c9d7a1b3e3695bb3bd50f8ceaa97f9c93e99ccc3d3d1a620301b6dd6d3d86` | Frozen `44d7675` greedy K=1/2/3 c1 and K2 c2/c4 plus HTTP stream/cancel passed; post-fix integration and performance pending; no automatic activation |
| Qwen3.5-2B MTP target | `unsloth/Qwen3.5-2B-MTP-GGUF@e05864f8066d874d5f85aaff007ae57a2a7d1efe` | `Qwen3.5-2B-Q4_K_M.gguf` | `18c92228fc1498a02506f64593f7cc8cc02f406f732a217c6ba99ad1254d4f0b` | Pending same-file correctness/performance qualification; no automatic activation |
| Qwen3.6-35B MoE MTP candidate | `unsloth/Qwen3.6-35B-A3B-MTP-GGUF@5bc3e238d916f48a861bac2f8a1990a0e9b7e98d` | `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | `0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b` | Upstream metadata + 16 MiB header range only; full-file SHA/runtime/MoE-head/performance/auto unverified |
| 35B comparison | `ggml-org/Qwen3.6-35B-A3B-GGUF@7878dc9a905c7c9379e5e8d6a1aa9affdd6839c5` | `Qwen3.6-35B-A3B-Q4_K_M.gguf` | `671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7` | Historical native benchmark; new C4/C4A and NUMA qualification pending |
| Gemma E2B text inference target | `unsloth/gemma-4-E2B-it-GGUF@0314792d7f1f7e229411f620751375812bb9faf2` | `gemma-4-E2B-it-Q4_K_M.gguf` | `740185b21d22ceb83a11c3aa62ad5842ef32c70f6096d756bbee85a1e4ec34b8` | Pending integrated real-model qualification |
| Gemma E4B text inference target | `ggml-org/gemma-4-E4B-it-GGUF@a69f018d7678a2b9f5e705f8903d6fb29b9b25dc` | `gemma-4-E4B-it-Q4_0.gguf` | `a555b900214b477d8880e7832e0b8925e139b0159640036b09fe472b6f2097f2` | Pending real-model qualification; pin is not a support claim |
| gpt-oss target | `ggml-org/gpt-oss-20b-GGUF@15dcfc7e7c608c6161507211ed43c5b590801b4b` | `gpt-oss-20b-MXFP4.gguf` | `27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901` | Pending quantized native and real-model qualification |

The MoE MTP candidate is 22,663,387,424 bytes. Its advertised LFS SHA is an
expected pin, not a downloaded-file checksum result. The verified header range
contains `nextn_predict_layers=1` and head tensors at `blk.40`; the full weights
have not been fetched or executed. The separate 35B benchmark artifact does not
qualify this MTP file, and header presence does not qualify automatic activation.

The local Day 0 Qwen3.5-0.8B file had a different SHA (`bd258782…dc517`).
Its result does not qualify the default file above. The Day 0 Gemma E2B fixture
(`ac0069eb…6845`) also differs from the public E2B pin in this table. Native
kernel fixtures do not qualify that public artifact. Local files with unknown
source revision remain local evidence; do not turn their filenames into pins.

Local MTP test files may have an added `-MTP` suffix. The catalog lists the
original repository filenames; the MTP/non-MTP repositories can use identical
filenames for different bytes, so compare SHA-256 and revision rather than names.

The local Qwen3.5-2B fixture was matched byte-for-byte to the public pin above
using the repository file SHA-256 metadata. This resolves its previously unknown
source identity without changing its quality qualification status.

Custom interactive CLI models require explicit integrity:

```bash
./bin/densecore-server run owner/repo --filename model.gguf \
  --revision <full-40-character-commit> --sha256 <64-character-sha256>
```

The cache includes repository and revision. Downloads use a temporary file,
verify SHA-256, and rename only after verification. An existing corrupt cache
returns an error; remove or replace that specific corrupt artifact before retry.
The Kubernetes downloader accepts the same repo/file/revision/checksum fields;
see [its inputs](../docker/downloader/README.md).
