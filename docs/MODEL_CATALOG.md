# Model artifacts and qualification

A pin identifies bytes. It is separate from implementation, real-model quality,
performance and automatic feature activation. The current v0.2 work does not
promote MTP or NUMA defaults from fixture tests. Historical v0.1 measurements are
linked from [Benchmarks](BENCHMARKS.md).

| Purpose | Repository and full revision | File | SHA-256 | v0.2 qualification |
|---|---|---|---|---|
| Default CLI, downloader and quick start | `lmstudio-community/Qwen3.5-0.8B-GGUF@7925ccdc665d4efdb1034791e6b553e11128e6f8` | `Qwen3.5-0.8B-Q4_K_M.gguf` | `f5b14da98939b60bbe1019a964eba656407e1e0b64f1fe3003ff6d650e93bfec` | Final integrated image/pod qualification pending |
| 35B comparison | `ggml-org/Qwen3.6-35B-A3B-GGUF@7878dc9a905c7c9379e5e8d6a1aa9affdd6839c5` | `Qwen3.6-35B-A3B-Q4_K_M.gguf` | `671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7` | Historical native benchmark; new C4/C4A and NUMA qualification pending |
| Gemma E2B text inference target | `unsloth/gemma-4-E2B-it-GGUF@0314792d7f1f7e229411f620751375812bb9faf2` | `gemma-4-E2B-it-Q4_K_M.gguf` | `740185b21d22ceb83a11c3aa62ad5842ef32c70f6096d756bbee85a1e4ec34b8` | Pending integrated real-model qualification |
| Gemma E4B text inference target | `ggml-org/gemma-4-E4B-it-GGUF@a69f018d7678a2b9f5e705f8903d6fb29b9b25dc` | `gemma-4-E4B-it-Q4_0.gguf` | `a555b900214b477d8880e7832e0b8925e139b0159640036b09fe472b6f2097f2` | Pending real-model qualification; pin is not a support claim |
| gpt-oss target | `ggml-org/gpt-oss-20b-GGUF@15dcfc7e7c608c6161507211ed43c5b590801b4b` | `gpt-oss-20b-MXFP4.gguf` | `27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901` | Pending quantized native and real-model qualification |

The local Day 0 Qwen3.5-0.8B file had a different SHA (`bd258782…dc517`).
Its result does not qualify the default file above. The Day 0 Gemma E2B fixture
(`ac0069eb…6845`) also differs from the public E2B pin in this table. Native
kernel fixtures do not qualify that public artifact. Local files with unknown
source revision remain local evidence; do not turn their filenames into pins.

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
