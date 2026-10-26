# Model artifacts and qualification

A pin identifies bytes. It is separate from implementation, real-model quality,
performance and automatic feature activation. The current v0.2 work does not
promote MTP or NUMA defaults from fixture tests. The integrated source snapshot is
`ef9fc61`; qualification rows below keep their own source boundaries. Historical
`7ecaf86`, private `5ad54a5`, prior Gemma diagnostics and Go-only `386b800` results
remain preserved. Current bounded H c=8 native gates pass; remaining model,
package and performance gates still block broad support promotion. Source `4ad6a35` JSON probes pass for the two pinned MTP files; its unchanged
probabilistic matrix remains overall FAIL and is not inherited by `ef9fc61`. Full gates and local raw ledger names are in
[release gates](V0_2_RELEASE_GATES.md). Historical v0.1 measurements are
linked from [Benchmarks](BENCHMARKS.md).

Current `ef9fc61` E4B short-answer evidence includes unchanged ctx2048/cap4/t8/Q8/NONE
initialization and one complete c1 five-planet/DONE answer (59-token EOS),
plus two exact-ID/complete-answer repeats.
The init plan is 3,468,717,568 <= 3,505,483,776 available bytes; whole-child peak
RSS is 6,240,676 KiB, not a minimum supported RAM estimate.
HTTP/package/support/performance remain pending.
Same-source pinned 0.8B and 2B MTP greedy/native/HTTP requalification passes; older MTP passes/failures below retain their source.
The manifest SHA is `2fc9bc7640a04aad7796c4a8022828170fbbc31351bdb9071cdc9c10c9e0a5a0`.


Current `ef9fc61` E4B c4 now passes two waves of four requests: all eight
retain the own-c1 59-token EOS IDs and complete five-planet/DONE answer.
This is native request isolation, not mixed physical graph or sustained
c4 generation. The pinned 0.8B MTP artifact
`ac7c9d7a1b3e3695bb3bd50f8ceaa97f9c93e99ccc3d3d1a620301b6dd6d3d86`
passes c1/repeat1 and c4/repeat2 greedy NONE/MTP exact-128 native parity,
actual accept/reject/rollback, cancel/recovery and HTTP
stream/nonstream/disconnect checks. It is separate from README's `f5b14d…` file.
Portable-package and fresh-source prefix results remain pending; no performance, automatic
activation, package or broad support qualification follows.

The same pinned 2B MTP file also passes c1/repeat1 and c4/repeat2 native
exact-128 greedy NONE/MTP parity and HTTP/lifecycle checks. These are explicit
MTP checks; automatic profiles remain zero.
A separate README-file native default-policy smoke passes: derived context
23,338, threads 16, concurrency 1, AUTO selects NONE, and the short Paris answer
finishes at 8-token EOS with Free/source/mapping pins. This does not qualify
HTTP, an SDK wheel or a full-23k-context workload.
Offline V3 qualification of the existing sustained E4B c1 capture passes:
254 native IDs, EOS, 216 body words and five ordered three-sentence paragraphs.
The independently reviewed correction permits interior cross-planet references;
original raw FAIL remains immutable, and reevaluation sends zero new requests.
Sustained c4 V3 now passes all eight 254-ID EOS requests against its qualified
own-c1 baseline, retaining five ordered paragraphs, three sentences each and
216 body words. This is native isolation; HTTP/package/support remain separate.
The new four-cell stochastic main strict gate passes 160 HTTP requests with zero
quality failures, 72 native final requests, eight cancellations and eight HTTP
disconnect/recoveries. All 32 stop probes suppress literal markers; 31 finish by
actual string stop. Two separate exact-stop-instruction negatives remain:
0.8B c1 NONE is verbose and 0.8B c4 MTP returns `Paris ` with a trailing space.
Historical `4ad6a35` French-punctuation/stop failures are preserved. Unseeded
cross-run results establish no causal, distribution, speed or automatic claim.

| Purpose | Repository and full revision | File | SHA-256 | v0.2 qualification |
|---|---|---|---|---|
| Default CLI, downloader and quick start | `lmstudio-community/Qwen3.5-0.8B-GGUF@7925ccdc665d4efdb1034791e6b553e11128e6f8` | `Qwen3.5-0.8B-Q4_K_M.gguf` | `f5b14da98939b60bbe1019a964eba656407e1e0b64f1fe3003ff6d650e93bfec` | Historical `7ecaf86` combined prefix checks passed; Go-only `386b800` 4 GiB serving/container+kind strict drain passed. Integrated `143a2ad` Python lifecycle passed but c8 greedy parity failed. Current `ef9fc61` original200 c1/c8 EOS parity and separate80 pressure gates pass within their native profiles; strict Count FAIL belongs to the earlier image, and the final root image is untested |
| Qwen3.5-2B baseline | `unsloth/Qwen3.5-2B-GGUF@f6d5376be1edb4d416d56da11e5397a961aca8ae` | `Qwen3.5-2B-Q4_K_M.gguf` | `aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223` | Historical `7ecaf86` combined prefix/5-turn/cancel/eviction checks passed; current final package/platform qualification pending |
| Qwen3 small dense baseline | `Qwen/Qwen3-0.6B-GGUF@23749fefcc72300e3a2ad315e1317431b06b590a` | `Qwen3-0.6B-Q8_0.gguf` | `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031` | Historical `7ecaf86` native prefix parity and 16 HTTP default/off semantic/protocol cases passed; number-only format gate still fails (also pinned llama) |
| Embedding baseline | `Qwen/Qwen3-Embedding-0.6B-GGUF@370f27d7550e0def9b39c1f16d3fbaa13aa67728` | `Qwen3-Embedding-0.6B-Q8_0.gguf` | `06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439` | Historical `7ecaf86` AUTO/LAST C/Python/HTTP 12 comparisons bitwise equal, finite/nonzero and normalized; later-root package qualification separate |
| Qwen3.5-0.8B MTP target | `unsloth/Qwen3.5-0.8B-MTP-GGUF@cf8a611f6ed2c2060046219a19f12cd3d5ecd67c` | `Qwen3.5-0.8B-Q4_K_M.gguf` | `ac7c9d7a1b3e3695bb3bd50f8ceaa97f9c93e99ccc3d3d1a620301b6dd6d3d86` | Current `ef9fc61` greedy NONE/MTP exact128 native c1/c4 and HTTP/lifecycle PASS. Historical `7ecaf86` greedy checks retained. Integrated `4ad6a35` JSON NONE/MTP c1/c4 native+HTTP/stream/cancel/recovery pass for two object cases. Probabilistic c1 PASS; c4 strict FAIL (French punctuation), plus a separate stop-instruction negative. Historical `48dd55a` adds only the narrow NONE/MTP zero-commit hard-mask lifecycle pass on this file; no new-source JSON/probabilistic or speed/auto/broad support promotion |
| Qwen3.5-2B MTP target | `unsloth/Qwen3.5-2B-MTP-GGUF@e05864f8066d874d5f85aaff007ae57a2a7d1efe` | `Qwen3.5-2B-Q4_K_M.gguf` | `18c92228fc1498a02506f64593f7cc8cc02f406f732a217c6ba99ad1254d4f0b` | Current `ef9fc61` explicit greedy native exact128 c1/c4 and HTTP/lifecycle PASS. Historical `7ecaf86` greedy checks retained. Integrated `4ad6a35` JSON NONE/MTP c1/c4 native+HTTP/stream/cancel/recovery pass for two object cases; probabilistic c1/c4 PASS. The four-cell matrix still fails on 0.8B c4. No speed/auto or broad support promotion |
| Qwen3.6-35B MoE MTP candidate | `unsloth/Qwen3.6-35B-A3B-MTP-GGUF@5bc3e238d916f48a861bac2f8a1990a0e9b7e98d` | `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | `0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b` | Upstream metadata + 16 MiB header range only; full-file SHA/runtime/MoE-head/performance/auto unverified |
| 35B comparison | `ggml-org/Qwen3.6-35B-A3B-GGUF@7878dc9a905c7c9379e5e8d6a1aa9affdd6839c5` | `Qwen3.6-35B-A3B-Q4_K_M.gguf` | `671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7` | Historical native benchmark only; canonical real 35B/C4/C4A/NUMA qualification unverified |
| Gemma E2B text inference target | `unsloth/gemma-4-E2B-it-GGUF@0314792d7f1f7e229411f620751375812bb9faf2` | `gemma-4-E2B-it-Q4_K_M.gguf` | `740185b21d22ceb83a11c3aa62ad5842ef32c70f6096d756bbee85a1e4ec34b8` | Private a9 passes and historical ad1/d408 failures retained. Root `48dd55a` formal 20 cache-ON/OFF c=1/c=4 raw IDs and four staggered own-c1 complete-answer checks PASS. No mixed-physical-graph, HTTP, broad support/default/performance promotion |
| Gemma E4B text inference target | `ggml-org/gemma-4-E4B-it-GGUF@a69f018d7678a2b9f5e705f8903d6fb29b9b25dc` | `gemma-4-E4B-it-Q4_0.gguf` | `a555b900214b477d8880e7832e0b8925e139b0159640036b09fe472b6f2097f2` | Current `ef9fc61` unchanged ctx2048/cap4/t8/Q8/NONE init PASS; complete 59-EOS c1/repeat and two c4 waves of four match own-c1 IDs/five-planet/DONE answer; sustained c1/c4 254-EOS own-ID gates also pass. HTTP/package/support/performance pending. Historical root `48dd55a` context 2048/capacity 4 init capacity FAIL: plan required 5,921,446,144 > available 4,681,576,448 bytes; zero generation. Peak failed-init RSS 6,236,108 KiB is not RAM savings or successful RAM sizing. Private a9 capacity 1 one-case PASS remains separate; no broad support claim |
| gpt-oss target | `ggml-org/gpt-oss-20b-GGUF@15dcfc7e7c608c6161507211ed43c5b590801b4b` | `gpt-oss-20b-MXFP4.gguf` | `27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901` | Packed quantized native fixtures passed; safe host-capacity rejection observed; actual HTTP/long/c1/c4 inference and supported RAM range unverified |

Integrated `4ad6a35` JSON checks on each pinned MTP file pass 16 main native
and 32 HTTP requests, c=1/4, with NONE/MTP, strict brace whitelist, streaming,
cancel/disconnect and recovery. Each file records 54 actual `grammar_complete`
finishes and batched MTP acceptance; these tiny objects do not establish broad
JSON coverage. Global action range is internal/fixture-only; real Python JSON
was not run.

The unchanged probabilistic four-cell matrix remains **overall FAIL**: 0.8B c1
and 2B c1/c4 pass, while 0.8B c4 has one NONE French punctuation failure among
160 HTTP requests. One separate NONE stop-instruction negative is preserved.
All 32 stop probes suppress the literal marker; 31 actual string-stop finishes
and MTP rejection/rollback are observed. Private `5ad54a5` six/160 failures and
all four failed profiles remain historical results, not overwritten controls.
No random-ID/population-distribution, causal quality-improvement or speed claim
follows from these unseeded samples. Automatic profiles remain zero.

On `48dd55a`, the pinned 0.8B MTP file passes the narrow hard-mask lifecycle
probe in NONE and MTP: four contradictory-mask errors commit no token/body, with
six healthy peers/recoveries. Only healthy MTP controls have observed proposal
verification. The V1 pre-engine TLS failure remains UNEXECUTED; reviewed V2 uses
qualified startup preload. The source `48dd55a` result does not inherit `4ad6a35` JSON or
probabilistic qualification, cover 2B on `48dd55a`, or promote model support/defaults.

Matched `4ad6a35`/`48dd55a` SDK loading confirms file mode uses automatic execve while
`-c`/stdin fail static TLS. Private Linux TLS candidate `a70cd4e` has three SDK
loader passes without restart but three raw teardown SIGSEGVs; it remains
unqualified/unmerged, with no engine/model inference. Private H `613a` still has
three of eight c8 parity failures and is not merged.

Prior integrated `143a2ad` c=8 tests on the default file passed bounded resource and
exactly-once cleanup checks but failed greedy EOS parity; cache-OFF and fixed
thread-policy controls also failed. The final integrated core image is untested.
The successful Go-only `386b800` lifecycle image has the earlier `7ecaf86` native
core and does not qualify later Gemma/stop numerical changes.

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
source identity without conflating source identity with the subsequent canonical quality checks.

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
