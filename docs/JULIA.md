# Julia-1 typed decisions on CPU

DenseCore has a metadata-driven Julia-1 execution profile, a lossless F32 GGUF
converter, and model-specific tokenization and typed-request serialization.
It uses the existing `/v1/systemone` interface and decision C ABI. Task quality,
Julia INT8 quality, GPU economics, and a 2x cost advantage are not established.

## Pinned model and acquisition

The [SupersonicLabs/Julia-1 checkpoint](https://huggingface.co/SupersonicLabs/Julia-1/tree/a85b127321d580d65176c89ced8273f305745d85)
is revision `a85b127321d580d65176c89ced8273f305745d85`. The local download of its
577,189,056-byte F32 checkpoint independently matched SHA-256
`df853bf7fe424420011f3d0c47a05d7341aa9eefa7fb9f203ea4aada4ad95b72`.
Upstream provenance reports `quality_gate: false` and inference policy reports
`replacement_qualified: false`; integrity and numerical checks do not promote
those flags.

```sh
python3 scripts/download_julia_model.py --manifest-only
python3 scripts/download_julia_model.py --output models/julia-1
python3 scripts/download_julia_model.py --output models/julia-1 --verify-only
```

The downloader pins size and SHA-256 for weights, tokenizer, encoder/head
configuration, provenance, inference policy, and reference source. It rejects
mismatching existing files and validates each download before publication.
Downloaded Python is never imported by the downloader or serving process.

## Conversion and independent numerical checking

`convert_julia_gguf.py` uses only the Python standard library. It validates the
complete pinned safetensors tensor set, dimensions, F32 dtype and payload byte
ranges, then copies all 169 learned tensors byte-for-byte into GGUF. The unused
upstream `temperature` buffer is omitted: Julia's named typed API uses raw
softmax. The converter adds two nonpersistent RoPE frequency buffers exported
from the pinned upstream model initialization; these generated buffers are
reported separately from learned tensor identity.

`verify_julia_gguf.py --prepare-only` initializes the pinned upstream model and
writes exact FP32 RoPE frequencies without executing a forward pass. The full
checker independently reads GGUF, checks every copied tensor against the
original checkpoint, and compares PyTorch versus native option logits, raw
probabilities and action outputs. It produces reusable input/reference fixtures.
Both reference modes must run inside a disposable restricted container because
they execute upstream Python. Use a read-only filesystem, no network, no GPU,
resource limits and read-only model/dependency/code mounts; allow only a dedicated
output directory to be writable.

Inside that container, the preparation command is:

```sh
python /owned/verify_julia_gguf.py --model /model --prepare-only \
  --output /out/rope-reference.json --threads 1
```

Conversion runs locally without importing upstream code:

```sh
python3 scripts/convert_julia_gguf.py --model models/julia-1 \
  --rope-reference PATH/rope-reference.json --output models/julia-1/julia-f32.gguf \
  --report PATH/conversion.json
```

Inside the restricted container, run the independent comparison with:

```sh
python /owned/verify_julia_gguf.py --model /model --gguf /out/julia-f32.gguf \
  --library /native/libdensecore_decision.so --threads 4 \
  --output /out/julia-parity.json --fixtures-output /out/julia-parity-inputs.json
```

If the host-built native library requires a newer glibc than the reference
container, separate oracle generation from native replay. Inside the restricted
container add `--reference-only`, omit `--library`, and retain `--gguf` and
`--fixtures-output`. This still independently checks all learned tensor bytes and
generates the FP32 oracle, but explicitly reports native parity as not executed.
Replay those fixed fixtures through the native library on the host; loading the
native library does not require importing upstream Python. Keep the reference
and replay reports distinct and bind both to fixture and library hashes.

For the generated pinned F32 GGUF, host replay is:

```sh
python3 scripts/replay_encoder_fixtures.py --model PATH/julia-f32.gguf \
  --model-sha256 6d5fc0afffff67d63115b2ab4c57fa41b5dffdd09e30faa3978b4aa1a91cff09 \
  --library PATH/libdensecore_decision.so \
  --fixtures benchmarks/fixtures/julia/parity-inputs.json \
  --threads 4 --output PATH/native-parity.json
```

The replay binds original checkpoint and tokenizer identities as well as GGUF,
fixture and library hashes. It checks single execution, duplicate batches,
mixed lengths/types, reversed order, and a long input followed by a short input.

The reference environment uses Python 3.11, CPU PyTorch 2.8.0, Transformers 5.0.0,
Safetensors 0.6.2 and NumPy 2.2.6. Full local dependency metadata and container
identity are retained in
`private/benchmarks/encoder-economics-20261016/julia-reference-dependencies.json`.
The numerical gate is fixed at option logit absolute error `1e-3`, raw probability
error `1e-4`, action logit error `1e-3 + 1e-6 * abs(reference)`, and action
probability error `1e-4`. A parity pass establishes numerical compatibility only
for the checked fixtures; it does not establish task accuracy.

The local canonical native library passed all 16 replay runs on six independent
FP32 fixtures, including a 339-token input that exercises local/global attention.
Maximum option-logit error was `2.28882e-5`, raw-probability error `3.21263e-6`,
action-logit error `9.98378e-7`, and action-probability error `2.92497e-7`.
The immutable source/model/library bindings and fixed limits are recorded in
`benchmarks/fixtures/julia/reference-parity.json`. This covers 27–339-token
fixtures; it does not establish long-context or dataset quality.

## Input and probability contract

The authoritative checkpoint inference policy specifies maximum context 8192,
head budget 512 and strict encoding. Source helper defaults of head budget 256
are not substituted for that policy. The native runtime reports its effective
workspace-limited sequence cap separately from the model's positional limit.
The current native profile rejects inputs over 4829 tokens before allocating a
graph; the numerical fixtures exercise lengths only through 339.

`julia_inputs.py` prepares strict, lossless JSONL requests using the hash-verified
`tokenizer.json` and an already installed optional `tokenizers` package:

```sh
python3 scripts/julia_inputs.py --model models/julia-1 --input requests.jsonl \
  --max-length 8192 --head-length 512 > encoded.jsonl
```

Each row contains `state` (text or JSON), `question`, `options` (2–20 nonempty
strings) and optional `type` (`choice`, `score`, `noul`). Boolean options are in
false/true order. The sequence is `<bos>`, typed question, `<eos>`, `<mask>` before
each option, `<eos>`, state, `<eos>`. Reserved markers, options over 48 tokens,
head overflow and context overflow are rejected. Valid tokens count the full
unpadded sequence once; input preparation does not imply successful inference.

The named HTTP adapter follows pinned `julia/typed.py`: choice label IDs remain
output keys while descriptions enter the encoder, score descriptions have no
synthetic prefix, and absent Boolean descriptions become literal `false` and
`true`. Named HTTP responses and quality metrics use raw softmax. The separate
legacy display thresholding helper is exposed only for explicit presentation
comparisons; it must not be used to conceal calibration differences.

The actual encoder has width 384, 22 layers, 6 heads, intermediate width 1152,
vocabulary size 256000, local attention distance at most 64 and RoPE theta
160000. Its two pre-norm decision-head layers use FFN width 1536 and ReLU; the
scorer and action head use GELU. Julia and Laya keep distinct immutable metadata
profiles, tokenizers and probability policies.

## Verification boundaries

```sh
python3 -m unittest discover -s scripts -p 'test_*julia*.py'
```

Offline regressions cover artifact corruption and overwrite preservation,
malformed tensor names/dtypes/shapes/ranges, lossless serialization rejection and
raw/display probability separation. Independent Rust tokenizer/reference
serialization fixtures cover Unicode and typed HTTP adapters. Numerical model
parity is reported separately in the local parity artifact.

Published H200 BF16 accuracy does not provide matched 128/256-token throughput
or a whole-instance price baseline. Julia GPU economics remain `unavailable`.
