# Julia-1 independent FP32 fixtures

`parity-inputs.json` contains six independent CPU PyTorch reference cases from
`SupersonicLabs/Julia-1` revision
`a85b127321d580d65176c89ced8273f305745d85`. Inputs and expected outputs were
produced by the pinned upstream model and tokenizer in a disposable restricted
container. Checkpoint, tokenizer and fixture hashes bind the numerical oracle.
No DenseCore output was used to create expected logits.

`reference-parity.json` records the original F32 checkpoint/GGUF tensor identity,
reference versions, native library hash, fixed acceptance limits and native
replay results. All 169 learned tensors match exactly; two generated RoPE buffers
are accounted separately. Native FP32 replay passed all 16 runs covering six
single cases, duplicate batches, mixed lengths/types, reverse order, and a long
input followed by a short input. The longest input is 339 tokens, exercising
local and global attention; this does not qualify 8192-token execution.

The maximum option-logit error was `2.28882e-5`, raw-probability error `3.21263e-6`,
action-logit error `9.98378e-7`, and action-probability error `2.92497e-7`.
Acceptance limits remain option-logit absolute error `1e-3`, probability error
`1e-4`, action-logit error `1e-3 + 1e-6 * abs(reference)` and action-probability
error `1e-4`.

Reproduce native replay without importing upstream code:

```sh
python3 scripts/replay_encoder_fixtures.py --model PATH/julia-f32.gguf \
  --model-sha256 6d5fc0afffff67d63115b2ab4c57fa41b5dffdd09e30faa3978b4aa1a91cff09 \
  --library PATH/libdensecore_decision.so --threads 4 \
  --fixtures benchmarks/fixtures/julia/parity-inputs.json --output PATH/parity.json
```

See `docs/JULIA.md` for artifact acquisition, lossless conversion and independent
oracle generation. Numerical parity on six fixtures does not establish task
accuracy, INT8 quality, throughput or GPU economics. Upstream provenance retains
`quality_gate: false` and `replacement_qualified: false`.
