# Laya typed decisions on CPU

DenseCore can run the Apache-2.0 [Laya model](https://huggingface.co/convaiinnovations/laya)
using an existing, self-contained [GGUF](https://huggingface.co/zerodegress/laya-gguf).
It returns choice probabilities, ordinal scores, and Boolean probabilities through
`POST /v1/systemone`. This is an encoder decision model, not a text generation API.
Jev weights and the Jev hosted API are not supported by this implementation.

The Go process handles tokenization and typed requests; C++/GGML executes
ModernBERT and the decision heads. Python is only used for downloading and
independent validation, never by the serving process.
F16 model projections expand per layer for F32 arithmetic to preserve numerical
compatibility. The runtime reuses its scratch workspace between layers and requests.

## Download and build

From a recursive repository checkout, with CMake, a C++17 compiler and Go 1.25.13+:

```bash
python3 scripts/download_laya_model.py
cmake -S core -B build -DCMAKE_BUILD_TYPE=Release \
  -DDENSECORE_DECISION_ONLY=ON -DGGML_NATIVE=OFF \
  -DGGML_CPU_REPACK=OFF -DGGML_LLAMAFILE=OFF
cmake --build build -j4
ctest --test-dir build --output-on-failure
cd server
go build -mod=readonly -o ../bin/densecore-decision ./cmd/densecore-decision
cd ..
```

The downloader pins revision `ff850ac9c08707e926847711a4d1819dc228ecc7`,
verifies 848,158,048 bytes and SHA-256
`62db2affc0fe4b9f2538ba61299e5fd8bf876f50f765089a343cd4113eef21a4`,
and refuses to overwrite a different existing file. No model conversion is needed.
The model is ignored by Git. Allow additional RAM for native weights and workspaces.

The standalone executable only links the decision runtime. A full DenseCore
build also exposes `densecore decision --model PATH predict|serve`.
Keep the generated shared libraries together in `build`; for another build
directory, provide its location through `CGO_LDFLAGS` when building Go and the
dynamic loader path when running.

## Serve and request a decision

```bash
./bin/densecore-decision serve --model models/laya/laya-f16.gguf --threads 4
```

In another terminal:

```bash
curl --fail http://127.0.0.1:8080/health
curl --fail http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "I was charged twice for my subscription.",
    "questions": {
      "route": {
        "type": "choice",
        "instructions": "Classify the support request.",
        "criteria": {
          "billing": "Payment or invoice",
          "technical": "Software problem",
          "other": "Other request"
        }
      }
    }
  }'
```

For stdin/stdout use `densecore-decision predict --model PATH`; it accepts the
same request JSON. `/v1/models` lists `laya`. Startup completes only after loading
and validating the model and tokenizer.

Question types:

| Type | Criteria | Answer |
| --- | --- | --- |
| `choice` | Ordered object of label-to-description pairs, or string array | `choice`, `probabilities` |
| `score` | Array of descriptions ordered from lowest to highest level | Probability-weighted zero-based `score`, `probabilities`, `legend` |
| `noul` | Optional object with `false` and `true` descriptions | `noul`, the probability that the statement holds |

All questions require `instructions`. Answers include `confidence`,
`answer_confidence`, and the model's `action.act_probability`. These are model
outputs, not proof of correctness or permission to execute an action. Choice and
score confidence use normalized entropy; `noul` confidence is the larger Boolean
probability. Upstream temperature buckets and their serving clamp are applied.
An optional `option_order` permutation controls presented option order while
answers retain the original labels.

## Limits

- Initial support is `general.architecture=laya`, `laya.weight_layout=ggml-canonical`
  with the published 28-layer architecture and F16/F32 tensors. The pinned F16
  artifact is the qualification target. Other GGUF layouts and quantized weights
  are rejected, including backbone-only files that need an external head.
- Context is 512 tokens per question, with a 192-token question/option budget.
  State arrays retain their end when truncated; strings and objects retain their
  beginning. Usage reports dropped state tokens and options collapsed by truncation.
- One request runs at a time, with at most 32 questions, 64 options per question,
  and a 1 MiB body. Questions execute sequentially and repeat the encoder pass.
  Concurrent requests return 429; invalid input returns 422 and oversized bodies
  return 413. Shutdown rejects new work and drains native inference.
- The standalone server binds to loopback by default. It does not implement the
  main chat server's authentication, TLS, or metrics configuration. External
  exposure requires an authenticated proxy.
- This is a source-build preview, not included in an already published Docker
  release. Numerical parity is distinct from task accuracy, calibration quality,
  and speed relative to other engines. No comparative performance claim is made.

## Independent validation

`scripts/verify_laya_gguf.py` compares GGUF learned tensors and native raw logits,
action logits, and calibrated probabilities against the independently downloaded
upstream safetensors checkpoint and pinned upstream PyTorch `DecisionModel`.
It checks input hashes and uses fixed error limits rather than accepting changed
outputs as golden data. Python packages are validation-only dependencies.
See its `--help` for required reference paths and the small committed artifacts
under `benchmarks/fixtures/laya/` for reproducible inputs and results.

The tokenizer tests compare Go NFC, Unicode pre-tokenization and byte BPE against
Hugging Face tokenization. Set `LAYA_TOKENIZER_JSON` to the embedded tokenizer JSON
to run the real-vocabulary fixture, and `LAYA_TOKENIZER_CASES` for an additional
differential corpus. Native integration tests use the downloaded GGUF; consult
the test source for their environment variables.

Architecture and preprocessing follow the Apache-2.0
[upstream Laya implementation](https://github.com/NandhaKishorM/laya/tree/6d942c92081fbc139e736bbd9ac0023223c29b7f).
Model ownership and licensing remain with the upstream publishers.
