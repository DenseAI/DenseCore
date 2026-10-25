This standalone oracle targets llama.cpp revision `74bca3950054eed1795f525cb9cf06bfe796492d` and one Qwen3.5-family NextN head. It runs in a separate process with llama.cpp's own ggml libraries.

Supply exact token IDs and an explicit row-major little-endian float32 hidden matrix with one embedding row per token. The first token is position zero. For prompt bootstrap, row zero must be zeros and each following row must contain the previous target token's **post-output-norm** hidden state. Synthetic hidden inputs can test head math independently of target differences. This tool does not generate hidden inputs implicitly.

```sh
python3 benchmarks/personal/probes/run_llama_nextn_oracle.py \
  --llama-root /home/jaewook/dc/llama.cpp \
  --build-lock /home/jaewook/dc/build.lock \
  --out /home/jaewook/dc/logs/nextn-oracle \
  --model MODEL.gguf --model-sha256 EXACT_SHA256 \
  --tokens 1,2,3 --hidden explicit-input.f32 --chunk-size 1 --threads 8
```

Use `--build-only` to compile and record dependencies without loading a model. Each output directory must be new. Outputs are `oracle.logits.f32` (rows × vocabulary), `oracle.hidden.f32` (rows × embedding, normalized draft feedback), shape/argmax metadata, raw build/run logs and a provenance manifest. Repeating with chunk sizes 1 and 16 checks sequential versus multi-position oracle execution.

A successful oracle run alone does not verify DenseCore's head. Compare matching explicit inputs, positions, model bytes, output shapes, complete logits and normalized feedback against a separate DenseCore probe, with numeric tolerances declared before comparing. Token parity through target acceptance is a separate gate and can pass even if head math is wrong. This tool provides no performance or automatic activation qualification.
