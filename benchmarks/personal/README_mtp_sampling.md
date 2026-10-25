# Probabilistic MTP integration smoke

`check_mtp_sampling.py` checks real NONE/MTP request integration. Preparation and
unit tests do not imply a model run. Wait for the newly qualified canonical KV
and boundary candidate; do not reuse historical8c results for a new binary.

Use pinned0.8B and2B MTP GGUF files. Run c1 and c4 as separate phases under the
shared lock to leave other model jobs a turn. Each phase creates fresh native
processes and HTTP servers for NONE and MTP. For example, from this worktree:

```sh
python3 benchmarks/personal/check_mtp_sampling.py \
  --library "$CANDIDATE_ROOT/build/libdensecore.so" \
  --server "$CANDIDATE_ROOT/bin/densecore-server" \
  --artifact-manifest "$CANDIDATE_ARTIFACT_JSON" \
  --model "$SAMPLING_MODEL_PATH" --expected-model-sha256 "$SAMPLING_MODEL_SHA" \
  --out "$SAMPLING_OUTPUT_DIRECTORY" \
  --build-lock /home/jaewook/dc/build.lock \
  --context 2048 --threads 8 --concurrency 1 --repeats 2 \
  --temperature 0.7 --top-k 40 --top-p 0.95 --repetition-penalty 1.05 \
  --max-tokens 1024 --http-max-tokens 1024
```

Repeat with `--concurrency 4` and a new output directory after c1 passes. Never lower explicit context or concurrency to turn a
capacity error into a pass.

The artifact manifest must have the canonical `files` mapping of absolute
resolved paths to SHA256 values, including the actual library and server. Model
SHA must match before execution. Library/server/dependencies and manifest are
checked again afterward. The report also records harness/helper/case hashes,
CPU/affinity, model identity, runtime plans and the exact probability parameters.

Min-p is0.0, the current native `SamplingParams` default: public C and HTTP do not
expose this parameter. The tool does not send an ignored `min_p` field or add a
seed API. Frequency/presence penalties remain native defaults0.0.

Capital and arithmetic cases require the correct complete final answer.
English/French sentence cases require multiple words, final punctuation and valid
UTF-8. Native length completion, empty/reasoning-only answers, duplicate terminal
or completion metadata, and missing token accounting fail. HTTP checks positive
usage, final `stop`, SSE `[DONE]`, valid UTF-8, stop text suppression and recovery
after disconnect. Native cancellation separately requires exactly one canceled
terminal/completion and a complete follow-up. Native stop/EOS token accounting
and HTTP usage are recorded separately; they are not forced to be equal.

The HTTP stop probe cannot distinguish EOS from string-stop internally because
the API reports both as `stop`; this limitation is explicit in its raw result.
MTP coverage requires actual selected MTP, target width greater than1 and nonzero
acceptance, rejection and rollback counters. Missing coverage is reported as
`branch not exercised` and does not silently pass.

Random strings and IDs are never compared between modes, streams or concurrent
runs. A small case batch does not establish population distribution equivalence.
Formal acceptance/residual/distribution tests and greedy128-token parity are
separate evidence. This tool makes no speed or automatic promotion claim.

Unit checks (no models or native builds):

```sh
python3 -m unittest discover -s benchmarks/personal -p 'test_check_mtp_*.py'
```
