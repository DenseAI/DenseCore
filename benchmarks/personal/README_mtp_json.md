# Native/HTTP JSON completion probe

`check_mtp_json.py` is a correctness harness prepared for a **future qualified combined JSON-completion artifact**. Preparation and mock tests do not establish model support. Do not run the current `4545e05` runtime with this tool. No actual model was run when preparing this harness.

The default matrix is NONE/MTP on the same model bytes, concurrency 1, context 2048, greedy temperature 0 / top-k 1 / top-p 1, and a 512-token budget. Opt into `--concurrency 1 4` and/or `--max-tokens 1024 --http-max-tokens 1024` explicitly. Capacity failures remain failures; the tool does not shrink context or concurrency. Optional positive-temperature settings are recorded, with no native seed, same-seed, random token parity, distribution-equivalence, speed, or automatic-activation claim.

Two cases are mandatory:

- Ordinary JSON must parse to exactly `{"answer":42}`. The value must be an integer, not a boolean, string, or float.
- Strict structural output must be `{}`. Native `CountTokens` and `DenseCoreTokenizeText` must return exactly one distinct global ID each for `{` and `}`. These two IDs form the entire whitelist with `allowed_tokens_strict=true`. Unsupported tokenizer shapes fail explicitly; the tool never expands a whitelist silently.

Native submission uses the public `SubmitRequestWithSamplingConstraintsCallbackEx` API with `json_mode=1`, together with `DenseCoreSetGenerationCompletionCallback`. `SubmitRequestWithTokenResults` cannot combine JSON mode and token constraints and is not used. Callback bytes are copied while borrowed data is valid and decoded incrementally as strict UTF-8. Nonterminal ID -1 text flushes are retained. Collected IDs are **delivered event IDs**, not the complete native sample sequence: EOS/empty samples may be omitted. Authoritative generated count must be positive and at least the delivered nonnegative ID count; equality is not required. Terminal and completion metadata must each arrive once, with metadata before terminal; callbacks after terminal are protocol errors, including whitespace or text-only flushes. These checks remain active through `FreeEngine` join. Callback owners and whitelist buffers remain live until that join.

The native cancellation probe starts a long JSON object containing a number array, cancels after its first event while still active, requires successful `CancelRequest`, error finish reason 3 with explicit request-cancellation text, exactly one terminal/metadata pair, and a completed ordinary JSON recovery. Finish reason 3 is the public generic ERROR enum, not a dedicated cancellation enum. HTTP uses `response_format={"type":"json_object"}`, the actual allow/deny field names, both streaming and nonstreaming requests, and a separate JSON disconnect prompt plus mandatory JSON follow-up. Individual HTTP results (including failures and runtime profile snapshots) are persisted immediately. HTTP completion requires stop, positive authoritative completion usage, and `[DONE]` for streams. No added stop strings hide truncation/completion bugs.

Validators reject duplicate keys, NaN/Infinity, extra prose, reasoning markup/content, invalid UTF-8, missing stream completion, duplicate finish events, semantic mismatch, and length finishes even if a partial or complete object happens to parse. Global `action_token_start` / `action_token_count` are internal fixture-only controls; no public C/HTTP surface exposes them. The tool refuses these fields, min-p, seed, and stop-string overrides rather than sending ignored or confounding fields.

## Actual execution coverage

A selected MTP profile or a successful API `finish_reason=stop` is insufficient. Each native MTP concurrency profile and the HTTP MTP process log must show actual `MTPVerify` observations with maximum target width greater than 1, verified count greater than 0, and accepted count greater than 0. Missing telemetry or zero coverage is an explicit FAIL/gap. Rejection is recorded but is not required for this narrow JSON gate; rejection correctness remains in the separate formal/sampling matrix.

The logged `accepted` counter represents algorithm acceptance, including an accepted terminal EOS or JSON closing output. It must not be described as retained speculative-input count: the terminal closing output can count as accepted while `accepted_draft_tokens` excludes it. This probe deliberately does not impose a retained-input minimum.

Every native profile and each HTTP mode process log must also contain at least one actual operational decode summary with exactly `finish_cause=grammar_complete`. API stop alone cannot attribute the new completion policy. Log paths, SHA256 hashes, qualified artifact source SHA, matching summary line numbers, and native/log-local request IDs are saved; log hashes are checked again at the end. HTTP counters and causes are aggregate process evidence, **not mapped to concurrent HTTP response rows**. If the selected architecture/build does not emit these operational observations, report a telemetry coverage gap rather than inferring the policy from output text or profile flags. Partial and failed process logs remain preserved.

## Artifact gate and future execution

Execution is not authorized by this document. A future integration owner must first qualify the combined source and produce the canonical artifact manifest. In addition to the existing absolute-path `files` SHA map and full `revision`, this harness requires the explicit boolean `json_completion_unit_qualified: true`, `json_completion_unit_source` matching the revision, and `json_completion_unit_report` containing an absolute path and SHA256. The report JSON itself must name the same `source`; its bytes are verified before and after execution. Only the integration owner sets this marker after combined native and ASAN JSON unit qualification. This records unit qualification, not actual model quality. This marker is an assertion by that qualification owner, not a result the harness grants itself. The `--qualified-source` full SHA must match `revision`. Every local `.so` dependency next to the library, the library, and the matching server must appear in the manifest. Model bytes are pinned separately. Old `4545e05` is rejected even if a marker is supplied.

```json
{
  "revision": "FULL_FUTURE_QUALIFIED_SOURCE_SHA",
  "json_completion_unit_qualified": true,
  "json_completion_unit_source": "FULL_FUTURE_QUALIFIED_SOURCE_SHA",
  "json_completion_unit_report": {"path": "/absolute/qualified/unit-report.json", "sha256": "REPORT_SHA256"},
  "files": {
    "/absolute/qualified/build/libdensecore.so": "SHA256",
    "/absolute/qualified/bin/densecore-server": "SHA256",
    "/absolute/qualified/build/libggml.so.0.9.8": "SHA256"
  }
}
```

Include **all** local runtime `.so` files; the abbreviated map above is not a runnable manifest. After the parent explicitly schedules the matching artifact, the command shape is:

```sh
PYTHONPATH=benchmarks/personal python3 benchmarks/personal/check_mtp_json.py \
  --library /absolute/qualified/build/libdensecore.so \
  --server /absolute/qualified/bin/densecore-server \
  --model /absolute/pinned/model.gguf --expected-model-sha256 MODEL_SHA256 \
  --artifact-manifest /absolute/qualified/manifest.json \
  --qualified-source FULL_FUTURE_QUALIFIED_SOURCE_SHA \
  --out /absolute/new-empty-json-result-directory
```

The full run holds the shared build/model lock. Artifact/model/dependency/manifest hashes are checked before and after execution, including direct native-child invocation. Runtime plans, host/affinity, context, sampling, source/harness/helper/data hashes, raw native logs, each native profile and partial failure, each HTTP request, server logs, disconnect/recovery, and final summary are retained. Existing output directories are refused, so failures from different profiles cannot be overwritten by retries. Keep each source-labeled failure; a new source requires a new output directory and requalification.

Pure tests, requiring no model, native library, C++ build, HTTP server, or paid resource:

```sh
PYTHONPATH=benchmarks/personal python3 -m unittest benchmarks/personal/test_check_mtp_json.py -v
```
