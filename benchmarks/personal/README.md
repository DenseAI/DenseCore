# Personal PC request comparison

Start each engine with the same model bytes, context, sampling and CPU limits,
then record its actual build flags, binary/library SHA-256 and host information.
The standard-library client stores raw SSE events, usage, TTFT, full completion
time and aggregate completed-output throughput. It requires a complete final
answer and an expected pattern in that answer; a correct word only in reasoning
does not pass. `max_tokens` truncation fails quality.

```bash
printf '["What is the capital of France? Reply with one word."]\n' > /tmp/prompts.json
python3 benchmarks/personal/compare_servers.py --url http://127.0.0.1:8080 \
  --prompts /tmp/prompts.json --expected-pattern '^Paris[.!]?$' --concurrency 1 \
  --out /tmp/dense-c1.json
```

Use a list of at least four prompts for `--concurrency 4`. A single request does
not become a concurrent workload merely because the worker limit is four.
This tool is a request recorder and answer gate; it does not automatically infer
a speedup. Public performance claims require matched five-block ABBA samples,
A/A noise, same-engine token parity for state changes, and every required
platform/concurrency cell passing independently.
