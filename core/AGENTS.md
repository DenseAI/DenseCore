# DenseCore Core Agent Rules

The repository-root `AGENTS.md` remains authoritative. Apply this file as the local execution contract for work under `DenseCore/core/`. For GPT-5.6 and later frontier models, keep conclusions tied to concrete files, call paths, tests, and benchmark artifacts; label unmeasured performance ideas as hypotheses.

## Graphify

This project has a graphify knowledge graph at `graphify-out/`.

Rules:
- Before answering architecture or codebase questions, read `graphify-out/GRAPH_REPORT.md` for god nodes and community structure.
- If `graphify-out/wiki/index.md` exists, use it to locate the owning code, then verify claims against the current source files.
- After modifying code files in this session, run `../.venv-graphify/bin/graphify update .` to keep the graph current (AST-only, no API cost).

## GCP Benchmark Cleanup

- Treat `gcloud` benchmark VMs as temporary. Once the requested run, artifact transfer, and remote inspection are complete, stop the VM immediately or delete it if it is disposable.
- Use `gcloud compute instances stop NAME --zone ZONE` only when the VM or attached disks must be reused. Use `gcloud compute instances delete NAME --zone ZONE --delete-disks=all` after copying required results when the experiment resources are no longer needed.
- Verify the final state is `TERMINATED` or the instance is absent before declaring the benchmark task complete. Never leave a VM `RUNNING` during analysis, report writing, or while waiting for the next instruction.
- Stopping compute does not eliminate storage, reserved static-IP, snapshot, or image charges. List any retained billable resources in the final report and delete experiment-only resources when they are no longer required.
