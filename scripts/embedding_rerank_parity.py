#!/usr/bin/env python3
"""Compare DenseCore embedding/rerank APIs against a Transformers reference.

This harness assumes the DenseCore server is already running. It can optionally
load a DenseCore model first. Reference embeddings are computed with
transformers/torch using the same pooling and normalization knobs sent to
DenseCore.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.error
import urllib.request
from typing import Any


DEFAULT_TEXTS = [
    "DenseCore runs CPU inference in Kubernetes.",
    "Autoscaling should follow request queue depth.",
    "This document is about cooking dinner.",
]
DEFAULT_QUERY = "CPU inference autoscaling"


def post_json(base_url: str, path: str, payload: dict[str, Any], timeout: float) -> Any:
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{path} failed with HTTP {err.code}: {body}") from err


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def max_abs_delta(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def load_reference_model(model_id: str, device: str, trust_remote_code: bool):
    try:
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install torch and transformers to run reference parity") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    rope_parameters = getattr(config, "rope_parameters", None)
    if not hasattr(config, "rope_theta") and isinstance(rope_parameters, dict) and "rope_theta" in rope_parameters:
        setattr(config, "rope_theta", rope_parameters["rope_theta"])
    model = AutoModel.from_pretrained(model_id, config=config, trust_remote_code=trust_remote_code)
    model.eval()
    model.to(device)
    return torch, tokenizer, model


def pool_hidden(torch, outputs, attention_mask, pooling: str):
    hidden = outputs.last_hidden_state
    if pooling == "cls":
        return hidden[:, 0]
    if pooling == "last":
        lengths = attention_mask.sum(dim=1) - 1
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[batch, lengths]
    if pooling == "max":
        mask = attention_mask.unsqueeze(-1).bool()
        masked = hidden.masked_fill(~mask, torch.finfo(hidden.dtype).min)
        return masked.max(dim=1).values

    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


def reference_embeddings(
    model_id: str, texts: list[str], pooling: str, normalize: bool, device: str, trust_remote_code: bool
) -> list[list[float]]:
    torch, tokenizer, model = load_reference_model(model_id, device, trust_remote_code)
    batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch, use_cache=False)
        vectors = pool_hidden(torch, outputs, batch["attention_mask"], pooling)
        if normalize:
            vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
    return vectors.detach().cpu().float().tolist()


def densecore_embeddings(base_url: str, texts: list[str], pooling: str, normalize: bool, timeout: float) -> list[list[float]]:
    body = post_json(
        base_url,
        "/v1/embeddings",
        {
            "model": "densecore",
            "input": texts,
            "pooling_type": pooling,
            "normalize": normalize,
        },
        timeout,
    )
    data = body.get("data")
    if not isinstance(data, list) or len(data) != len(texts):
        raise RuntimeError(f"unexpected embedding response: {body!r}")
    return [item["embedding"] for item in sorted(data, key=lambda item: item["index"])]


def densecore_rerank(base_url: str, query: str, documents: list[str], top_n: int, timeout: float) -> list[int]:
    body = post_json(
        base_url,
        "/v1/rerank",
        {
            "model": "densecore",
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": True,
        },
        timeout,
    )
    results = body.get("results")
    if not isinstance(results, list):
        raise RuntimeError(f"unexpected rerank response: {body!r}")
    return [int(item["index"]) for item in results]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--reference-model", required=True, help="Transformers model id or local path")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to Transformers")
    parser.add_argument("--densecore-model-path", default="", help="Optional model path to load into DenseCore first")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--pooling", choices=["mean", "cls", "last", "max"], default="mean")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-cosine", type=float, default=0.9999)
    parser.add_argument("--max-abs-delta", type=float, default=0.02)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--text", action="append", dest="texts", help="Text to embed; repeat for multiple inputs")
    parser.add_argument("--text-prefix", default="", help="Prefix applied to all embedding texts for both engines")
    parser.add_argument("--query-prefix", default="", help="Prefix applied to rerank query for both engines")
    parser.add_argument("--document-prefix", default="", help="Prefix applied to rerank documents for both engines")
    parser.add_argument("--top-n", type=int, default=2)
    parser.add_argument("--skip-rerank", action="store_true", help="Only validate /v1/embeddings")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    raw_texts = args.texts or DEFAULT_TEXTS
    texts = [args.text_prefix + text for text in raw_texts]
    if args.densecore_model_path:
        payload: dict[str, Any] = {"model_path": args.densecore_model_path}
        if args.threads > 0:
            payload["threads"] = args.threads
        post_json(args.base_url, "/v1/models/load", payload, args.timeout)

    ref = reference_embeddings(
        args.reference_model, texts, args.pooling, args.normalize, args.device, args.trust_remote_code
    )
    got = densecore_embeddings(args.base_url, texts, args.pooling, args.normalize, args.timeout)

    failures: list[str] = []
    for idx, (dense_vec, ref_vec) in enumerate(zip(got, ref)):
        if len(dense_vec) != len(ref_vec):
            failures.append(f"embedding[{idx}] dim mismatch: densecore={len(dense_vec)} reference={len(ref_vec)}")
        sim = cosine(dense_vec, ref_vec)
        delta = max_abs_delta(dense_vec, ref_vec)
        print(
            f"embedding[{idx}]\tcosine={sim:.8f}\tmax_abs_delta={delta:.8f}"
            f"\tdim_dense={len(dense_vec)}\tdim_ref={len(ref_vec)}\ttext={raw_texts[idx]!r}"
        )
        if sim < args.min_cosine:
            failures.append(f"embedding[{idx}] cosine {sim:.8f} < {args.min_cosine}")
        if delta > args.max_abs_delta:
            failures.append(f"embedding[{idx}] max_abs_delta {delta:.8f} > {args.max_abs_delta}")

    if not args.skip_rerank:
        rerank_query = args.query_prefix + args.query
        rerank_docs = [args.document_prefix + text for text in raw_texts]
        ref_all = reference_embeddings(
            args.reference_model,
            [rerank_query] + rerank_docs,
            args.pooling,
            args.normalize,
            args.device,
            args.trust_remote_code,
        )
        ref_query = ref_all[0]
        ref_docs = ref_all[1:]
        ref_ranking = sorted(range(len(rerank_docs)), key=lambda idx: cosine(ref_query, ref_docs[idx]), reverse=True)[
            : args.top_n
        ]
        dense_ranking = densecore_rerank(args.base_url, rerank_query, rerank_docs, args.top_n, args.timeout)
        print(f"rerank\treference={ref_ranking}\tdensecore={dense_ranking}")
        if dense_ranking != ref_ranking:
            failures.append(f"rerank order mismatch: reference={ref_ranking}, densecore={dense_ranking}")

    if failures:
        for failure in failures:
            print(f"FAIL\t{failure}", file=sys.stderr)
        return 1
    print("ok\tembedding/rerank parity passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
