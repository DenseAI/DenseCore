#!/usr/bin/env python3
"""Verify a real accepted stream survives SIGTERM while new admission stops."""
import argparse
import errno
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import re
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


def connection_closed(error):
    """Only explicit transport closure proves unavailability; timeouts do not."""
    cause = error.reason if isinstance(error, urllib.error.URLError) else error
    return (isinstance(cause, http.client.RemoteDisconnected) or
        isinstance(cause, OSError) and not isinstance(cause, (TimeoutError, socket.timeout))
        and cause.errno in (errno.ECONNREFUSED, errno.ECONNRESET, errno.EPIPE))


def run(base_url, signal_command, output, timeout, max_tokens=256, active_metric="densecore_active_requests"):
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    began = time.monotonic()
    observed = threading.Event()
    completed = threading.Event()
    result = {"events": [], "ready_removed": False, "admission_blocked": False,
        "probe_events": [], "shutdown_timeout_seconds": timeout,
        "legacy_5s_regression": "not_exercised", "active_probe_rounds": [],
        "ready_removed_while_stream_active": False, "admission_blocked_while_stream_active": False,
        "native_active_during_shutdown_probes": False, "request_max_tokens": max_tokens, "done_count": 0, "finish_count": 0}
    payload = json.dumps({"model": "densecore", "messages": [{"role": "user", "content":
        "List the integers from 1 to 200, one at a time, without skipping any."}],
        "max_tokens": max_tokens, "temperature": 0, "stream": True,
        "chat_template_kwargs": {"enable_thinking": False}}).encode()

    def stream():
        try:
            request = urllib.request.Request(base_url + "/v1/chat/completions", data=payload,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                for line in response:
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        result["done_count"] += 1
                        result["done"] = True
                        if result["done_count"] != 1:
                            raise RuntimeError("duplicate stream [DONE] terminal")
                        continue
                    if result["done_count"]:
                        raise RuntimeError("stream payload after [DONE]")
                    event = json.loads(data)
                    result["events"].append(event)
                    if "error" in event:
                        raise RuntimeError(str(event["error"]))
                    for choice in event.get("choices", []):
                        delta = choice.get("delta", {})
                        if delta.get("content") or delta.get("reasoning_content"):
                            if not observed.is_set():
                                result["first_token_seconds"] = time.monotonic() - began
                            observed.set()
                        if choice.get("finish_reason"):
                            result["finish_count"] += 1
                            result["finish_reason"] = choice["finish_reason"]
        except Exception as error:
            result["error"] = str(error)
        finally:
            result["stream_completed_seconds"] = time.monotonic() - began
            completed.set()

    thread = threading.Thread(target=stream, daemon=True)
    thread.start()
    try:
        if not observed.wait(timeout):
            raise RuntimeError("no stream token observed before drain")
        if completed.is_set():
            raise RuntimeError("stream completed before SIGTERM; active drain was not tested")
        # An unfinished buffered HTTP stream alone does not prove native work
        # remains active. Require the native engine gauge before delivering SIGTERM.
        with urllib.request.urlopen(base_url + "/metrics", timeout=min(timeout, 2)) as metrics:
            raw_metrics = metrics.read().decode()
        pattern = r"^" + re.escape(active_metric) + r"(?:\{[^}]*\})?\s+([0-9.eE+-]+)(?:\s|$)"
        samples = re.findall(pattern, raw_metrics, re.MULTILINE)
        result["active_metric_name"] = active_metric
        result["active_requests_before_signal"] = sum(float(value) for value in samples)
        result["active_metric_seconds"] = time.monotonic() - began
        if not samples or result["active_requests_before_signal"] != 1 or completed.is_set():
            raise RuntimeError("exactly one active native request was not proven before SIGTERM")
        result["stream_incomplete_before_signal"] = True
        result["signal_seconds"] = time.monotonic() - began
        result["events_before_signal"] = len(result["events"])
        started = time.monotonic()
        deadline = started + timeout

        def remaining():
            budget = deadline - time.monotonic()
            if budget <= 0:
                raise RuntimeError("accepted request exceeded drain deadline")
            return budget

        def probe(path, data=None):
            event = {"path": path, "started_seconds": time.monotonic() - began}
            unavailable = False
            try:
                request = urllib.request.Request(base_url + path, data=data,
                    headers={"Content-Type": "application/json"} if data else {})
                with urllib.request.urlopen(request, timeout=min(1, remaining())) as response:
                    event["status"] = response.status
                    unavailable = response.status == 503
                    if path == "/metrics":
                        event["body"] = response.read(1048576).decode(errors="replace")
                        values = re.findall(pattern, event["body"], re.MULTILINE)
                        event["active_requests"] = sum(float(value) for value in values) if values else None
            except urllib.error.HTTPError as error:
                event.update(status=error.code, body=error.read(2048).decode(errors="replace"))
                unavailable = error.code == 503
            except (urllib.error.URLError, OSError, http.client.HTTPException) as error:
                event.update(error_type=type(error).__name__, error=str(error))
                unavailable = connection_closed(error)
            finally:
                event.update(completed_seconds=time.monotonic() - began, unavailable=unavailable)
                event["accepted_stream_incomplete"] = not completed.is_set()
                result["probe_events"].append(event)
            return event

        subprocess.run(signal_command, check=True, timeout=min(10, remaining()))
        result["signal_command_completed_seconds"] = time.monotonic() - began
        admission_payload = json.dumps({**json.loads(payload), "max_tokens": 1}).encode()
        result["admission_probe_max_tokens"] = 1
        # Readiness and admission can change independently. Probe both alongside
        # the native gauge rather than waiting for one before observing the other.
        with ThreadPoolExecutor(max_workers=3) as probes:
            while not completed.is_set():
                remaining()
                round_started = time.monotonic() - began
                ready_future = probes.submit(probe, "/health/ready")
                admission_future = probes.submit(probe, "/v1/chat/completions", admission_payload)
                metric_future = probes.submit(probe, "/metrics")
                ready, admission, metric = (future.result() for future in
                    (ready_future, admission_future, metric_future))
                ready_active = ready["unavailable"] and ready["accepted_stream_incomplete"]
                admission_active = admission["unavailable"] and admission["accepted_stream_incomplete"]
                native_active = metric.get("active_requests") is not None and metric["active_requests"] > 0 and metric["accepted_stream_incomplete"]
                qualified = ready_active and admission_active and native_active
                result["active_probe_rounds"].append({"started_seconds": round_started,
                    "completed_seconds": time.monotonic() - began,
                    "ready_unavailable": ready["unavailable"],
                    "admission_unavailable": admission["unavailable"],
                    "native_active_requests": metric.get("active_requests"),
                    "ready_stream_incomplete": ready["accepted_stream_incomplete"],
                    "admission_stream_incomplete": admission["accepted_stream_incomplete"],
                    "metrics_stream_incomplete": metric["accepted_stream_incomplete"],
                    "qualified": qualified})
                result["ready_removed"] |= ready["unavailable"]
                result["admission_blocked"] |= admission["unavailable"]
                result["ready_removed_while_stream_active"] |= ready_active
                result["admission_blocked_while_stream_active"] |= admission_active
                result["native_active_during_shutdown_probes"] |= native_active
                if qualified:
                    result["qualified_active_probe_round"] = len(result["active_probe_rounds"]) - 1
                    break
                if completed.wait(min(0.05, remaining())):
                    break
        if not completed.wait(remaining()):
            raise RuntimeError("accepted request exceeded drain deadline")
        remaining()  # Probes and signal delivery consume the same overall budget.
        result["drain_seconds"] = time.monotonic() - started
        # Command completion is a conservative delivery boundary. This is observed
        # client stream lifetime, not an estimate based on token count or speed.
        result["observed_stream_seconds_after_signal_command"] = max(0,
            result["stream_completed_seconds"] - result["signal_command_completed_seconds"])
        if result["observed_stream_seconds_after_signal_command"] > 5:
            result["legacy_5s_regression"] = "exercised_client_stream_window"
        if result["done_count"] != 1 or result["finish_count"] != 1 or result.get("finish_reason") not in ("stop", "length") or result.get("error"):
            raise RuntimeError("accepted request did not finish successfully during drain")
        if "qualified_active_probe_round" not in result:
            raise RuntimeError("readiness removal and new admission blocking with native active work were not proven while accepted stream remained active")
    except Exception as error:
        result["failure"] = str(error)
        raise
    finally:
        Path(output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--max-tokens", type=int, default=256, help="Bounded accepted workload; keep within the configured shutdown deadline")
    parser.add_argument("--active-metric", default="densecore_active_requests", help="Native active-request gauge exposed by /metrics")
    parser.add_argument("signal_command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.signal_command
    if command and command[0] == "--": command = command[1:]
    if not command: parser.error("provide the SIGTERM command after --")
    if args.max_tokens < 1: parser.error("--max-tokens must be positive")
    run(args.url.rstrip("/"), command, args.output, args.timeout, args.max_tokens, args.active_metric)
