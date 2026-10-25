#!/usr/bin/env python3
"""Verify a real accepted stream survives SIGTERM while new admission stops."""
import argparse
import errno
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
        "legacy_5s_regression": "not_exercised", "request_max_tokens": max_tokens, "done_count": 0, "finish_count": 0}
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

        def probe_unavailable(path, data=None):
            event = {"path": path, "started_seconds": time.monotonic() - began}
            unavailable = False
            try:
                request = urllib.request.Request(base_url + path, data=data,
                    headers={"Content-Type": "application/json"} if data else {})
                with urllib.request.urlopen(request, timeout=min(1, remaining())) as response:
                    event["status"] = response.status
                    unavailable = response.status == 503
            except urllib.error.HTTPError as error:
                event.update(status=error.code, body=error.read(2048).decode(errors="replace"))
                unavailable = error.code == 503
            except (urllib.error.URLError, OSError, http.client.HTTPException) as error:
                event.update(error_type=type(error).__name__, error=str(error))
                unavailable = connection_closed(error)
            finally:
                event.update(completed_seconds=time.monotonic() - began, unavailable=unavailable)
                result["probe_events"].append(event)
            return unavailable

        subprocess.run(signal_command, check=True, timeout=min(10, remaining()))
        result["signal_command_completed_seconds"] = time.monotonic() - began
        readiness_deadline = min(deadline, started + 10)
        while time.monotonic() < readiness_deadline:
            result["ready_removed"] = probe_unavailable("/health/ready")
            if result["ready_removed"]:
                break
            time.sleep(min(0.05, remaining()))
        result["admission_blocked"] = probe_unavailable("/v1/chat/completions", payload)
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
        if not result["ready_removed"] or not result["admission_blocked"]:
            raise RuntimeError("readiness or new admission stayed available during drain")
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
