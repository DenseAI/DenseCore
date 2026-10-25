#!/usr/bin/env python3
"""Verify a real accepted stream survives SIGTERM while new admission stops."""
import argparse
import json
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


def run(base_url, signal_command, output, timeout):
    observed = threading.Event()
    completed = threading.Event()
    result = {"events": [], "ready_removed": False, "admission_blocked": False}
    payload = json.dumps({"model": "densecore", "messages": [{"role": "user", "content":
        "List the integers from 1 to 200, one at a time, without skipping any."}],
        "max_tokens": 256, "temperature": 0, "stream": True,
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
                        result["done"] = True
                        break
                    event = json.loads(data)
                    result["events"].append(event)
                    if "error" in event:
                        raise RuntimeError(str(event["error"]))
                    for choice in event.get("choices", []):
                        delta = choice.get("delta", {})
                        if delta.get("content") or delta.get("reasoning_content"):
                            observed.set()
                        if choice.get("finish_reason"):
                            result["finish_reason"] = choice["finish_reason"]
        except Exception as error:
            result["error"] = str(error)
        finally:
            completed.set()

    thread = threading.Thread(target=stream, daemon=True)
    thread.start()
    try:
        if not observed.wait(timeout):
            raise RuntimeError("no stream token observed before drain")
        if completed.is_set():
            raise RuntimeError("stream completed before SIGTERM; active drain was not tested")
        started = time.monotonic()
        subprocess.run(signal_command, check=True, timeout=10)
        while time.monotonic() - started < min(timeout, 10):
            try:
                with urllib.request.urlopen(base_url + "/health/ready", timeout=1) as ready:
                    result["ready_removed"] = ready.status != 200
            except (urllib.error.URLError, OSError):
                result["ready_removed"] = True
            if result["ready_removed"]:
                break
            time.sleep(0.05)
        try:
            probe = urllib.request.Request(base_url + "/v1/chat/completions", data=payload,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(probe, timeout=2) as response:
                result["admission_blocked"] = response.status == 503
        except urllib.error.HTTPError as error:
            result["admission_blocked"] = error.code == 503
        except (urllib.error.URLError, OSError):
            result["admission_blocked"] = True
        if not completed.wait(timeout):
            raise RuntimeError("accepted request exceeded drain deadline")
        result["drain_seconds"] = time.monotonic() - started
        if not result.get("done") or result.get("finish_reason") not in ("stop", "length") or result.get("error"):
            raise RuntimeError("accepted request did not finish successfully during drain")
        if not result["ready_removed"] or not result["admission_blocked"]:
            raise RuntimeError("readiness or new admission stayed available during drain")
    finally:
        Path(output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("signal_command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.signal_command
    if command and command[0] == "--": command = command[1:]
    if not command: parser.error("provide the SIGTERM command after --")
    run(args.url.rstrip("/"), command, args.output, args.timeout)
