"""Shared ComfyUI HTTP layer. No model logic lives here.

All model providers use ComfyClient: submit an API-format workflow, wait for
completion, download the result file. ComfyUI does ALL image/video generation.
"""
from __future__ import annotations

import http.client
import json
import os
import time
import urllib.parse
import urllib.request
from urllib.parse import urlparse


class ComfyError(RuntimeError):
    """Any failure talking to ComfyUI (unreachable, rejected, job failed)."""


def _http_json(base_url: str, method: str, path: str, payload=None, timeout: int = 60) -> dict:
    url = base_url.rstrip("/") + path
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:4000]
        raise ComfyError(f"ComfyUI {method} {path} failed ({exc.code}): {body}") from exc
    except Exception as exc:
        raise ComfyError(f"ComfyUI {method} {path} failed: {exc}") from exc


def format_job_error(prompt_id: str, entry: dict, workflow: dict | None = None) -> str:
    """Render the complete ComfyUI failure: messages with node context + traceback."""
    status = entry.get("status") or {}
    node_names: dict[str, str] = {}
    if workflow:
        for nid, node in workflow.items():
            node_names[str(nid)] = node.get("class_type", "?")
    lines = [f"ComfyUI job {prompt_id} failed (status={status.get('status_str')})."]
    for msg in status.get("messages", []):
        if isinstance(msg, list) and len(msg) == 2 and msg[0] == "execution_error":
            err = msg[1] or {}
            cls = node_names.get(str(err.get("node_id")), "?")
            lines.append(f"--- failing node {err.get('node_id')} ({cls}) ---")
            lines.append(f"exception: {err.get('exception_type')}: "
                         f"{err.get('exception_message')}")
            lines.extend((err.get("traceback") or [])[-8:])
        else:
            lines.append(json.dumps(msg)[:2000])
    return "\n".join(lines)


def first_output_file(outputs: dict, keys: tuple[str, ...]) -> dict:
    for node_out in outputs.values():
        for key in keys:
            files = node_out.get(key)
            if files:
                return files[0]
    raise ComfyError(f"No output files (keys={keys}) in ComfyUI history: {list(outputs)}")


class ComfyClient:
    """Thin wrapper over the ComfyUI HTTP API (no workflow knowledge)."""

    def __init__(self, base_url: str, timeout: int = 60,
                 poll_interval: float = 2.0, queue_timeout: int = 5400):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.queue_timeout = queue_timeout

    def check_reachable(self, timeout: int = 10) -> None:
        try:
            with urllib.request.urlopen(f"{self.base_url}/system_stats",
                                        timeout=timeout) as resp:
                resp.read()
        except Exception as exc:
            raise ComfyError(f"ComfyUI not reachable at {self.base_url}: {exc}") from exc

    def queue(self, workflow: dict) -> str:
        result = _http_json(self.base_url, "POST", "/prompt",
                            {"prompt": workflow}, self.timeout)
        prompt_id = result.get("prompt_id")
        if not prompt_id:
            raise ComfyError(f"ComfyUI did not return a prompt_id: {result}")
        return prompt_id

    def cancel(self, prompt_id: str) -> None:
        """Cancel one prompt: always delete it from the queue; send the
        global /interrupt ONLY if this prompt is currently executing.
        Never blindly interrupt: an unconditional interrupt would kill
        unrelated running jobs (friendly fire)."""
        try:
            _http_json(self.base_url, "POST", "/queue",
                       {"delete": [prompt_id]}, self.timeout)
        except ComfyError:
            pass
        try:
            queue = _http_json(self.base_url, "GET", "/queue", self.timeout)
        except ComfyError:
            return
        running = queue.get("queue_running", []) if isinstance(queue, dict) else []
        ids = set()
        for entry in running:
            if isinstance(entry, dict) and entry.get("prompt_id"):
                ids.add(entry["prompt_id"])
            elif isinstance(entry, list) and len(entry) > 1 and isinstance(entry[1], str):
                ids.add(entry[1])
        if prompt_id in ids:
            try:
                _http_json(self.base_url, "POST", "/interrupt", {}, self.timeout)
            except ComfyError:
                pass

    def wait(self, prompt_id: str, workflow: dict | None = None,
             max_transport_retries: int = 10) -> dict:
        """Wait for completion. Job errors raise immediately with the full
        server error. Transient transport failures (reset/timeout while the
        server is busy) are retried with backoff, bounded by
        max_transport_retries — never silently forever."""
        deadline = time.time() + self.queue_timeout
        transport_failures = 0
        backoff = 2.0
        while True:
            try:
                history = _http_json(self.base_url, "GET",
                                     f"/history/{prompt_id}", self.timeout)
            except ComfyError as exc:
                transport_failures += 1
                if transport_failures > max_transport_retries:
                    raise ComfyError(
                        f"ComfyUI unreachable while waiting for job {prompt_id} "
                        f"({transport_failures} failed polls): {exc}") from exc
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                continue
            transport_failures = 0
            backoff = 2.0
            if prompt_id in history:
                entry = history[prompt_id]
                status = (entry.get("status") or {})
                if status.get("status_str") == "error":
                    raise ComfyError(format_job_error(prompt_id, entry, workflow))
                if status.get("status_str") == "interrupted":
                    raise ComfyError(f"ComfyUI job {prompt_id} was interrupted.")
                if status.get("completed") or entry.get("outputs"):
                    return entry
            if time.time() > deadline:
                raise ComfyError(f"Timed out waiting for ComfyUI prompt {prompt_id}")
            time.sleep(self.poll_interval)

    def download(self, file_info: dict, dest_path: str) -> str:
        params = urllib.parse.urlencode(
            {"filename": file_info["filename"],
             "subfolder": file_info.get("subfolder", ""),
             "type": file_info.get("type", "output")})
        os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
        try:
            with urllib.request.urlopen(f"{self.base_url}/view?{params}",
                                        timeout=self.timeout) as resp, \
                    open(dest_path, "wb") as f:
                f.write(resp.read())
        except Exception as exc:
            raise ComfyError(
                f"Failed to download {file_info.get('filename')}: {exc}") from exc
        return dest_path

    def upload_image(self, image_path: str) -> str:
        """Upload a local image to the ComfyUI input dir. Returns stored name."""
        return self._upload_file(image_path, "Image")

    def upload_video(self, video_path: str) -> str:
        """Upload a local motion clip to the ComfyUI input dir.

        Same multipart input upload as images; LoadVideo combo nodes read
        from the input dir, so the stored filename is what the workflow
        needs. Returns stored name.
        """
        return self._upload_file(video_path, "Video")

    def _upload_file(self, file_path: str, kind: str) -> str:
        boundary = "----ai-video-boundary"
        filename = os.path.basename(file_path)
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode() + file_bytes + f"\r\n--{boundary}--\r\n".encode()
        parsed = urlparse(self.base_url)
        conn_cls = (http.client.HTTPSConnection if parsed.scheme == "https"
                    else http.client.HTTPConnection)
        conn = conn_cls(parsed.hostname,
                        parsed.port or (443 if parsed.scheme == "https" else 80),
                        timeout=self.timeout)
        try:
            conn.request("POST", "/upload/image", body,
                         {"Content-Type": f"multipart/form-data; boundary={boundary}"})
            resp = conn.getresponse()
            raw = resp.read().decode("utf-8", "replace")
            if resp.status != 200:
                raise ComfyError(f"{kind} upload failed ({resp.status}): {raw[:1000]}")
            return json.loads(raw).get("name", filename)
        finally:
            conn.close()

    def run(self, workflow: dict, output_keys: tuple[str, ...], dest_path: str) -> str:
        """Submit workflow, wait, download first output file. Returns dest path."""
        prompt_id = self.queue(workflow)
        try:
            entry = self.wait(prompt_id, workflow)
        except ComfyError:
            raise
        info = first_output_file(entry.get("outputs", {}), output_keys)
        return self.download(info, dest_path)
