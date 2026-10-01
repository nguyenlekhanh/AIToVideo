"""Unit tests for the shared ComfyUI client (no server needed)."""
from __future__ import annotations

import unittest

from providers.comfy import ComfyClient, first_output_file, format_job_error
from providers.comfy import ComfyError


class FormatJobErrorTest(unittest.TestCase):
    def test_execution_error_includes_node_and_traceback(self):
        entry = {"status": {
            "status_str": "error",
            "completed": False,
            "messages": [
                ["execution_start", {"prompt_id": "abc"}],
                ["execution_error", {
                    "prompt_id": "abc",
                    "node_id": "1377",
                    "node_type": "LTXVConcatAVLatent",
                    "exception_message": "boom",
                    "exception_type": "RuntimeError",
                    "traceback": ["line1", "line2"],
                }],
            ],
        }}
        wf = {"1377": {"class_type": "LTXVConcatAVLatent", "inputs": {}}}
        text = format_job_error("abc", entry, wf)
        self.assertIn("1377", text)
        self.assertIn("LTXVConcatAVLatent", text)
        self.assertIn("RuntimeError", text)
        self.assertIn("boom", text)
        self.assertIn("line2", text)

    def test_first_output_file_prefers_key_order(self):
        outputs = {"9": {"gifs": [{"filename": "b.mp4"}]},
                   "7": {"images": [{"filename": "a.png"}]}}
        self.assertEqual(first_output_file(outputs, ("video", "gifs", "images"))["filename"], "b.mp4")
        self.assertEqual(first_output_file(outputs, ("images",))["filename"], "a.png")
        with self.assertRaises(ComfyError):
            first_output_file({"9": {}}, ("images",))


class ClientInitTest(unittest.TestCase):
    def test_strips_trailing_slash(self):
        self.assertEqual(ComfyClient("http://127.0.0.1:8188/").base_url,
                         "http://127.0.0.1:8188")


class CancelScopeTest(unittest.TestCase):
    def _client_with_log(self, running_ids):
        import providers.comfy as comfy_mod
        calls = []
        real = comfy_mod._http_json

        def fake(base_url, method, path, payload=None, timeout=60):
            calls.append((method, path, payload))
            if method == "GET" and path == "/queue":
                return {"queue_running": [{"prompt_id": pid} for pid in running_ids],
                        "queue_pending": []}
            return {}

        comfy_mod._http_json = fake
        self.addCleanup(setattr, comfy_mod, "_http_json", real)
        return ComfyClient("http://127.0.0.1:8188"), calls

    def test_interrupt_only_when_running(self):
        client, calls = self._client_with_log(["mine"])
        client.cancel("mine")
        methods = [(m, p) for m, p, _ in calls]
        self.assertIn(("POST", "/queue"), methods)  # delete always sent
        self.assertIn(("POST", "/interrupt"), methods)  # running -> interrupt

    def test_no_interrupt_for_other_jobs(self):
        client, calls = self._client_with_log(["someone-else"])
        client.cancel("mine")
        methods = [(m, p) for m, p, _ in calls]
        self.assertIn(("POST", "/queue"), methods)
        self.assertNotIn(("POST", "/interrupt"), methods)  # no friendly fire


class WaitRetryTest(unittest.TestCase):
    def test_transient_failures_retried_then_success(self):
        import providers.comfy as comfy_mod
        calls = {"n": 0}
        real = comfy_mod._http_json

        def flaky(base_url, method, path, payload=None, timeout=60):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise ComfyError("ConnectionResetError(10054)")
            return {"pid": {"status": {"status_str": "success", "completed": True},
                            "outputs": {"1": {"images": [{"filename": "a.png"}]}}}}

        comfy_mod._http_json = flaky
        try:
            client = ComfyClient("http://127.0.0.1:8188", poll_interval=0.01)
            entry = client.wait("pid")
        finally:
            comfy_mod._http_json = real
        self.assertTrue(entry["status"]["completed"])
        self.assertEqual(calls["n"], 3)

    def test_persistent_outage_fails_loud(self):
        import providers.comfy as comfy_mod
        real = comfy_mod._http_json

        def dead(base_url, method, path, payload=None, timeout=60):
            raise ComfyError("refused")

        comfy_mod._http_json = dead
        try:
            client = ComfyClient("http://127.0.0.1:8188", poll_interval=0.01)
            with self.assertRaises(ComfyError) as ctx:
                client.wait("pid", max_transport_retries=2)
        finally:
            comfy_mod._http_json = real
        self.assertIn("unreachable", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
