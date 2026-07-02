from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
from io import BytesIO
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import llamawrap_core as core  # noqa: E402


class FakeResponse:
    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self.body = body.encode("utf-8")

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def make_preset(model_path: str = "/models/demo.gguf", port: int = 8080) -> dict:
    flags = core.default_cli_flags()
    flags["--host"]["value"] = "127.0.0.1"
    flags["--host"]["enabled"] = True
    flags["--port"]["value"] = str(port)
    flags["--port"]["enabled"] = True
    flags["-ngl"]["value"] = "99"
    flags["-ngl"]["enabled"] = True
    return {
        "format_version": 1,
        "preset_name": "Demo",
        "inferer": "llama.cpp",
        "inferer_executable": "/bin/echo",
        "model_path": model_path,
        "mmproj_path": "",
        "draft_model_path": "",
        "extra_args": "--no-webui",
        "hidden_flags": [],
        "flags": flags,
    }


class CoreTests(unittest.TestCase):
    def test_build_command_from_preset(self) -> None:
        command = core.build_command_from_preset(make_preset())
        self.assertEqual(command[:3], ["/bin/echo", "-m", "/models/demo.gguf"])
        self.assertIn("--fit", command)
        self.assertIn("off", command)
        self.assertIn("-ngl", command)
        self.assertIn("--metrics", command)
        self.assertIn("--no-webui", command)

    def test_import_export_round_trip_and_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "presets.json"
            data = {"format_version": 1, "presets": [make_preset()], "runs": [], "settings": {}}
            warnings = core.export_presets(data, out, portable=True)
            self.assertTrue(out.exists())
            self.assertTrue(any("absolute path" in warning for warning in warnings))
            imported = {"format_version": 1, "presets": [], "runs": [], "settings": {}}
            count, skipped = core.import_presets(imported, out)
            self.assertEqual(count, 1)
            self.assertEqual(skipped, [])
            self.assertEqual(imported["presets"][0]["preset_name"], "Demo")

    def test_selected_export_requires_existing_preset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "preset not found"):
                core.export_presets({"presets": []}, Path(tmp) / "x.json", names=["Missing"])

    def test_endpoint_url(self) -> None:
        self.assertEqual(core.endpoint_url("127.0.0.1", 8080, "health"), "http://127.0.0.1:8080/health")
        self.assertEqual(core.endpoint_url("localhost", 1234, "/v1/models"), "http://localhost:1234/v1/models")

    def test_parse_chat_response(self) -> None:
        content, tokens = core.parse_chat_response({
            "choices": [{"message": {"content": "hello world"}}],
            "usage": {"completion_tokens": 2},
        })
        self.assertEqual(content, "hello world")
        self.assertEqual(tokens, 2)

    def test_http_json_success(self) -> None:
        with mock.patch("urllib.request.urlopen", return_value=FakeResponse(200, '{"ok": true}')):
            result = core.http_json("GET", "http://local/health")
        self.assertTrue(result.ok)
        self.assertEqual(result.status, 200)
        self.assertEqual(result.data["ok"], True)

    def test_http_json_malformed_json(self) -> None:
        with mock.patch("urllib.request.urlopen", return_value=FakeResponse(200, 'not-json')):
            result = core.http_json("GET", "http://local/health")
        self.assertFalse(result.ok)
        self.assertIn("malformed JSON", result.error)

    def test_http_json_http_error(self) -> None:
        error = urllib.error.HTTPError("http://local/health", 500, "server error", {}, BytesIO(b"server error"))
        with mock.patch("urllib.request.urlopen", side_effect=error):
            result = core.http_json("GET", "http://local/health")
        self.assertFalse(result.ok)
        self.assertEqual(result.status, 500)

    def test_http_json_connection_failure(self) -> None:
        with mock.patch("urllib.request.urlopen", side_effect=ConnectionRefusedError("refused")):
            result = core.http_json("GET", "http://local/health")
        self.assertFalse(result.ok)
        self.assertIsNone(result.status)
        self.assertIn("refused", result.error)

    def test_http_json_timeout(self) -> None:
        with mock.patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            result = core.http_json("GET", "http://local/health")
        self.assertFalse(result.ok)
        self.assertIsNone(result.status)
        self.assertIn("timed out", result.error)

    def test_probe_report_rejects_bad_payload(self) -> None:
        with mock.patch("llamawrap_core.http_json", return_value=core.EndpointResult(True, 200, "url", data={"choices": []})):
            ok, lines = core.probe_report(make_preset())
        self.assertFalse(ok)
        self.assertTrue(any("response parsed" in line for line in lines))

    def test_runtime_context_detection_prefers_manual_override(self) -> None:
        preset = make_preset(port=1234)
        preset["flags"]["-c"]["value"] = "12000"
        with mock.patch("llamawrap_core.http_json") as mocked:
            detected = core.detect_runtime_context(preset)
        mocked.assert_not_called()
        self.assertEqual(detected["effective_context"], 12000)
        self.assertEqual(detected["confidence"], "manual")

    def test_runtime_context_detection_uses_llama_slots(self) -> None:
        preset = make_preset(port=1234)
        preset["flags"]["-c"]["value"] = ""

        def fake_http(method: str, url: str, payload: dict | None = None, timeout: float = 0) -> core.EndpointResult:
            if url.endswith("/slots"):
                return core.EndpointResult(True, 200, url, data=[{"n_ctx": 8192, "n_tokens": 1024}])
            return core.EndpointResult(False, 404, url, error="not found")

        with mock.patch("llamawrap_core.http_json", side_effect=fake_http):
            detected = core.detect_runtime_context(preset)
        self.assertEqual(detected["effective_context"], 8192)
        self.assertEqual(detected["current_active_context"], 1024)
        self.assertEqual(detected["confidence"], "confirmed")

    def test_runtime_context_detection_unknown_without_context_source(self) -> None:
        preset = make_preset(port=1234)
        preset["flags"]["-c"]["value"] = ""
        with mock.patch("llamawrap_core.http_json", return_value=core.EndpointResult(False, 404, "url", error="not found")):
            detected = core.detect_runtime_context(preset)
        self.assertIsNone(detected["effective_context"])
        self.assertEqual(detected["confidence"], "unknown")

    def test_stress_percentage_calculation_protects_output_reserve(self) -> None:
        self.assertEqual(core.stress_stage_prompt_tokens(1000, 96, 200), 768)
        self.assertEqual(core.stress_stage_prompt_tokens(1000, 30, 200), 300)

    def test_context_stress_runs_percentage_modes_and_sustained_agent(self) -> None:
        preset = make_preset(port=1234)
        preset["flags"]["-c"]["value"] = "1000"
        chat_payloads = []

        def fake_http(method: str, url: str, payload: dict | None = None, timeout: float = 0) -> core.EndpointResult:
            if url.endswith("/tokenize"):
                return core.EndpointResult(True, 200, url, data={"tokens": list(range(len(payload["content"].split())))})
            chat_payloads.append(payload)
            return core.EndpointResult(True, 200, url, data={
                "choices": [{"message": {"content": "OK"}}],
                "usage": {"prompt_tokens": len(payload["messages"][0]["content"].split()), "completion_tokens": 10},
            })

        with mock.patch("llamawrap_core.http_json", side_effect=fake_http):
            ok, lines = core.context_stress_report(
                preset,
                max_prompt_tokens=500,
                stage_percents=(10, 30),
                boundary_percents=(85,),
                agent_turns=2,
            )

        self.assertTrue(ok)
        self.assertEqual(len(chat_payloads), 5)
        self.assertTrue(any("Mode: Fill and decode" in line for line in lines))
        self.assertTrue(any("Mode: Sustained agent" in line for line in lines))
        self.assertTrue(any("Mode: Boundary probe" in line for line in lines))
        self.assertTrue(any("Recommended practical working limit" in line for line in lines))
        self.assertLessEqual(max(len(p["messages"][0]["content"].split()) for p in chat_payloads), 500)

    def test_context_stress_records_context_overflow_and_continues(self) -> None:
        preset = make_preset(port=1234)
        preset["flags"]["-c"]["value"] = "1000"
        calls = {"chat": 0}

        def fake_http(method: str, url: str, payload: dict | None = None, timeout: float = 0) -> core.EndpointResult:
            if url.endswith("/tokenize"):
                return core.EndpointResult(True, 200, url, data={"tokens": list(range(len(payload["content"].split())))})
            calls["chat"] += 1
            if calls["chat"] == 1:
                return core.EndpointResult(False, 400, url, error=json.dumps({
                    "error": {
                        "type": "exceed_context_size_error",
                        "n_prompt_tokens": 2000,
                        "n_ctx": 1000,
                    }
                }))
            return core.EndpointResult(True, 200, url, data={
                "choices": [{"message": {"content": "OK"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1},
            })

        with mock.patch("llamawrap_core.http_json", side_effect=fake_http):
            ok, lines = core.context_stress_report(
                preset,
                max_prompt_tokens=256,
                stage_percents=(96,),
                boundary_percents=(85,),
                agent_turns=1,
            )

        self.assertTrue(ok)
        self.assertGreater(calls["chat"], 1)
        self.assertTrue(any("context overflow" in line for line in lines))

    def test_context_stress_reports_bad_payload(self) -> None:
        def fake_http(method: str, url: str, payload: dict | None = None, timeout: float = 0) -> core.EndpointResult:
            if url.endswith("/tokenize"):
                return core.EndpointResult(True, 200, url, data={"tokens": [1, 2]})
            return core.EndpointResult(True, 200, url, data={"choices": []})

        with mock.patch("llamawrap_core.http_json", side_effect=fake_http):
            ok, lines = core.context_stress_report(make_preset(), max_prompt_tokens=128, stage_percents=(10,), boundary_percents=(), agent_turns=0)
        self.assertFalse(ok)
        self.assertTrue(any("FAIL" in line for line in lines))

    def test_vram_log_parser_reads_gpu_buffers(self) -> None:
        self.assertEqual(
            core.parse_llama_vram_log_line("llama_kv_cache_init:      CUDA0 KV buffer size = 1024.00 MiB"),
            ("kv", 1024 * 1024 * 1024),
        )
        self.assertEqual(
            core.parse_llama_vram_log_line("llama_new_context_with_model: CUDA0 compute buffer size = 256.50 MiB"),
            ("compute", int(256.5 * 1024 * 1024)),
        )
        self.assertIsNone(
            core.parse_llama_vram_log_line("llama_new_context_with_model: CPU compute buffer size = 256.50 MiB")
        )

    def test_metrics_skipped_for_custom_inferer(self) -> None:
        preset = make_preset()
        preset["inferer"] = "Custom"
        preset["inferer_executable"] = "/usr/bin/other-server"
        command = core.build_command_from_preset(preset)
        self.assertNotIn("--metrics", command)

    def test_metrics_added_for_llama_server_executable(self) -> None:
        preset = make_preset()
        preset["inferer"] = "Custom"
        preset["inferer_executable"] = "/opt/llama/llama-server"
        command = core.build_command_from_preset(preset)
        self.assertIn("--metrics", command)

    def test_import_unknown_flag_with_negative_value(self) -> None:
        preset, _changed, _skipped = core.preset_from_command("neg", "llama-server -m /m.gguf --seed -1")
        self.assertIn("--seed", preset["flags"])
        self.assertEqual(preset["flags"]["--seed"]["value"], "-1")
        self.assertNotIn("-1", preset["flags"])

    def test_save_history_is_atomic_and_creates_parents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "history.json"
            core.save_history(path, {"presets": [], "runs": [], "settings": {}})
            self.assertTrue(path.exists())
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["format_version"], 1)
            self.assertFalse(path.with_name("history.json.tmp").exists())

    def test_find_history_env_override_and_platform_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_hist = Path(tmp) / "custom.json"
            with mock.patch.dict(os.environ, {"LLAMA_WRAP_HISTORY": str(env_hist)}):
                self.assertEqual(core.find_history(), env_hist)
            empty = Path(tmp) / "empty"
            empty.mkdir()
            with mock.patch.dict(os.environ):
                os.environ.pop("LLAMA_WRAP_HISTORY", None)
                with mock.patch.object(core, "APP_DIR", empty):
                    found = core.find_history(cwd=empty)
            self.assertEqual(found, core.platform_data_dir() / "history.json")

    def test_classify_stress_error_categories(self) -> None:
        cases = [
            (core.EndpointResult(False, None, "u", error="connection reset by peer"), "connection reset"),
            (core.EndpointResult(False, None, "u", error="timed out"), "timeout"),
            (core.EndpointResult(False, 500, "u", error="CUDA error: out of memory"), "OOM indication"),
            (core.EndpointResult(False, 404, "u", error="not found"), "HTTP error"),
            (core.EndpointResult(False, None, "u", error="mystery"), "unknown"),
        ]
        for result, expected in cases:
            self.assertEqual(core.classify_stress_error(result), expected)
        overflow = core.EndpointResult(False, 400, "u", error=json.dumps({
            "error": {"type": "exceed_context_size_error", "n_prompt_tokens": 2000, "n_ctx": 1000}
        }))
        self.assertEqual(core.classify_stress_error(overflow), "context overflow")

    def test_probe_report_streams_lines_to_callback(self) -> None:
        seen: list[str] = []
        with mock.patch("llamawrap_core.http_json", return_value=core.EndpointResult(False, None, "url", error="refused")):
            _ok, lines = core.probe_report(make_preset(), on_line=seen.append)
        self.assertEqual(seen, lines)

    def test_context_stress_cancel_before_first_request(self) -> None:
        class Cancelled:
            def is_set(self) -> bool:
                return True

        preset = make_preset(port=1234)
        preset["flags"]["-c"]["value"] = "1000"
        with mock.patch("llamawrap_core.http_json") as mocked:
            ok, lines = core.context_stress_report(preset, cancel=Cancelled())
        mocked.assert_not_called()
        self.assertFalse(ok)
        self.assertIn("Stress result: CANCELLED", lines)

    def test_run_benchmark_aggregates_streamed_iterations(self) -> None:
        preset = make_preset(port=1234)
        stream_result = {
            "ok": True, "status": 200, "error": "", "ttft_ms": 100.0, "total_ms": 1100.0,
            "tokens": 11, "content": "hello world",
            "timings": {"prompt_ms": 90.0, "prompt_per_second": 500.0, "predicted_per_second": 10.0},
            "usage": {"completion_tokens": 11, "prompt_tokens": 12},
        }
        warmup = core.EndpointResult(True, 200, "url", data={"choices": [{"message": {"content": "OK"}}]})
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("llamawrap_core.http_json", return_value=warmup), \
                 mock.patch("llamawrap_core.http_stream_chat", return_value=dict(stream_result)):
                row, paths, lines = core.run_benchmark(preset, out_dir=Path(tmp))
            self.assertEqual(row["status"], "pass")
            self.assertEqual(row["iterations"], 3)
            self.assertEqual(row["ttft_ms"], 90.0)
            self.assertEqual(row["generation_tokens_per_second"], 10.0)
            self.assertEqual(row["prefill_tokens_per_second"], 500.0)
            self.assertEqual(row["tokens_per_second"], 10.0)
            self.assertTrue(paths)
            self.assertTrue(any("[PASS] benchmark" in line for line in lines))

    def test_run_benchmark_reports_warmup_failure(self) -> None:
        preset = make_preset(port=1234)
        warmup = core.EndpointResult(False, None, "url", error="refused")
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("llamawrap_core.http_json", return_value=warmup), \
                 mock.patch("llamawrap_core.http_stream_chat") as stream:
                row, _paths, lines = core.run_benchmark(preset, out_dir=Path(tmp))
            stream.assert_not_called()
            self.assertEqual(row["status"], "fail")
            self.assertIn("refused", row["error"])
            self.assertTrue(any("Benchmark request failed." in line for line in lines))

    def test_vram_calibration_matches_command_and_model_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_bytes(b"GGUF")
            command = ["/bin/echo", "-m", str(model), "--metrics"]
            calibration = core.make_vram_calibration(
                preset_name="Demo",
                command=command,
                model_path=str(model),
                estimated_bytes=100,
                observed_bytes=125,
                observed_source="test",
                log_allocations={"model": 64},
            )
            self.assertTrue(core.calibration_matches(calibration, command, str(model)))
            self.assertEqual(calibration["correction_ratio"], 1.25)
            self.assertFalse(core.calibration_matches(calibration, command + ["--no-webui"], str(model)))


def _gguf_bytes(pairs: list[tuple[str, int]]) -> bytes:
    """Minimal GGUF v3 header with uint32 metadata values."""

    def kv(key: str, value: int) -> bytes:
        encoded = key.encode("utf-8")
        return struct.pack("<Q", len(encoded)) + encoded + struct.pack("<I", 4) + struct.pack("<I", value)

    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(pairs))
    return header + b"".join(kv(key, value) for key, value in pairs)


class GGUFParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import llamawrap as gui  # tk import is guarded, safe headless

        cls.gui = gui

    def test_parse_gguf_reads_core_metadata(self) -> None:
        pairs = [
            ("llama.block_count", 32),
            ("llama.embedding_length", 4096),
            ("llama.attention.head_count_kv", 8),
            ("llama.attention.key_length", 128),
            ("llama.attention.value_length", 128),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.gguf"
            path.write_bytes(_gguf_bytes(pairs))
            meta = self.gui.parse_gguf(str(path))
        self.assertEqual(meta.n_layers, 32)
        self.assertEqual(meta.n_embd, 4096)
        self.assertEqual(meta.n_kv_heads, 8)
        self.assertEqual(meta.n_embd_k_gqa, 128)
        self.assertEqual(meta.n_embd_v_gqa, 128)
        self.assertEqual(meta.warnings, [])

    def test_parse_gguf_skips_irrelevant_keys(self) -> None:
        pairs = [
            ("general.some_other_key", 7),
            ("llama.block_count", 16),
            ("llama.embedding_length", 2048),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.gguf"
            path.write_bytes(_gguf_bytes(pairs))
            meta = self.gui.parse_gguf(str(path))
        self.assertEqual(meta.n_layers, 16)
        self.assertEqual(meta.n_embd, 2048)

    def test_parse_gguf_non_gguf_file_warns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.gguf"
            path.write_bytes(b"this is not a gguf file")
            meta = self.gui.parse_gguf(str(path))
        self.assertEqual(meta.n_layers, 0)
        self.assertTrue(meta.warnings)

    def test_parse_gguf_truncated_metadata_warns_instead_of_crashing(self) -> None:
        pairs = [("llama.block_count", 32)]
        data = _gguf_bytes(pairs)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.gguf"
            truncated = data[:-2]
            # Claim more metadata entries than the file contains.
            truncated = truncated[:16] + struct.pack("<Q", 5) + truncated[24:]
            path.write_bytes(truncated)
            meta = self.gui.parse_gguf(str(path))
        self.assertTrue(meta.warnings)


class LocalOpenAIHandler(BaseHTTPRequestHandler):
    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        elif self.path == "/v1/models":
            self._json(200, {"data": [{"id": "local"}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path == "/tokenize":
            length = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                payload = {}
            content = str(payload.get("content", ""))
            self._json(200, {"tokens": list(range(len(content.split())))})
            return
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {}
        if payload.get("stream"):
            chunks = [
                {"choices": [{"delta": {"content": "OK"}}]},
                {"choices": [{"delta": {"content": " done"}}]},
                {
                    "choices": [{"delta": {}}],
                    "usage": {"completion_tokens": 2, "prompt_tokens": 5},
                    "timings": {
                        "prompt_ms": 1.0, "prompt_per_second": 5000.0,
                        "predicted_n": 2, "predicted_ms": 2.0, "predicted_per_second": 1000.0,
                    },
                },
            ]
            sse = b"".join(b"data: " + json.dumps(chunk).encode("utf-8") + b"\n\n" for chunk in chunks)
            sse += b"data: [DONE]\n\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(sse)))
            self.end_headers()
            self.wfile.write(sse)
            return
        self._json(200, {
            "choices": [{"message": {"content": "OK"}}],
            "usage": {"completion_tokens": 1},
        })

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class CliSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), LocalOpenAIHandler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.thread.join(timeout=2)
        cls.server.server_close()

    def write_history(self, directory: Path) -> Path:
        model = directory / "model.gguf"
        model.write_bytes(b"GGUF")
        history = directory / "history.json"
        history.write_text(json.dumps({
            "format_version": 1,
            "presets": [make_preset(str(model), self.port)],
            "runs": [],
            "settings": {},
        }), encoding="utf-8")
        return history

    def run_cli(self, history: Path, *args: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["LLAMA_WRAP_HISTORY"] = str(history)
        return subprocess.run(
            [sys.executable, str(ROOT / "llamawrap-cli.py"), *args],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=10,
        )

    def test_cli_doctor_probe_bench_export_import_presets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            history = self.write_history(temp)

            doctor = self.run_cli(history, "doctor", "Demo")
            self.assertEqual(doctor.returncode, 0, doctor.stderr + doctor.stdout)
            self.assertIn("Doctor result: PASS", doctor.stdout)

            probe = self.run_cli(history, "probe", "Demo")
            self.assertEqual(probe.returncode, 0, probe.stderr + probe.stdout)
            self.assertIn("Probe result: PASS", probe.stdout)

            bench_dir = temp / "bench"
            bench = self.run_cli(history, "bench", "Demo", "--csv", "--out-dir", str(bench_dir))
            self.assertEqual(bench.returncode, 0, bench.stderr + bench.stdout)
            self.assertEqual(len(list(bench_dir.glob("*.json"))), 1)
            self.assertEqual(len(list(bench_dir.glob("*.csv"))), 1)

            stress = self.run_cli(history, "stress", "Demo")
            self.assertEqual(stress.returncode, 0, stress.stderr + stress.stdout)
            self.assertIn("Stress result: PASS", stress.stdout)
            self.assertIn("Mode: Sustained agent", stress.stdout)

            exported = temp / "export.json"
            export = self.run_cli(history, "export-presets", "--out", str(exported), "--portable")
            self.assertEqual(export.returncode, 0, export.stderr + export.stdout)
            self.assertTrue(exported.exists())

            imported_history = temp / "imported-history.json"
            imported_history.write_text('{"format_version": 1, "presets": [], "runs": [], "settings": {}}', encoding="utf-8")
            import_result = self.run_cli(imported_history, "import-presets", str(exported))
            self.assertEqual(import_result.returncode, 0, import_result.stderr + import_result.stdout)
            data = json.loads(imported_history.read_text(encoding="utf-8"))
            self.assertEqual(data["presets"][0]["preset_name"], "Demo")


if __name__ == "__main__":
    unittest.main()
