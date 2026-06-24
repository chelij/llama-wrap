from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import shlex
import shutil
import socket
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


APP_DIR = Path(__file__).resolve().parent
LOCAL_DATA_DIR = ".llama-wrap"
BENCHMARK_DIR = "benchmarks"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_TIMEOUT = 5.0
BENCH_PROMPT = "Reply with one short sentence about local AI diagnostics."
STRESS_CONTEXT_FRACTION = 0.70
STRESS_MAX_PROMPT_TOKENS = 262_144
STRESS_MAX_OUTPUT_TOKENS = 2048
STRESS_STAGE_PERCENTS = (10, 30, 60, 70, 75, 80, 85, 90, 92, 96)
STRESS_BOUNDARY_PERCENTS = (85, 90, 92, 94, 96, 98)
STRESS_OUTPUT_RESERVE_PERCENT = 5
STRESS_AGENT_TURNS = 40
STRESS_AGENT_SEED = 1337
SIZE_RE = r"([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?i?B)"


ALIAS_TO_FLAG = {
    "--model": "-m",
    "-m": "-m",
    "--gpu-layers": "-ngl",
    "--n-gpu-layers": "-ngl",
    "-ngl": "-ngl",
    "--ctx-size": "-c",
    "-c": "-c",
    "--threads": "-t",
    "-t": "-t",
    "--threads-batch": "-tb",
    "-tb": "-tb",
    "--flash-attn": "-fa",
    "-fa": "-fa",
    "--cache-type-k": "-ctk",
    "-ctk": "-ctk",
    "--cache-type-v": "-ctv",
    "-ctv": "-ctv",
    "--port": "--port",
    "--host": "--host",
    "--batch-size": "-b",
    "-b": "-b",
    "--ubatch-size": "-ub",
    "-ub": "-ub",
    "--parallel": "-np",
    "-np": "-np",
    "--alias": "-a",
    "-a": "-a",
    "--timeout": "-to",
    "-to": "-to",
    "--mmproj": "--mmproj",
    "-mm": "--mmproj",
    "--spec-draft-model": "-md",
    "--model-draft": "-md",
    "-md": "-md",
    "--spec-type": "--spec-type",
    "--spec-draft-n-max": "--spec-draft-n-max",
    "--draft-max": "--spec-draft-n-max",
    "--draft": "--spec-draft-n-max",
    "--spec-draft-n-min": "--spec-draft-n-min",
    "--draft-min": "--spec-draft-n-min",
    "--spec-draft-p-min": "--spec-draft-p-min",
    "--draft-p-min": "--spec-draft-p-min",
    "--spec-draft-ngl": "-ngld",
    "--gpu-layers-draft": "-ngld",
    "--n-gpu-layers-draft": "-ngld",
    "-ngld": "-ngld",
    "--jinja": "--jinja",
    "--fit": "--fit",
    "--fit-margin": "--fit-margin",
    "-mla": "-mla",
    "--mla-use": "-mla",
    "-fmoe": "-fmoe",
    "--fused-moe": "-fmoe",
    "-cram": "-cram",
    "--cache-ram": "-cram",
    "-khad": "-khad",
    "--k-cache-hadamard": "-khad",
    "-vhad": "-vhad",
    "--v-cache-hadamard": "-vhad",
    "-ncmoe": "-ncmoe",
    "--n-cpu-moe": "-ncmoe",
    "--cpu-moe": "-ncmoe",
    "--reasoning": "--reasoning",
    "-rea": "--reasoning",
}


@dataclass
class EndpointResult:
    ok: bool
    status: int | None
    url: str
    data: Any = None
    error: str = ""
    elapsed_ms: float = 0.0


def find_history(argv0: str | None = None, cwd: Path | None = None) -> Path:
    env_path = os.environ.get("LLAMA_WRAP_HISTORY")
    if env_path:
        return Path(env_path).expanduser()
    candidates = [
        Path(argv0).resolve().parent if argv0 and getattr(sys, "frozen", False) else None,
        APP_DIR,
        cwd or Path.cwd(),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        path = candidate / "history.json"
        if path.exists():
            return path
    return (cwd or Path.cwd()) / "history.json"


def load_history(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    return json.loads(path.read_text(encoding="utf-8"))


def load_or_init_history(path: Path) -> dict[str, Any]:
    if path.exists():
        return load_history(path)
    return {"format_version": 1, "presets": [], "runs": [], "settings": {}}


def normalize_history(data: dict[str, Any]) -> dict[str, Any]:
    data["format_version"] = data.get("format_version", 1)
    data["presets"] = data.get("presets", [])
    data["runs"] = (data.get("runs") or [])[-100:]
    data["settings"] = data.get("settings", {})
    return data


def save_history(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalize_history(data), indent=2), encoding="utf-8")


def find_preset(data: dict[str, Any], name: str) -> dict[str, Any] | None:
    for preset in data.get("presets", []):
        if preset.get("preset_name") == name:
            return preset
    return None


def upsert_preset(data: dict[str, Any], preset: dict[str, Any]) -> None:
    name = preset.get("preset_name", "")
    presets = [p for p in data.get("presets", []) if p.get("preset_name") != name]
    presets.append(preset)
    data["presets"] = sorted(presets, key=lambda p: p.get("preset_name", "").lower())


def default_cli_flags() -> dict[str, dict[str, Any]]:
    return {
        "-ngl": {"value": "0", "enabled": False, "value_required": True, "custom": False, "step_mode": ""},
        "--fit": {"value": "on", "enabled": False, "value_required": True, "custom": False, "step_mode": ""},
        "--fit-margin": {"value": "1024", "enabled": False, "value_required": True, "custom": False, "step_mode": ""},
        "-cram": {"value": "8192", "enabled": False, "value_required": True, "custom": False, "step_mode": ""},
        "-ncmoe": {"value": "", "enabled": False, "value_required": True, "custom": False, "step_mode": ""},
        "-c": {"value": "4096", "enabled": False, "value_required": True, "custom": False, "step_mode": "context"},
        "-ctk": {"value": "f16", "enabled": False, "value_required": True, "custom": False, "step_mode": ""},
        "-ctv": {"value": "f16", "enabled": False, "value_required": True, "custom": False, "step_mode": ""},
        "-khad": {"value": "", "enabled": False, "value_required": False, "custom": False, "step_mode": ""},
        "-vhad": {"value": "", "enabled": False, "value_required": False, "custom": False, "step_mode": ""},
        "-t": {"value": "-1", "enabled": False, "value_required": True, "custom": False, "step_mode": ""},
        "-tb": {"value": "-1", "enabled": False, "value_required": True, "custom": False, "step_mode": ""},
        "-fa": {"value": "auto", "enabled": False, "value_required": True, "custom": False, "step_mode": ""},
        "-mla": {"value": "0", "enabled": False, "value_required": True, "custom": False, "step_mode": ""},
        "-fmoe": {"value": "", "enabled": False, "value_required": False, "custom": False, "step_mode": ""},
        "--port": {"value": str(DEFAULT_PORT), "enabled": False, "value_required": True, "custom": False, "step_mode": ""},
        "--host": {"value": DEFAULT_HOST, "enabled": False, "value_required": True, "custom": False, "step_mode": ""},
        "-np": {"value": "1", "enabled": False, "value_required": True, "custom": False, "step_mode": ""},
        "-a": {"value": "", "enabled": False, "value_required": True, "custom": False, "step_mode": ""},
        "-to": {"value": "600", "enabled": False, "value_required": True, "custom": False, "step_mode": ""},
        "-b": {"value": "2048", "enabled": False, "value_required": True, "custom": False, "step_mode": ""},
        "-ub": {"value": "512", "enabled": False, "value_required": True, "custom": False, "step_mode": ""},
        "--spec-type": {"value": "none", "enabled": False, "value_required": True, "custom": False, "step_mode": ""},
        "--spec-draft-n-max": {"value": "16", "enabled": False, "value_required": True, "custom": False, "step_mode": ""},
        "--spec-draft-n-min": {"value": "0", "enabled": False, "value_required": True, "custom": False, "step_mode": ""},
        "--spec-draft-p-min": {"value": "0.75", "enabled": False, "value_required": True, "custom": False, "step_mode": ""},
        "-ngld": {"value": "0", "enabled": False, "value_required": True, "custom": False, "step_mode": ""},
        "--reasoning": {"value": "auto", "enabled": False, "value_required": True, "custom": False, "step_mode": ""},
        "--jinja": {"value": "", "enabled": False, "value_required": False, "custom": False, "step_mode": ""},
        "--mmproj": {"value": "", "enabled": False, "value_required": True, "custom": False, "step_mode": ""},
    }


def consume_import_value(tokens: list[str], idx: int, inline_value: str | None, flag: str) -> tuple[str, int]:
    if inline_value is not None:
        return inline_value, idx
    if idx + 1 >= len(tokens):
        raise ValueError(f"{flag} needs a value.")
    return tokens[idx + 1], idx + 1


def preset_from_command(name: str, command_text: str) -> tuple[dict[str, Any], int, list[str]]:
    if not command_text.strip():
        raise ValueError("Paste a server command first.")
    try:
        tokens = shlex.split(command_text)
    except ValueError as exc:
        raise ValueError(f"Could not read the command: {exc}") from exc
    if not tokens:
        raise ValueError("Paste a server command first.")

    executable = "llama-server"
    if tokens[0] and not tokens[0].startswith("-"):
        executable = tokens[0]
        tokens = tokens[1:]

    flags = default_cli_flags()
    valueless_flags = {flag for flag, cfg in flags.items() if not cfg.get("value_required", True)}
    model_path = ""
    mmproj_path = ""
    draft_model_path = ""
    extra_tokens: list[str] = []
    skipped: list[str] = []
    changed = 0
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token.startswith("--") and "=" in token:
            token, inline_value = token.split("=", 1)
        else:
            inline_value = None

        if token in {"-m", "--model"}:
            model_path, idx = consume_import_value(tokens, idx, inline_value, token)
            changed += 1
        elif token in ALIAS_TO_FLAG:
            flag = ALIAS_TO_FLAG[token]
            if flag in valueless_flags:
                value = inline_value or ""
            else:
                value, idx = consume_import_value(tokens, idx, inline_value, token)
            if flag == "--mmproj":
                mmproj_path = value
                flags["--mmproj"]["value"] = value
                flags["--mmproj"]["enabled"] = bool(value)
            elif flag == "-md":
                draft_model_path = value
            elif flag in flags:
                flags[flag]["value"] = value
                flags[flag]["enabled"] = True
            else:
                skipped.append(token)
                extra_tokens.extend([token, value])
            changed += 1
        elif token.startswith("-"):
            value = inline_value or ""
            value_required = inline_value is not None
            if inline_value is None and idx + 1 < len(tokens) and not tokens[idx + 1].startswith("-"):
                idx += 1
                value = tokens[idx]
                value_required = True
            flags[token] = {
                "value": value,
                "enabled": True,
                "value_required": value_required,
                "custom": True,
                "step_mode": "",
            }
            changed += 1
        else:
            extra_tokens.append(token)
        idx += 1

    preset = {
        "format_version": 1,
        "preset_name": name.strip(),
        "inferer": "llama.cpp" if Path(executable).name == "llama-server" else "Custom",
        "inferer_executable": executable,
        "model_path": model_path,
        "mmproj_path": mmproj_path,
        "draft_model_path": draft_model_path,
        "extra_args": shlex.join(extra_tokens) if extra_tokens else "",
        "hidden_flags": [],
        "flags": flags,
    }
    return preset, changed, skipped


def build_command_from_preset(preset: dict[str, Any], *, validate_paths: bool = False) -> list[str]:
    executable = preset.get("inferer_executable", "llama-server")
    try:
        command = shlex.split(executable)
    except ValueError as exc:
        raise ValueError(f"invalid executable in preset: {exc}") from exc
    if not command:
        raise ValueError("executable is required")

    model_path = (preset.get("model_path") or "").strip()
    if not model_path:
        raise ValueError("preset has no model path set")
    if validate_paths and not Path(model_path).expanduser().exists():
        raise ValueError("model path does not exist")
    command.extend(["-m", model_path])

    draft_path = (preset.get("draft_model_path") or "").strip()
    if draft_path:
        if validate_paths and not Path(draft_path).expanduser().exists():
            raise ValueError("draft model path does not exist")
        command.extend(["-md", draft_path])

    flags = preset.get("flags", {})
    fit_enabled = any(
        flag == "--fit" and isinstance(cfg, dict) and cfg.get("enabled", False)
        for flag, cfg in flags.items()
    )
    ngl_enabled = any(
        flag == "-ngl" and isinstance(cfg, dict) and cfg.get("enabled", False) and cfg.get("value", "").strip()
        for flag, cfg in flags.items()
    )
    for flag, cfg in sorted(flags.items()):
        if not isinstance(cfg, dict) or not cfg.get("enabled", False):
            continue
        if fit_enabled and flag == "-ngl":
            continue
        if flag == "-ngl" and ngl_enabled and not fit_enabled:
            command.extend(["--fit", "off"])
        if flag == "--mmproj" and not cfg.get("value", "").strip():
            continue
        if cfg.get("value_required", True):
            value = cfg.get("value", "").strip()
            if value:
                command.extend([flag, value])
        else:
            command.append(flag)

    extra = (preset.get("extra_args") or "").strip()
    if extra:
        try:
            command.extend(shlex.split(extra))
        except ValueError as exc:
            raise ValueError(f"invalid extra_args in preset: {exc}") from exc

    if "--metrics" not in command:
        command.append("--metrics")
    return command


def preset_endpoint(preset: dict[str, Any]) -> tuple[str, int]:
    flags = preset.get("flags", {})
    host = DEFAULT_HOST
    port = DEFAULT_PORT
    if isinstance(flags.get("--host"), dict) and str(flags["--host"].get("value", "")).strip():
        host = str(flags["--host"].get("value", "")).strip()
    if isinstance(flags.get("--port"), dict) and str(flags["--port"].get("value", "")).strip():
        try:
            port = int(str(flags["--port"].get("value", "")).strip())
        except ValueError:
            port = DEFAULT_PORT
    return host, port


def preset_context_size(preset: dict[str, Any], default: int = 4096) -> int:
    flags = preset.get("flags", {})
    cfg = flags.get("-c")
    if isinstance(cfg, dict):
        value = str(cfg.get("value", "") or "").strip()
        if value:
            try:
                parsed = int(value)
            except ValueError:
                return default
            return parsed if parsed > 0 else default
    return default


def manual_context_override(preset: dict[str, Any]) -> int | None:
    flags = preset.get("flags", {})
    cfg = flags.get("-c")
    if not isinstance(cfg, dict):
        return None
    value = str(cfg.get("value", "") or "").strip()
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _first_int_from_keys(data: Any, keys: tuple[str, ...]) -> int | None:
    if isinstance(data, dict):
        for key in keys:
            value = data.get(key)
            if isinstance(value, int) and value > 0:
                return value
            if isinstance(value, str):
                try:
                    parsed = int(value)
                except ValueError:
                    pass
                else:
                    if parsed > 0:
                        return parsed
        for value in data.values():
            found = _first_int_from_keys(value, keys)
            if found:
                return found
    elif isinstance(data, list):
        for value in data:
            found = _first_int_from_keys(value, keys)
            if found:
                return found
    return None


def detect_runtime_context(preset: dict[str, Any], *, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    manual = manual_context_override(preset)
    result: dict[str, Any] = {
        "advertised_context": None,
        "effective_context": None,
        "current_active_context": None,
        "source": "unknown",
        "confidence": "unknown",
        "details": "",
    }
    if manual:
        result.update({
            "effective_context": manual,
            "source": "manual",
            "confidence": "manual",
            "details": "-c flag/manual context override",
        })
        return result

    host, port = preset_endpoint(preset)
    base = endpoint_url(host, port)

    slots = http_json("GET", f"{base}/slots", timeout=timeout)
    if slots.ok:
        effective = _first_int_from_keys(slots.data, ("n_ctx", "n_ctx_slot", "ctx_size", "context_size"))
        active = _first_int_from_keys(slots.data, ("n_tokens", "n_prompt_tokens", "n_past", "tokens"))
        if effective:
            result.update({
                "effective_context": effective,
                "current_active_context": active,
                "source": "llama.cpp /slots",
                "confidence": "confirmed",
                "details": "/slots runtime context",
            })
            return result

    props = http_json("GET", f"{base}/props", timeout=timeout)
    if props.ok:
        advertised = _first_int_from_keys(props.data, ("n_ctx_train", "context_length", "max_context_length"))
        effective = _first_int_from_keys(props.data, ("n_ctx", "ctx_size", "context_size"))
        if effective or advertised:
            result.update({
                "advertised_context": advertised,
                "effective_context": effective or advertised,
                "source": "llama.cpp /props",
                "confidence": "confirmed" if effective else "metadata",
                "details": "/props context metadata",
            })
            return result

    ps = http_json("GET", f"{base}/api/ps", timeout=timeout)
    if ps.ok:
        effective = _first_int_from_keys(ps.data, ("context_length", "num_ctx", "n_ctx"))
        if effective:
            result.update({
                "effective_context": effective,
                "source": "Ollama /api/ps",
                "confidence": "confirmed",
                "details": "loaded model runtime context",
            })
            return result

    model_name = preset.get("preset_name") or Path(str(preset.get("model_path", "model"))).stem or "model"
    show = http_json("POST", f"{base}/api/show", {"model": model_name}, timeout=timeout)
    if show.ok:
        advertised = _first_int_from_keys(show.data, ("context_length", "num_ctx", "n_ctx"))
        if advertised:
            result.update({
                "advertised_context": advertised,
                "effective_context": advertised,
                "source": "Ollama /api/show",
                "confidence": "metadata",
                "details": "model metadata only",
            })
            return result

    result["details"] = "no runtime context endpoint available"
    return result


def count_prompt_tokens(preset: dict[str, Any], text: str, *, timeout: float = DEFAULT_TIMEOUT) -> tuple[int, str]:
    host, port = preset_endpoint(preset)
    result = http_json("POST", endpoint_url(host, port, "/tokenize"), {"content": text}, timeout=timeout)
    if result.ok:
        tokens = result.data.get("tokens") if isinstance(result.data, dict) else None
        if isinstance(tokens, list):
            return len(tokens), "exact"
        count = result.data.get("count") if isinstance(result.data, dict) else None
        if isinstance(count, int):
            return count, "exact"
    return max(1, len(text.split())), "estimated"


def endpoint_url(host: str, port: int, path: str = "") -> str:
    clean_path = path if path.startswith("/") else f"/{path}" if path else ""
    return f"http://{host}:{port}{clean_path}"


def executable_exists(executable_text: str) -> bool:
    try:
        command = shlex.split(executable_text)
    except ValueError:
        return False
    if not command:
        return False
    executable = command[0]
    if any(sep in executable for sep in ("/", "\\")):
        return Path(executable).expanduser().exists()
    return bool(shutil.which(executable))


def port_is_listening(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> EndpointResult:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers, method=method.upper())
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            elapsed_ms = (time.perf_counter() - started) * 1000
            try:
                data = json.loads(raw) if raw.strip() else {}
            except json.JSONDecodeError as exc:
                return EndpointResult(False, response.status, url, error=f"malformed JSON: {exc}", elapsed_ms=elapsed_ms)
            return EndpointResult(200 <= response.status < 300, response.status, url, data=data, elapsed_ms=elapsed_ms)
    except urllib.error.HTTPError as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        exc.close()
        return EndpointResult(False, exc.code, url, error=detail or exc.reason, elapsed_ms=elapsed_ms)
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        return EndpointResult(False, None, url, error=str(exc), elapsed_ms=elapsed_ms)


def chat_payload(prompt: str, *, max_tokens: int = 32, temperature: float = 0.0) -> dict[str, Any]:
    return {
        "model": "local-model",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }


def build_context_stress_prompt(target_tokens: int) -> str:
    target = max(128, int(target_tokens))
    header_words = [
        "Context", "stress", "payload.", "Read", "the", "entire", "payload",
        "and", "reply", "with", "OK.", "Ignore", "the", "repeated", "fillers.",
    ]
    filler_count = max(0, target - len(header_words))
    return " ".join([*header_words, *(["the"] * filler_count)])


def context_exceeded_details(result: EndpointResult) -> tuple[int, int] | None:
    try:
        payload = json.loads(result.error)
    except (TypeError, json.JSONDecodeError):
        return None
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        return None
    if error.get("type") != "exceed_context_size_error":
        return None
    try:
        n_prompt_tokens = int(error["n_prompt_tokens"])
        n_ctx = int(error["n_ctx"])
    except (KeyError, TypeError, ValueError):
        return None
    if n_prompt_tokens <= 0 or n_ctx <= 0:
        return None
    return n_prompt_tokens, n_ctx


def classify_stress_error(result: EndpointResult) -> str | None:
    if result.ok:
        return None
    if context_exceeded_details(result):
        return "context overflow"
    error = (result.error or "").lower()
    if result.status is not None:
        if "timeout" in error:
            return "timeout"
        if "out of memory" in error or "oom" in error or "cuda error" in error:
            return "OOM indication"
        return "HTTP error"
    if "reset" in error:
        return "connection reset"
    if "timed out" in error or "timeout" in error:
        return "timeout"
    return "unknown"


def stress_completion_budget(context: int, percent: float = STRESS_OUTPUT_RESERVE_PERCENT) -> int:
    return max(32, min(STRESS_MAX_OUTPUT_TOKENS, int(context * percent / 100)))


def stress_stage_prompt_tokens(context: int, stage_percent: int, completion_budget: int) -> int:
    target = int(context * stage_percent / 100)
    return max(128, min(target, max(128, context - completion_budget - 32)))


def synthetic_agent_turn(rng: random.Random, turn: int, target_words: int) -> str:
    sections = [
        f"assistant action summary: turn {turn} inspected synthetic files and planned a focused patch.",
        "synthetic file tree:\napp/\n  src/main.ts\n  src/components/Editor.tsx\n  src/lib/runner.py\n  tests/test_runner.py\n  package.json\n  tsconfig.json",
        "rg results:\nsrc/main.ts:42: TODO wire context budget\nsrc/lib/runner.py:88: raise RuntimeError('synthetic failure')",
        "source excerpt:\nfunction runTask(input: string) {\n  const plan = parseInput(input);\n  return executePlan(plan);\n}",
        "git diff:\n- const maxTokens = 512\n+ const maxTokens = contextBudget.output",
        "pytest failure:\nE AssertionError: expected retry_count == 1\nE traceback: tests/test_runner.py:54",
        "TypeScript error:\nTS2322: Type 'string | undefined' is not assignable to type 'string'.",
        "JSON output:\n{\"status\":\"needs_review\",\"changed_files\":3,\"warnings\":[\"synthetic\"]}",
        "long log:\nINFO loading synthetic module\nWARN cache miss\nINFO retrying request\nINFO completed stage",
        "configuration file:\n[stress]\nmode = sustained-agent\nseed = 1337\n",
    ]
    words: list[str] = []
    while len(words) < target_words:
        block = rng.choice(sections)
        words.extend(block.split())
    return " ".join(words[:target_words])


def run_stress_request(
    preset: dict[str, Any],
    url: str,
    prompt: str,
    completion_budget: int,
    *,
    timeout: float,
    stage_percent: int | None,
    turn: int | None,
    effective_context: int,
    detection: dict[str, Any],
    retry_count: int = 0,
) -> dict[str, Any]:
    result = http_json("POST", url, chat_payload(prompt, max_tokens=completion_budget), timeout=timeout)
    prompt_tokens, prompt_count_source = count_prompt_tokens(preset, prompt, timeout=min(timeout, DEFAULT_TIMEOUT))
    completion_tokens = None
    parse_error = ""
    if result.ok:
        try:
            _content, completion_tokens = parse_chat_response(result.data)
        except ValueError as exc:
            parse_error = str(exc)
            completion_tokens = None
    usage = result.data.get("usage") if result.ok and isinstance(result.data, dict) and isinstance(result.data.get("usage"), dict) else {}
    if isinstance(usage.get("prompt_tokens"), int):
        prompt_tokens = usage["prompt_tokens"]
        prompt_count_source = "usage"
    if isinstance(usage.get("completion_tokens"), int):
        completion_tokens = usage["completion_tokens"]
    active = prompt_tokens + (completion_tokens or 0)
    latency = result.elapsed_ms / 1000 if result.elapsed_ms else None
    generation_tps = (completion_tokens / latency) if completion_tokens and latency and latency > 0 else None
    return {
        "stage_percentage": stage_percent,
        "turn_number": turn,
        "effective_context_size": effective_context,
        "detection_source": detection.get("source"),
        "detection_confidence": detection.get("confidence"),
        "prompt_tokens": prompt_tokens,
        "prompt_token_source": prompt_count_source,
        "completion_tokens": completion_tokens,
        "requested_completion_budget": completion_budget,
        "estimated_active_context_tokens": active,
        "estimated_active_context_percentage": round((active / effective_context) * 100, 2) if effective_context else None,
        "TTFT": None,
        "prefill_tokens_per_sec": None,
        "generation_tokens_per_sec": round(generation_tps, 2) if generation_tps else None,
        "total_latency": round(result.elapsed_ms, 2),
        "success": result.ok and not parse_error,
        "HTTP_status": result.status,
        "error_category": "unexpected generation stop" if parse_error else classify_stress_error(result),
        "error_message": parse_error or result.error,
        "truncated": False,
        "compacted": False,
        "retry_count": retry_count,
        "RAM_used_available": None,
        "GPU_VRAM_used_total": None,
        "GPU_utilization": None,
    }


def format_stress_record(record: dict[str, Any], label: str) -> str:
    status = "PASS" if record["success"] else "FAIL"
    location = f"stage {record['stage_percentage']}%" if record.get("stage_percentage") is not None else f"turn {record.get('turn_number')}"
    active = record.get("estimated_active_context_percentage")
    active_text = f"{active}%" if active is not None else "unknown"
    tps = record.get("generation_tokens_per_sec")
    tps_text = f", {tps} tok/s" if tps is not None else ""
    error = f", {record['error_category']}" if record.get("error_category") else ""
    return f"[{status}] {label} {location}: active {active_text}, prompt {record.get('prompt_tokens')} {record.get('prompt_token_source')}, completion {record.get('completion_tokens')}{tps_text}{error}"


def summarize_stress(records: list[dict[str, Any]], effective_context: int) -> list[str]:
    clean = [r for r in records if r.get("success") and not r.get("error_category")]
    failed = [r for r in records if not r.get("success")]
    degraded = [
        r for r in records
        if r.get("success") and r.get("estimated_active_context_percentage") is not None and r["estimated_active_context_percentage"] >= 90
    ]
    highest_clean = max((r.get("stage_percentage") or r.get("estimated_active_context_percentage") or 0 for r in clean), default=0)
    first_degraded = min((r.get("stage_percentage") or r.get("estimated_active_context_percentage") or 0 for r in degraded), default=None)
    first_failure = min((r.get("stage_percentage") or r.get("estimated_active_context_percentage") or 0 for r in failed), default=None)
    practical = highest_clean
    if first_degraded:
        practical = min(practical, max(10, first_degraded - 10))
    if first_failure:
        practical = min(practical, max(10, first_failure - 10))
    return [
        "=== Stress Summary ===",
        f"Effective context: {effective_context}",
        f"Highest clean sustained stage: {highest_clean}%",
        f"First degradation stage: {first_degraded if first_degraded is not None else 'none'}",
        f"First hard failure stage: {first_failure if first_failure is not None else 'none'}",
        f"Recommended practical working limit: {practical}%",
    ]


def context_stress_report(
    preset: dict[str, Any],
    *,
    timeout: float = 120.0,
    fraction: float = STRESS_CONTEXT_FRACTION,
    max_prompt_tokens: int = STRESS_MAX_PROMPT_TOKENS,
    max_output_tokens: int = STRESS_MAX_OUTPUT_TOKENS,
    stage_percents: tuple[int, ...] = STRESS_STAGE_PERCENTS,
    boundary_percents: tuple[int, ...] = STRESS_BOUNDARY_PERCENTS,
    agent_turns: int = STRESS_AGENT_TURNS,
    random_seed: int = STRESS_AGENT_SEED,
) -> tuple[bool, list[str]]:
    detection = detect_runtime_context(preset)
    context = detection.get("effective_context")
    if not context:
        return False, [
            f"Stress: {preset.get('preset_name', '(unsaved)')}",
            "[FAIL] effective runtime context unavailable",
            f"Detection source: {detection.get('source')} confidence: {detection.get('confidence')}",
            "Stress result: FAIL",
        ]
    context = int(context)
    output_tokens = max(32, min(max_output_tokens, max(32, context // 5)))
    host, port = preset_endpoint(preset)
    lines = [
        f"Stress: {preset.get('preset_name', '(unsaved)')}",
        f"Effective runtime context: {context} ({detection.get('confidence')} via {detection.get('source')})",
        f"Advertised model context: {detection.get('advertised_context')}",
        f"Current active context: {detection.get('current_active_context')}",
    ]
    url = endpoint_url(host, port, "/v1/chat/completions")
    records: list[dict[str, Any]] = []

    lines.append("Mode: Fill and decode")
    for percent in stage_percents:
        completion_budget = min(output_tokens, max(32, int(context * STRESS_OUTPUT_RESERVE_PERCENT / 100)))
        target = stress_stage_prompt_tokens(context, percent, completion_budget)
        prompt = build_context_stress_prompt(min(target, max_prompt_tokens))
        record = run_stress_request(
            preset, url, prompt, completion_budget,
            timeout=timeout, stage_percent=percent, turn=None,
            effective_context=context, detection=detection,
        )
        records.append(record)
        lines.append(format_stress_record(record, "fill/decode"))

    lines.append("Mode: Sustained agent")
    rng = random.Random(random_seed)
    history_parts: list[str] = []
    active_words = 0
    for turn in range(1, agent_turns + 1):
        add_percent = rng.uniform(1.5, 6.0)
        output_percent = rng.uniform(0.3, 1.5)
        if rng.random() < 0.12:
            add_percent *= 1.8
        add_words = max(64, int(context * add_percent / 100))
        completion_budget = max(32, min(max_output_tokens, int(context * output_percent / 100)))
        history_parts.append(synthetic_agent_turn(rng, turn, add_words))
        active_words += add_words
        compacted = False
        max_prompt_words = max(128, min(max_prompt_tokens, context - completion_budget - 32))
        while active_words > max_prompt_words and history_parts:
            removed = history_parts.pop(0)
            active_words -= len(removed.split())
            compacted = True
        prompt = "\n\n".join(history_parts)
        record = run_stress_request(
            preset, url, prompt, completion_budget,
            timeout=timeout, stage_percent=None, turn=turn,
            effective_context=context, detection=detection,
        )
        record["compacted"] = compacted
        records.append(record)
        lines.append(format_stress_record(record, "sustained-agent"))

    lines.append("Mode: Boundary probe")
    for percent in boundary_percents:
        completion_budget = min(output_tokens, max(32, int(context * STRESS_OUTPUT_RESERVE_PERCENT / 100)))
        target = stress_stage_prompt_tokens(context, percent, completion_budget)
        prompt = build_context_stress_prompt(min(target, max_prompt_tokens))
        record = run_stress_request(
            preset, url, prompt, completion_budget,
            timeout=timeout, stage_percent=percent, turn=None,
            effective_context=context, detection=detection,
        )
        records.append(record)
        lines.append(format_stress_record(record, "boundary"))

    lines.extend(summarize_stress(records, context))
    ok = any(record.get("success") for record in records)
    lines.append(f"Stress result: {'PASS' if ok else 'FAIL'}")
    return ok, lines


def parse_chat_response(data: Any) -> tuple[str, int | None]:
    if not isinstance(data, dict):
        raise ValueError("response JSON is not an object")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("response has no choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("first choice is not an object")
    message = first.get("message")
    content = ""
    if isinstance(message, dict):
        content = str(message.get("content") or "")
    elif "text" in first:
        content = str(first.get("text") or "")
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    completion_tokens = usage.get("completion_tokens")
    if isinstance(completion_tokens, int):
        return content, completion_tokens
    if content:
        return content, max(1, len(content.split()))
    return content, None


def probe_endpoint(preset: dict[str, Any], *, timeout: float = DEFAULT_TIMEOUT) -> EndpointResult:
    host, port = preset_endpoint(preset)
    return http_json("POST", endpoint_url(host, port, "/v1/chat/completions"), chat_payload("Say OK.", max_tokens=8), timeout)


def doctor_report(preset: dict[str, Any], *, timeout: float = DEFAULT_TIMEOUT) -> tuple[bool, list[str]]:
    lines: list[str] = []
    ok = True
    name = preset.get("preset_name", "(unsaved)")
    lines.append(f"Doctor: {name}")

    executable = preset.get("inferer_executable", "llama-server")
    exe_ok = executable_exists(executable)
    lines.append(f"[{'PASS' if exe_ok else 'FAIL'}] executable: {executable}")
    ok = ok and exe_ok

    for label, key in (("model", "model_path"), ("mmproj", "mmproj_path"), ("draft model", "draft_model_path")):
        value = (preset.get(key) or "").strip()
        if not value:
            if key == "model_path":
                lines.append(f"[FAIL] {label}: not set")
                ok = False
            else:
                lines.append(f"[SKIP] {label}: not set")
            continue
        exists = Path(value).expanduser().exists()
        lines.append(f"[{'PASS' if exists else 'FAIL'}] {label}: {value}")
        ok = ok and exists

    host, port = preset_endpoint(preset)
    listening = port_is_listening(host, port)
    lines.append(f"[{'PASS' if listening else 'WARN'}] port: {host}:{port} is {'listening' if listening else 'not accepting connections'}")

    for path in ("/health", "/v1/models"):
        result = http_json("GET", endpoint_url(host, port, path), timeout=timeout)
        lines.append(format_endpoint_line(path, result))
        ok = ok and result.ok

    probe = probe_endpoint(preset, timeout=timeout)
    lines.append(format_endpoint_line("/v1/chat/completions", probe))
    if probe.ok:
        try:
            content, tokens = parse_chat_response(probe.data)
            detail = f"{tokens} token estimate" if tokens is not None else "no token count"
            lines.append(f"[PASS] chat response parsed: {detail}; {content[:80]!r}")
        except ValueError as exc:
            lines.append(f"[FAIL] chat response parsed: {exc}")
            ok = False
    ok = ok and probe.ok
    lines.append(f"Doctor result: {'PASS' if ok else 'FAIL'}")
    return ok, lines


def format_endpoint_line(label: str, result: EndpointResult) -> str:
    if result.ok:
        return f"[PASS] {label}: HTTP {result.status} in {result.elapsed_ms:.0f} ms"
    status = f"HTTP {result.status}" if result.status is not None else "connection failed"
    detail = f" ({result.error})" if result.error else ""
    return f"[FAIL] {label}: {status}{detail}"


def probe_report(preset: dict[str, Any], *, timeout: float = DEFAULT_TIMEOUT) -> tuple[bool, list[str]]:
    result = probe_endpoint(preset, timeout=timeout)
    lines = [f"Probe: {preset.get('preset_name', '(unsaved)')}", format_endpoint_line("/v1/chat/completions", result)]
    if not result.ok:
        lines.append("Probe result: FAIL")
        return False, lines
    try:
        content, tokens = parse_chat_response(result.data)
    except ValueError as exc:
        lines.append(f"[FAIL] response parsed: {exc}")
        lines.append("Probe result: FAIL")
        return False, lines
    token_text = f"{tokens} token estimate" if tokens is not None else "no token count"
    lines.append(f"[PASS] response parsed: {token_text}; {content[:120]!r}")
    lines.append("Probe result: PASS")
    return True, lines


def run_benchmark(
    preset: dict[str, Any],
    *,
    timeout: float = DEFAULT_TIMEOUT,
    out_dir: Path | None = None,
    csv_out: bool = False,
) -> tuple[dict[str, Any], list[Path], list[str]]:
    host, port = preset_endpoint(preset)
    result = http_json(
        "POST",
        endpoint_url(host, port, "/v1/chat/completions"),
        chat_payload(BENCH_PROMPT, max_tokens=32),
        timeout=timeout,
    )
    row: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "preset": preset.get("preset_name", "(unsaved)"),
        "host": host,
        "port": port,
        "status": "pass" if result.ok else "fail",
        "http_status": result.status,
        "latency_ms": round(result.elapsed_ms, 2),
        "completion_tokens": None,
        "tokens_per_second": None,
        "error": result.error,
    }
    lines = [f"Bench: {row['preset']}", format_endpoint_line("/v1/chat/completions", result)]
    if result.ok:
        try:
            content, tokens = parse_chat_response(result.data)
            row["completion_tokens"] = tokens
            if tokens is not None and result.elapsed_ms > 0:
                row["tokens_per_second"] = round(tokens / (result.elapsed_ms / 1000), 2)
            row["response_preview"] = content[:200]
            lines.append(format_benchmark_result(row))
        except ValueError as exc:
            row["status"] = "fail"
            row["error"] = str(exc)
            lines.append(f"[FAIL] response parsed: {exc}")
    else:
        lines.append("Benchmark request failed.")

    saved_paths = save_benchmark_result(row, out_dir=out_dir, csv_out=csv_out)
    lines.extend(f"saved: {path}" for path in saved_paths)
    return row, saved_paths, lines


def format_benchmark_result(row: dict[str, Any]) -> str:
    tokens = row.get("completion_tokens")
    speed = row.get("tokens_per_second")
    token_text = f"{tokens} completion tokens" if tokens is not None else "no token count"
    speed_text = f", {speed} tok/s" if speed is not None else ""
    return f"[PASS] benchmark: {row.get('latency_ms')} ms, {token_text}{speed_text}"


def save_benchmark_result(row: dict[str, Any], *, out_dir: Path | None = None, csv_out: bool = False) -> list[Path]:
    base = out_dir or (APP_DIR / LOCAL_DATA_DIR / BENCHMARK_DIR)
    base.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in str(row.get("preset", "preset"))).strip("-") or "preset"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = base / f"{stamp}-{safe_name}.json"
    json_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    paths = [json_path]
    if csv_out:
        csv_path = json_path.with_suffix(".csv")
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        paths.append(csv_path)
    return paths


PATH_KEYS = ("model_path", "mmproj_path", "draft_model_path")


def machine_path_warnings(preset: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    name = preset.get("preset_name", "(unnamed)")
    executable = str(preset.get("inferer_executable", "") or "")
    if executable and any(sep in shlex.split(executable)[0] for sep in ("/", "\\")):
        warnings.append(f"{name}: inferer executable is machine-specific: {executable}")
    for key in PATH_KEYS:
        value = str(preset.get(key, "") or "")
        if value and Path(value).expanduser().is_absolute():
            warnings.append(f"{name}: {key} is an absolute path: {value}")
    return warnings


def export_presets(data: dict[str, Any], out_file: Path, *, names: list[str] | None = None, portable: bool = False) -> list[str]:
    selected = []
    missing = []
    wanted = set(names or [])
    for preset in data.get("presets", []):
        if not wanted or preset.get("preset_name") in wanted:
            selected.append(preset)
    if wanted:
        found = {preset.get("preset_name") for preset in selected}
        missing = sorted(wanted - found)
    if missing:
        raise ValueError(f"preset not found: {', '.join(missing)}")
    warnings = []
    for preset in selected:
        warnings.extend(machine_path_warnings(preset))
    payload = {
        "format_version": 1,
        "kind": "llama-wrap-presets",
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "portable": portable,
        "warnings": warnings,
        "presets": selected,
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return warnings


def import_presets(data: dict[str, Any], in_file: Path, *, force: bool = False) -> tuple[int, list[str]]:
    payload = json.loads(in_file.read_text(encoding="utf-8"))
    presets = payload.get("presets")
    if not isinstance(presets, list):
        raise ValueError("import file does not contain a presets list")
    skipped: list[str] = []
    imported = 0
    for preset in presets:
        if not isinstance(preset, dict):
            continue
        name = str(preset.get("preset_name", "")).strip()
        if not name:
            continue
        if find_preset(data, name) and not force:
            skipped.append(name)
            continue
        upsert_preset(data, preset)
        imported += 1
    return imported, skipped


def command_hash(command: list[str]) -> str:
    payload = json.dumps(command, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def model_signature(path_text: str) -> dict[str, Any]:
    path = Path(path_text).expanduser()
    signature: dict[str, Any] = {"path": str(path)}
    try:
        stat = path.stat()
    except OSError:
        signature.update({"exists": False, "size": None, "mtime_ns": None})
    else:
        signature.update({"exists": True, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return signature


def calibration_matches(calibration: dict[str, Any] | None, command: list[str], model_path: str) -> bool:
    if not isinstance(calibration, dict):
        return False
    if calibration.get("command_hash") != command_hash(command):
        return False
    current = model_signature(model_path)
    saved = calibration.get("model") if isinstance(calibration.get("model"), dict) else {}
    return (
        saved.get("path") == current.get("path")
        and saved.get("size") == current.get("size")
        and saved.get("mtime_ns") == current.get("mtime_ns")
    )


def make_vram_calibration(
    *,
    preset_name: str,
    command: list[str],
    model_path: str,
    estimated_bytes: float,
    observed_bytes: int,
    observed_source: str,
    log_allocations: dict[str, int] | None = None,
) -> dict[str, Any]:
    estimate = max(0.0, float(estimated_bytes))
    observed = max(0, int(observed_bytes))
    return {
        "format_version": 1,
        "preset_name": preset_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command_hash": command_hash(command),
        "model": model_signature(model_path),
        "estimated_bytes": round(estimate),
        "observed_bytes": observed,
        "correction_ratio": round(observed / estimate, 4) if estimate > 0 else None,
        "observed_source": observed_source,
        "log_allocations": dict(log_allocations or {}),
    }


def parse_memory_size_bytes(value: str, unit: str) -> int:
    amount = float(value)
    normalized = unit.lower()
    multipliers = {
        "b": 1,
        "kb": 1000,
        "mb": 1000**2,
        "gb": 1000**3,
        "tb": 1000**4,
        "kib": 1024,
        "mib": 1024**2,
        "gib": 1024**3,
        "tib": 1024**4,
    }
    return int(amount * multipliers.get(normalized, 1))


def parse_llama_vram_log_line(line: str) -> tuple[str, int] | None:
    lower = line.lower()
    if "buffer size" not in lower and "allocation" not in lower:
        return None
    if any(token in lower for token in ("host", "cpu", "mapped")):
        return None
    backend_tokens = ("cuda", "rocm", "hip", "vulkan", "metal", "gpu")
    if not any(token in lower for token in backend_tokens):
        return None
    match = re.search(SIZE_RE, line, re.IGNORECASE)
    if not match:
        return None
    if "model buffer" in lower:
        category = "model"
    elif "kv buffer" in lower:
        category = "kv"
    elif "compute buffer" in lower:
        category = "compute"
    elif "alloc" in lower:
        category = "allocation"
    else:
        category = "other"
    return category, parse_memory_size_bytes(match.group(1), match.group(2))
