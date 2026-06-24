from __future__ import annotations

import json
import os
import queue
import re
import runpy
import shlex
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import urllib.request as urllib_req
import webbrowser

import llamawrap_core as core

try:
    import tkinter as tk
    from tkinter import messagebox, simpledialog, ttk
    TK_IMPORT_ERROR: Exception | None = None
except Exception as exc:
    tk = None
    messagebox = None
    simpledialog = None
    ttk = None
    TK_IMPORT_ERROR = exc
TK_TCL_ERROR = tk.TclError if tk is not None else RuntimeError


APP_TITLE = "llama-wrap"
DEFAULT_SERVER_PORT = 8080
APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
HISTORY_FILE = Path(os.environ["LLAMA_WRAP_HISTORY"]).expanduser() if os.environ.get("LLAMA_WRAP_HISTORY") else APP_DIR / "history.json"
GB = 1024**3

# Known llama-server flags for the "Add flag" suggestion dropdown.
# These are NOT a hardcoded UI — they only appear as autocomplete hints
# when a user clicks + to add a custom flag.
KNOWN_LLAMA_FLAGS: tuple[str, ...] = (
    "--no-webui", "--embedding", "--log-file", "--log-format", "--log-level",
    "--log-colors", "--cont-batching", "--slot-save-file", "--listen",
    "--ssl-file-key", "--ssl-file-cert", "--api-key", "--slots",
    "--endpoint", "--endpoint-file",
    "--chat-template", "--chat-template-kwargs", "--jinja", "--jinja++",
    "--grp-attn-n", "--grp-attn-w", "--rope-scaling", "--rope-freq-base",
    "--rope-freq-scale", "--rope-scaling-type", "--rope-freq-scale-policy",
    "--rope-freq-scale-fill", "--yarn-orig-ctx", "--yarn-ext-factor",
    "--yarn-attn-factor", "--yarn-beta-fast", "--yarn-beta-slow",
    "--no-mmap", "--numa", "--tensor-split", "--main-gpu",
    "--split-mode", "--gpu-device", "--mlock",
    "--samplers", "--sparams", "--temp", "--top-k", "--top-p",
    "--min-p", "--xtc-probability", "--xtc-threshold", "--typical-p",
    "--repeat-last-n", "--repeat-penalty", "--frequency-penalty",
    "--presence-penalty", "--dry-multiplier", "--dry-base",
    "--dry-allowed-length", "--dry-penalty-last-n",
    "--mirostat", "--mirostat-lr", "--mirostat-ent",
    "--dynatemp-range", "--dynatemp-exp",
    "--seed", "--prompt-cache", "--prompt-cache-all",
    "--prompt-cache-ro", "--keep", "--batch-seq-len",
    "--gen-sec", "--n-predict", "--predict",
    "--ignore-eos", "--interactive", "--interactive-first",
    "--in-prefix", "--in-suffix", "--reverse-prompt",
    "--speculative-ngram", "--spec-type", "--spec-draft-n-max",
    "--spec-draft-p-min", "--draft", "--model-draft",
    "--multiline-input", "--simple-io", "--cb-style",
    "--verbose", "--no-display-prompt",
    "--prio", "--prio-pointer-fp16",
    "--offload-kqv", "--no-kqv-offload",
    "--memory-f32", "--memory-float",
    "--ctx-evict", "--cache-reuse",
    "--dont-evict",
    "--spm", "--grammar", "--grammar-file",
    "--ppl-output-token-prob",
    "--check-tensors", "--override-kv",
    "--lora", "--lora-base",
    "--control-vector", "--control-vector-scaled-cn",
)
KV_BYTES = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 1.0625,
    "q5_0": 0.6875,
    "q5_1": 0.75,
    "q4_0": 0.5625,
    "q4_1": 0.625,
    "iq4_nl": 0.5625,
}


@dataclass
class FlagConfig:
    label: str
    value: str = ""
    enabled: bool = True
    value_required: bool = True
    choices: tuple[str, ...] = ()
    inferers: tuple[str, ...] = ()
    custom: bool = False
    step_mode: str = ""
    group: str = ""


@dataclass
class InfererConfig:
    executable: str


@dataclass
class GGUFMetadata:
    path: str
    size: int
    n_layers: int = 0
    n_embd: int = 0
    n_kv_heads: int = 0
    n_embd_k_gqa: int = 0
    n_embd_v_gqa: int = 0
    sliding_window: int = 0
    warnings: list[str] = field(default_factory=list)


class HistoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.presets: list[dict[str, Any]] = []
        self.runs: list[dict[str, Any]] = []
        self.settings: dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.save()
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.presets = list(data.get("presets", []))
            self.runs = list(data.get("runs", []))
            self.settings = dict(data.get("settings", {}))
        except Exception:
            backup = self.path.with_suffix(f".corrupt-{int(time.time())}.json")
            self.path.rename(backup)
            self.presets = []
            self.runs = []
            self.save()

    def save(self) -> None:
        data = {
            "format_version": 1,
            "presets": self.presets,
            "runs": self.runs[-100:],
            "settings": self.settings,
        }
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def set_setting(self, key: str, value: Any) -> None:
        self.settings[key] = value
        self.save()

    def upsert_preset(self, preset: dict[str, Any]) -> None:
        name = preset["preset_name"]
        self.presets = [p for p in self.presets if p.get("preset_name") != name]
        self.presets.append(preset)
        self.presets.sort(key=lambda p: p.get("preset_name", "").lower())
        self.save()

    def delete_preset(self, name: str) -> None:
        self.presets = [p for p in self.presets if p.get("preset_name") != name]
        self.save()



    def add_run(self, model: str, command: list[str]) -> None:
        self.runs.append(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model": model,
                "command": shlex.join(command),
            }
        )
        self.save()


def human_bytes(size: float | int | None) -> str:
    if size is None:
        return "unknown"
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024 or unit == "TiB":
            return f"{value:.0f} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def metadata_key_matches(key: str, suffixes: tuple[str, ...]) -> bool:
    normalized = key.lower()
    return any(normalized == suffix or normalized.endswith(f".{suffix}") for suffix in suffixes)


def read_gguf_value(data: bytes, offset: int, value_type: int) -> tuple[Any, int]:
    if offset < 0 or offset > len(data):
        raise ValueError("metadata outside scanned header")
    scalar_formats = {
        0: "<B",
        1: "<b",
        2: "<H",
        3: "<h",
        4: "<I",
        5: "<i",
        6: "<f",
        7: "<?",
        10: "<Q",
        11: "<q",
        12: "<d",
    }
    if value_type == 8:
        length = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        if length > len(data) - offset:
            raise ValueError("metadata string extends past scanned header")
        value = data[offset : offset + length].decode("utf-8", errors="replace")
        return value, offset + length
    if value_type == 9:
        item_type = struct.unpack_from("<I", data, offset)[0]
        count = struct.unpack_from("<Q", data, offset + 4)[0]
        offset += 12
        values = []
        if count > 1024:
            raise ValueError("large metadata array skipped")
        for _ in range(count):
            value, offset = read_gguf_value(data, offset, item_type)
            values.append(value)
        return values, offset
    fmt = scalar_formats.get(value_type)
    if not fmt:
        raise ValueError("metadata format is newer than this launcher understands")
    if offset + struct.calcsize(fmt) > len(data):
        raise ValueError("metadata value extends past scanned header")
    return struct.unpack_from(fmt, data, offset)[0], offset + struct.calcsize(fmt)


def skip_gguf_value(data: bytes, offset: int, value_type: int) -> int:
    scalar_sizes = {
        0: 1,
        1: 1,
        2: 2,
        3: 2,
        4: 4,
        5: 4,
        6: 4,
        7: 1,
        10: 8,
        11: 8,
        12: 8,
    }
    if value_type == 8:
        if offset + 8 > len(data):
            raise ValueError("metadata string extends past scanned header")
        length = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        if length > len(data) - offset:
            raise ValueError("metadata string extends past scanned header")
        return offset + length
    if value_type == 9:
        if offset + 12 > len(data):
            raise ValueError("metadata array extends past scanned header")
        item_type = struct.unpack_from("<I", data, offset)[0]
        count = struct.unpack_from("<Q", data, offset + 4)[0]
        offset += 12
        for _ in range(count):
            if item_type == 8:
                # String: 8-byte length + string bytes
                s_len = struct.unpack_from("<Q", data, offset)[0]
                offset += 8 + s_len
            elif item_type == 9:
                # Nested array
                offset = skip_gguf_value(data, offset, item_type)
            elif item_type == 0: offset += 1
            elif item_type == 1: offset += 1
            elif item_type == 2: offset += 2
            elif item_type == 3: offset += 2
            elif item_type == 4: offset += 4
            elif item_type == 5: offset += 4
            elif item_type == 6: offset += 4
            elif item_type == 7: offset += 1
            elif item_type == 10: offset += 8
            elif item_type == 11: offset += 8
            elif item_type == 12: offset += 8
            else: offset += 4
        return offset
    size = scalar_sizes.get(value_type)
    if size is None:
        return offset + 8  # Skip a default amount for unknown types
    offset += size
    if offset > len(data):
        raise ValueError("metadata value extends past scanned header")
    return offset


def parse_gguf(path: str) -> GGUFMetadata:
    model = Path(path).expanduser()
    meta = GGUFMetadata(path=str(model), size=model.stat().st_size)
    try:
        with model.open("rb") as fh:
            data = fh.read(min(max(meta.size, 128), 2 * 1024 * 1024))
        if len(data) < 24 or data[:4] != b"GGUF":
            meta.warnings.append("This file does not look like a standard GGUF model. The launcher will skip metadata-based estimates.")
            return meta
        version = struct.unpack_from("<I", data, 4)[0]
        if version < 2:
            meta.warnings.append("This model uses an older GGUF format, so the launcher may estimate memory less accurately.")
        metadata_count = struct.unpack_from("<Q", data, 16)[0]
        offset = 24
        for _ in range(metadata_count):
            if offset + 12 > len(data):
                raise ValueError("metadata extends past scanned header")
            key_len = struct.unpack_from("<Q", data, offset)[0]
            offset += 8
            if key_len > len(data) - offset:
                raise ValueError("metadata key extends past scanned header")
            key = data[offset : offset + key_len].decode("utf-8", errors="replace")
            offset += key_len
            if offset + 4 > len(data):
                raise ValueError("metadata type extends past scanned header")
            value_type = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            is_relevant = metadata_key_matches(
                key,
                (
                    "block_count",
                    "n_layers",
                    "n_layer",
                    "embedding_length",
                    "n_embd",
                    "attention.key_length",
                    "n_embd_head_k",
                    "n_embd_k_gqa",
                    "attention.value_length",
                    "n_embd_head_v",
                    "n_embd_v_gqa",
                    "attention.head_count_kv",
                    "n_head_kv",
                    "n_kv_heads",
                    "attention.sliding_window",
                ),
            )
            if is_relevant:
                value, offset = read_gguf_value(data, offset, value_type)
            else:
                offset = skip_gguf_value(data, offset, value_type)
                continue
            def to_int(v):
                if isinstance(v, list) and v:
                    return int(v[0])
                return int(v)
            if metadata_key_matches(key, ("block_count", "n_layers", "n_layer")):
                meta.n_layers = to_int(value)
            elif metadata_key_matches(key, ("embedding_length", "n_embd")):
                meta.n_embd = to_int(value)
            elif metadata_key_matches(key, ("attention.key_length", "n_embd_head_k", "n_embd_k_gqa")):
                meta.n_embd_k_gqa = to_int(value)
            elif metadata_key_matches(key, ("attention.value_length", "n_embd_head_v", "n_embd_v_gqa")):
                meta.n_embd_v_gqa = to_int(value)
            elif metadata_key_matches(key, ("attention.head_count_kv", "n_head_kv", "n_kv_heads")):
                meta.n_kv_heads = to_int(value)
            elif metadata_key_matches(key, ("attention.sliding_window",)):
                meta.sliding_window = to_int(value)
            
            # Keep looking for sliding_window even if core metrics are found
            core_found = meta.n_layers and meta.n_embd and meta.n_kv_heads and meta.n_embd_k_gqa and meta.n_embd_v_gqa
            if core_found and meta.sliding_window:
                break
            if core_found and metadata_key_matches(key, ("attention.sliding_window",)):
                pass # continue to next to find sliding window
            elif core_found:
                break
    except Exception:
        meta.warnings.append("The launcher could not read all model metadata. Launching should still work, but the VRAM estimate may be less accurate.")
    return meta


class GPUMonitor:
    """Returns (process_used, total_used, total_available, source)."""

    def usage(self, pid: int | None = None) -> tuple[int | None, int | None, int | None, str]:
        nvidia = self._nvidia_usage(pid)
        if nvidia[3] != "not found":
            return nvidia
        amd = self._amd_usage()
        if amd[3] != "not found":
            return amd
        return None, None, None, "GPU not detected"

    def _nvidia_usage(self, pid: int | None = None) -> tuple[int | None, int | None, int | None, str]:
        if not shutil.which("nvidia-smi"):
            return None, None, None, "not found"
        try:
            if pid is not None:
                return self._nvidia_usage_per_process(pid)
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=3,
            )
            line = output.strip().splitlines()[0]
            used_mib, total_mib = [int(part.strip()) for part in line.split(",")[:2]]
            return used_mib * 1024 * 1024, used_mib * 1024 * 1024, total_mib * 1024 * 1024, "NVIDIA"
        except Exception:
            return None, None, None, "NVIDIA"

    def _nvidia_usage_per_process(self, pid: int) -> tuple[int | None, int | None, int | None, str]:
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_gpu_memory,gpu_uuid",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=3,
            )
            proc_mib = 0
            pid_found = False
            for line in output.strip().splitlines():
                parts = line.split(",")
                if len(parts) >= 2:
                    line_pid_str = parts[0].strip()
                    try:
                        line_pid = int(line_pid_str)
                    except ValueError:
                        continue
                    if line_pid == pid:
                        pid_found = True
                        proc_mib += int(parts[1].strip())
            # Get total and total_used from GPU 0
            total_output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=3,
            )
            line = total_output.strip().splitlines()[0]
            total_used_mib, total_mib = [int(part.strip()) for part in line.split(",")[:2]]
            # If PID not found yet (e.g. during model loading), fall back to total GPU usage
            if not pid_found:
                proc_mib = total_used_mib
            return proc_mib * 1024 * 1024, total_used_mib * 1024 * 1024, total_mib * 1024 * 1024, "NVIDIA"
        except Exception:
            return None, None, None, "NVIDIA"

    def _amd_usage(self) -> tuple[int | None, int | None, int | None, str]:
        if not shutil.which("rocm-smi"):
            return None, None, None, "not found"
        try:
            output = subprocess.check_output(["rocm-smi", "--showmeminfo", "vram", "--json"], text=True, stderr=subprocess.DEVNULL, timeout=3)
            data = json.loads(output)
            used = None
            total = None
            for gpu in data.values():
                if not isinstance(gpu, dict):
                    continue
                for key, value in gpu.items():
                    lower = key.lower()
                    amount = int(float(str(value).split()[0]) * 1024 * 1024)
                    if "used memory" in lower:
                        used = amount
                    elif "total memory" in lower:
                        total = amount
            return used, used, total, "AMD"
        except Exception:
            return None, None, None, "AMD"


class Tooltip:
    def __init__(self, widget: tk.Widget, text: str, delay_ms: int = 450) -> None:
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self.after_id: str | None = None
        self.window: tk.Toplevel | None = None
        widget.bind("<Enter>", self.schedule, add="+")
        widget.bind("<Leave>", self.hide, add="+")
        widget.bind("<ButtonPress>", self.hide, add="+")

    def schedule(self, _event: tk.Event | None = None) -> None:
        self.cancel()
        self.after_id = self.widget.after(self.delay_ms, self.show)

    def cancel(self) -> None:
        if self.after_id:
            self.widget.after_cancel(self.after_id)
            self.after_id = None

    def show(self) -> None:
        if self.window or not self.text:
            return
        x = self.widget.winfo_rootx() + 18
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        self.window = tk.Toplevel(self.widget)
        self.window.wm_overrideredirect(True)
        self.window.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            self.window,
            text=self.text,
            justify="left",
            bg="#131721",
            fg="#f0f2f5",
            relief="solid",
            bd=1,
            padx=10,
            pady=8,
            font=("TkDefaultFont", 9),
            wraplength=420,
        )
        label.pack()

    def hide(self, _event: tk.Event | None = None) -> None:
        self.cancel()
        if self.window:
            self.window.destroy()
            self.window = None


class LauncherApp:
    def __init__(self) -> None:
        self.history = HistoryStore(HISTORY_FILE)
        self.gpu = GPUMonitor()
        self.model_meta: GGUFMetadata | None = None
        self.draft_model_meta: GGUFMetadata | None = None
        self.process: subprocess.Popen[str] | None = None
        self.selected_preset: str | None = None
        self.saved_snapshot: str = ""
        self.dirty = False
        self.inferers = self.default_inferers()
        self.flags = self.default_flags()
        self.hidden_flags: set[str] = set()
        self.flag_vars: dict[str, dict[str, tk.Variable]] = {}
        self.log_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.estimate: dict[str, float] = {}
        self.vram_used: int | None = None
        self.vram_total_used: int | None = None
        self.vram_total: int | None = None
        self.vram_source = "GPU not detected"
        self.model_loaded = False
        self.intentional_stop = False
        self._stop_logged = False
        self.extra_args_var: tk.StringVar | None = None
        self.inferer_var: tk.StringVar | None = None
        self.inferer_executable_var: tk.StringVar | None = None
        self.inferer_executable_entry: tk.Entry | None = None
        self.draft_model_var: tk.StringVar | None = None
        self.tokps_var: tk.StringVar | None = None
        self.last_tokens_per_second: float | None = None
        self._session_ttft_ms: list[float] = []
        self._session_gen_tokens: int = 0
        self._session_gen_time_ms: float = 0.0
        self._session_stats_var: tk.StringVar | None = None
        self._auto_restart_count: int = 0
        self.diagnostics_busy = False
        self._vram_command: list[str] = []
        self._vram_command_hash: str = ""
        self._vram_baseline_total_used: int | None = None
        self._vram_log_allocations: dict[str, int] = {}
        self._vram_calibration_saved_for_hash: str = ""
        self._vram_calibration_wait_logged = False
        self.loading_preset = False
        self._server_ready: bool = False
        self._health_check_timer: str | None = None

        self.root = tk.Tk()
        self.root.title(APP_TITLE)
        self.root.geometry("1400x720")
        self.root.minsize(800, 600)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.configure_style()
        self.configure_text_shortcuts()
        self.build_ui()
        self.render_presets()
        self.render_flags()
        self.recalculate_vram()
        self.saved_snapshot = self.snapshot_state()
        self.dirty = False
        self.refresh_process_state()
        self.refresh_gpu_usage()
        self.drain_log_queue()

    def default_inferers(self) -> dict[str, InfererConfig]:
        return {
            "llama.cpp": InfererConfig(
                "llama-server",
            ),
            "Custom": InfererConfig(
                "llama-server",
            ),
        }

    def default_flags(self) -> dict[str, FlagConfig]:
        return {
            # -- Model / VRAM --
            "-ngl": FlagConfig("GPU layers", "0", False, group="Model / VRAM"),
            "--fit": FlagConfig("Fit", "on", False, True, ("on", "off"), group="Model / VRAM"),
            "--fit-margin": FlagConfig("Fit margin MiB", "1024", False, True, (), ("ik_llama.cpp",), group="Model / VRAM"),
            "-cram": FlagConfig("Cache RAM MiB", "8192", False, True, group="Model / VRAM"),
            "-ncmoe": FlagConfig("CPU MoE layers", "", False, group="Model / VRAM"),
            # -- Context / KV Cache --
            "-c": FlagConfig("Context size", "4096", False, True, step_mode="context", group="Context / KV Cache"),
            "-ctk": FlagConfig("KV cache K", "f16", False, True, ("f32", "f16", "bf16", "q8_0", "q5_0", "q5_1", "q4_0", "q4_1", "iq4_nl"), group="Context / KV Cache"),
            "-ctv": FlagConfig("KV cache V", "f16", False, True, ("f32", "f16", "bf16", "q8_0", "q5_0", "q5_1", "q4_0", "q4_1", "iq4_nl"), group="Context / KV Cache"),
            "-khad": FlagConfig("K Hadamard", "", False, False, (), ("ik_llama.cpp",), group="Context / KV Cache"),
            "-vhad": FlagConfig("V Hadamard", "", False, False, (), ("ik_llama.cpp",), group="Context / KV Cache"),
            # -- Threads --
            "-t": FlagConfig("Threads", "-1", False, group="Threads"),
            "-tb": FlagConfig("Batch threads", "-1", False, group="Threads"),
            # -- Memory --
            "-fa": FlagConfig("Flash attention", "auto", False, True, ("auto", "on", "off"), group="Memory"),
            "-mla": FlagConfig("MLA mode", "0", False, True, ("0", "1", "2", "3"), ("ik_llama.cpp",), group="Memory"),
            "-fmoe": FlagConfig("Fused MoE", "", False, False, (), ("ik_llama.cpp",), group="Memory"),
            # -- Server --
            "--port": FlagConfig("Port", str(DEFAULT_SERVER_PORT), False, group="Server"),
            "--host": FlagConfig("Host", "127.0.0.1", False, group="Server"),
            "-np": FlagConfig("Parallel slots", "1", False, group="Server"),
            "-a": FlagConfig("Alias", "", False, group="Server"),
            "-to": FlagConfig("Timeout", "600", False, group="Server"),
            # -- Batch --
            "-b": FlagConfig("Batch", "2048", False, group="Batch"),
            "-ub": FlagConfig("UBatch", "512", False, group="Batch"),
            # -- Speculative Decoding --
            "--spec-type": FlagConfig(
                "Spec type",
                "none",
                False,
                True,
                ("draft-mtp", "draft-simple", "draft-eagle3", "mtp", "none", "ngram-cache", "ngram-simple", "ngram-map-k", "ngram-map-k4v", "ngram-mod"),
                group="Speculative Decoding",
            ),
            "--spec-draft-n-max": FlagConfig("Draft tokens", "16", False, group="Speculative Decoding"),
            "--spec-draft-n-min": FlagConfig("Draft min", "0", False, group="Speculative Decoding"),
            "--spec-draft-p-min": FlagConfig("Draft probability", "0.75", False, group="Speculative Decoding"),
            "-ngld": FlagConfig("Draft GPU layers", "0", False, group="Speculative Decoding"),
            # -- Generation --
            "--reasoning": FlagConfig("Reasoning", "auto", False, True, ("auto", "on", "off"), group="Generation"),
            "--jinja": FlagConfig("Jinja templates", "", False, False, group="Generation"),
            # -- Multimodal --
            "--mmproj": FlagConfig("MMProj", "", False, group="Multimodal"),
        }

    def flag_help(self) -> dict[str, str]:
        return {
            "-ngl": "Max layers to offload to GPU. 'auto' (default) lets llama.cpp pick based on VRAM; 'all' tries full offload; 0 = CPU-only; or a specific layer count.",
            "-c": "Prompt context size in tokens. 0 = use model default. Larger values allow longer conversations but consume more KV-cache memory.",
            "-t": "CPU threads for generation. Set near your physical core count, or leave disabled for llama.cpp to choose.",
            "-tb": "CPU threads for prompt/batch processing. Leave disabled to inherit from generation threads.",
            "-fa": "Flash Attention. 'auto' (default) lets the backend decide; 'on' forces it; 'off' disables it. Disable if your GPU reports flash-attention errors.",
            "-ctk": "KV cache data type for keys. f16 = best quality/compatibility; q8_0/q5/q4 = less memory, possible quality or speed tradeoffs.",
            "-ctv": "KV cache data type for values. Match K for simplicity; quantized types reduce VRAM/RAM usage for long contexts.",
            "--port": "HTTP server port. Clients connect at http://host:port.",
            "--host": "Bind address. 127.0.0.1 = local only; 0.0.0.0 = expose to the network.",
            "-b": "Logical maximum batch size (default: 2048). Affects prompt processing throughput vs. memory usage.",
            "-ub": "Physical maximum batch size (default: 512). Lower if you hit memory errors during prompt processing.",
            "-np": "Number of parallel server slots (default: -1 = auto). Higher supports more concurrent requests but uses more memory.",
            "-a": "Model alias for API clients. Useful when a client expects a specific model name string.",
            "-to": "Server read/write timeout in seconds (default: 600).",
            "--spec-type": "Speculative decoding type(s). Comma-separated: draft-mtp, draft-simple, draft-eagle3, ngram-cache, ngram-simple, ngram-map-k, ngram-map-k4v, ngram-mod. 'none' = disabled.",
            "--spec-draft-n-max": "Max tokens to draft per speculative decoding step (default: 16). MTP models typically use small values (2-6).",
            "--spec-draft-n-min": "Minimum draft tokens before verification (default: 0).",
            "--spec-draft-p-min": "Min probability threshold for greedy draft acceptance (default: 0.75). Higher = stricter acceptance.",
            "-ngld": "GPU layers for the draft/MTP path (long form: --spec-draft-ngl). High value for full draft offload when VRAM allows.",
            "--jinja": "Disable jinja chat template engine. Enabled by default; disable only for legacy clients that need raw model output.",
            "--fit": "Auto-fit layers to VRAM. 'on' (default) adjusts unset args to fit device memory; 'off' disables auto-fit. Set to 'off' when using explicit -ngl.",
            "--fit-margin": "ik_llama.cpp only. VRAM safety margin in MiB for --fit. Increase if model loading hits OOM. (Standard llama.cpp: use --fit-target instead.)",
            "-mla": "ik_llama.cpp only. MLA mode for DeepSeek-style Multi-Latent Attention models. Leave disabled unless the model/docs recommend it.",
            "-fmoe": "ik_llama.cpp only. Enable fused MoE kernels for mixture-of-experts models when supported by your build.",
            "-cram": "Maximum KV/prompt cache size in MiB (default: 8192). 0 = disable, -1 = no limit. Requires unified KV for best results.",
            "-khad": "ik_llama.cpp only. Hadamard transform on K cache for aggressive KV quantization experiments.",
            "-vhad": "ik_llama.cpp only. Hadamard transform on V cache for aggressive KV quantization experiments.",
            "-ncmoe": "Keep MoE expert weights on the CPU for the first N layers. Reduces VRAM usage for MoE models while keeping non-expert layers on GPU.",
            "--reasoning": "Control reasoning/thinking behavior. 'auto' (default) = detect from chat template; 'on' = force enabled; 'off' = disable thought tags.",
        }

    def configure_style(self) -> None:
        self.colors = {
            "bg": "#0a0d14",
            "panel": "#131721",
            "panel_soft": "#1a1f2e",
            "field": "#0d1117",
            "border": "#283040",
            "text": "#f0f2f5",
            "muted": "#8b949e",
            "accent": "#2f81f7",
            "accent_hover": "#1f6feb",
            "add": "#58a6ff",
            "add_hover": "#79c0ff",
            "remove": "#ff7b72",
            "remove_hover": "#ffa198",
            "good": "#238636",
            "good_hover": "#2ea043",
            "danger": "#da3633",
            "danger_hover": "#f85149",
            "warn": "#d29922",
            "import": "#8957e5",
            "import_hover": "#a371f7",
            "conf_green": "#3fb950",
            "conf_yellow": "#d29922",
            "conf_red": "#ff7b72",
        }
        self.root.configure(bg=self.colors["bg"])
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background=self.colors["bg"])
        style.configure("TLabel", background=self.colors["bg"], foreground=self.colors["text"], font=("TkDefaultFont", 10))
        style.configure(
            "Horizontal.TProgressbar",
            troughcolor=self.colors["field"],
            background=self.colors["accent"],
            bordercolor=self.colors["border"],
            lightcolor=self.colors["accent"],
            darkcolor=self.colors["accent"],
        )
        style.configure(
            "Vertical.TScrollbar",
            troughcolor=self.colors["field"],
            background=self.colors["panel_soft"],
            bordercolor=self.colors["border"],
            arrowcolor=self.colors["text"],
            lightcolor=self.colors["panel_soft"],
            darkcolor=self.colors["panel_soft"],
            width=14,
        )
        style.configure(
            "Horizontal.TScrollbar",
            troughcolor=self.colors["field"],
            background=self.colors["panel_soft"],
            bordercolor=self.colors["border"],
            arrowcolor=self.colors["text"],
            lightcolor=self.colors["panel_soft"],
            darkcolor=self.colors["panel_soft"],
            width=14,
        )
        style.map(
            "Vertical.TScrollbar",
            background=[("active", self.colors["border"]), ("pressed", self.colors["muted"])],
        )
        style.map(
            "Horizontal.TScrollbar",
            background=[("active", self.colors["border"]), ("pressed", self.colors["muted"])],
        )
        style.configure(
            "Dark.TCombobox",
            fieldbackground=self.colors["field"],
            background=self.colors["field"],
            foreground=self.colors["text"],
            bordercolor=self.colors["border"],
            arrowcolor=self.colors["text"],
            selectbackground=self.colors["field"],
            selectforeground=self.colors["text"],
            padding=(4, 6),
            font=("TkDefaultFont", 11),
        )

    def configure_text_shortcuts(self) -> None:
        def select_all(event: tk.Event) -> str:
            widget = event.widget
            try:
                if isinstance(widget, tk.Entry):
                    widget.select_range(0, "end")
                    widget.icursor("end")
                elif isinstance(widget, tk.Text):
                    widget.tag_add("sel", "1.0", "end-1c")
                    widget.mark_set("insert", "end-1c")
                return "break"
            except tk.TclError:
                return "break"

        for class_name in ("Entry", "Text"):
            self.root.bind_class(class_name, "<Control-a>", select_all)
            self.root.bind_class(class_name, "<Control-A>", select_all)
            for key, virtual_event in (("c", "<<Copy>>"), ("C", "<<Copy>>"), ("x", "<<Cut>>"), ("X", "<<Cut>>"), ("v", "<<Paste>>"), ("V", "<<Paste>>")):
                self.root.bind_class(class_name, f"<Control-{key}>", lambda event, ve=virtual_event: (event.widget.event_generate(ve), "break")[1])

        self.root.bind("<Control-s>", lambda _event: self.save_current_preset() or "break")

    def build_ui(self) -> None:
        top_bar = tk.Frame(self.root, bg="#080b12", height=54)
        top_bar.pack(fill="x")
        top_bar.pack_propagate(False)
        tk.Label(top_bar, text=APP_TITLE, bg="#080b12", fg=self.colors["text"], font=("TkDefaultFont", 15, "bold")).pack(side="left", padx=18)

        body = tk.Frame(self.root, bg=self.colors["bg"], padx=16, pady=16)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=15, minsize=120)
        body.columnconfigure(1, weight=85)
        body.rowconfigure(0, weight=1)

        left = tk.Frame(body, bg=self.colors["panel"], highlightbackground=self.colors["border"], highlightthickness=1)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        left.rowconfigure(2, weight=1)
        left.columnconfigure(0, weight=1)

        preset_header = tk.Frame(left, bg=self.colors["panel"])
        preset_header.grid(row=0, column=0, sticky="ew", padx=10, pady=(14, 8))
        preset_header.columnconfigure(0, weight=1)
        tk.Label(preset_header, text="Presets", bg=self.colors["panel"], fg=self.colors["text"], font=("TkDefaultFont", 13, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        preset_actions = tk.Frame(left, bg=self.colors["panel"])
        preset_actions.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        for col in range(3):
            preset_actions.columnconfigure(col, weight=1, uniform="preset_actions")
        new_button = self.make_button(preset_actions, "New", self.new_preset, width=38, bg=self.colors["add"], hover=self.colors["add_hover"])
        new_button.grid(row=0, column=0, sticky="ew", padx=(0, 3))
        Tooltip(new_button, "New\n\nClear all fields to start a fresh preset.")
        delete_button = self.make_button(preset_actions, "Delete", self.delete_selected_preset, width=38, bg=self.colors["remove"], hover=self.colors["remove_hover"])
        delete_button.grid(row=0, column=1, sticky="ew", padx=(0, 3))
        Tooltip(delete_button, "Delete\n\nRemove the selected preset.")
        import_button = self.make_button(preset_actions, "Import", self.import_command_dialog, width=38, bg=self.colors["import"], hover=self.colors["import_hover"])
        import_button.grid(row=0, column=2, sticky="ew")
        Tooltip(import_button, "Import\n\nPaste a server command and load recognized arguments into the UI.")
        save_button = self.make_button(preset_actions, "Save", self.save_current_preset, width=38, bg=self.colors["good"], hover=self.colors["good_hover"])
        save_button.grid(row=1, column=0, sticky="ew", padx=(0, 3), pady=(4, 0))
        Tooltip(save_button, "Save (Ctrl+S)\n\nUpdate the selected preset with current settings.")
        save_as_button = self.make_button(preset_actions, "Save as", self.save_as_preset, width=38, bg=self.colors["good"], hover=self.colors["good_hover"])
        save_as_button.grid(row=1, column=1, sticky="ew", padx=(0, 3), pady=(4, 0))
        Tooltip(save_as_button, "Save as\n\nSave current settings as a new preset with a new name.")
        tk.Label(preset_actions, text="𐙼", bg=self.colors["panel"], fg=self.colors["muted"], font=("TkDefaultFont", 12, "bold")).grid(
            row=1, column=2, sticky="nsew", pady=(4, 0)
        )

        preset_list_frame = tk.Frame(left, bg=self.colors["panel"])
        preset_list_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 14))
        preset_list_frame.rowconfigure(0, weight=1)
        preset_list_frame.columnconfigure(0, weight=1)
        preset_xscroll = ttk.Scrollbar(preset_list_frame, orient="horizontal")
        self.preset_list = tk.Listbox(
            preset_list_frame,
            activestyle="none",
            bd=0,
            bg=self.colors["field"],
            fg=self.colors["text"],
            selectbackground=self.colors["accent"],
            selectforeground="#ffffff",
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            highlightcolor=self.colors["border"],
            font=("TkDefaultFont", 11, "bold"),
            relief="flat",
            xscrollcommand=preset_xscroll.set,
        )
        preset_xscroll.configure(command=self.preset_list.xview)
        self.preset_list.grid(row=0, column=0, sticky="nsew")
        preset_xscroll.grid(row=1, column=0, sticky="ew")
        self.preset_tooltip = Tooltip(self.preset_list, "")
        self.preset_list.bind("<<ListboxSelect>>", self.on_preset_selected)
        self.preset_list.bind("<Double-Button-1>", self.load_selected_preset)
        self.preset_list.bind("<Return>", self.load_selected_preset)
        self.preset_list.bind("<Motion>", self.update_preset_tooltip)

        right_shell = tk.Frame(body, bg=self.colors["bg"])
        right_shell.grid(row=0, column=1, sticky="nsew")
        right_shell.columnconfigure(0, weight=1)
        right_shell.rowconfigure(0, weight=1)
        self.right_canvas = tk.Canvas(right_shell, bg=self.colors["bg"], highlightthickness=0)
        self.right_canvas.grid(row=0, column=0, sticky="nsew")
        right_scrollbar = ttk.Scrollbar(right_shell, orient="vertical", command=self.right_canvas.yview)
        right_scrollbar.grid(row=0, column=1, sticky="ns")
        self.right_canvas.configure(yscrollcommand=right_scrollbar.set)
        right = tk.Frame(self.right_canvas, bg=self.colors["bg"])
        self.right_content = right
        self.right_window = self.right_canvas.create_window((0, 0), window=right, anchor="nw")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1, minsize=200)
        right.rowconfigure(4, weight=2, minsize=150)
        right.bind("<Configure>", lambda _event: self.update_right_scrollregion())
        self.right_canvas.bind("<Configure>", self._on_right_canvas_resize)
        self.root.bind_all("<MouseWheel>", self.on_mousewheel, add="+")
        self.root.bind_all("<Button-4>", self.on_mousewheel, add="+")
        self.root.bind_all("<Button-5>", self.on_mousewheel, add="+")

        model_panel = self.make_panel(right, padx=14, pady=14)
        model_panel.grid(row=0, column=0, sticky="ew")
        model_panel.columnconfigure(1, weight=1)
        model_panel.columnconfigure(2, weight=1)
        tk.Label(model_panel, text="Inferer", bg=self.colors["panel"], fg=self.colors["muted"], font=("TkDefaultFont", 10, "bold")).grid(
            row=0, column=0, sticky="w", padx=(0, 12)
        )
        self.inferer_var = tk.StringVar(value="llama.cpp")
        self.inferer_var.trace_add("write", lambda *_: self.on_inferer_changed())
        inferer_selector = self.make_choice_selector(model_panel, self.inferer_var, tuple(self.inferers.keys()))
        inferer_selector.grid(row=0, column=1, sticky="ew")
        Tooltip(inferer_selector, "Inferer\n\nChoose llama.cpp or Custom for a manually entered server command/path.")
        self.inferer_executable_var = tk.StringVar(value=self.current_inferer().executable)
        self.inferer_executable_var.trace_add("write", lambda *_: (self.mark_dirty(), self.update_command_preview()))
        self.inferer_executable_entry = self.make_entry(model_panel, self.inferer_executable_var)
        self.inferer_executable_entry.grid(row=0, column=2, sticky="ew", padx=(8, 0), ipady=6)
        Tooltip(self.inferer_executable_entry, "Executable\n\nCommand or path to the server binary.")
        tk.Label(model_panel, text="Model", bg=self.colors["panel"], fg=self.colors["muted"], font=("TkDefaultFont", 10, "bold")).grid(
            row=1, column=0, sticky="w", padx=(0, 12), pady=(10, 0)
        )
        self.model_var = tk.StringVar()
        self.model_var.trace_add("write", lambda *_: self.on_model_changed())
        self.model_entry = self.make_entry(model_panel, self.model_var)
        self.model_entry.grid(row=1, column=1, columnspan=2, sticky="ew", pady=(10, 0), ipady=6)
        self.make_button(model_panel, "Browse", lambda: self.open_file_picker("model"), width=78).grid(row=1, column=3, padx=(8, 0), pady=(10, 0))
        self.mmproj_label = tk.Label(model_panel, text="MMProj", bg=self.colors["panel"], fg=self.colors["muted"], font=("TkDefaultFont", 10, "bold"))
        self.mmproj_label.grid(row=2, column=0, sticky="w", padx=(0, 12), pady=(10, 0))
        self.mmproj_var = tk.StringVar()
        self.mmproj_var.trace_add("write", lambda *_: self.on_mmproj_changed())
        self.mmproj_entry = self.make_entry(model_panel, self.mmproj_var)
        self.mmproj_entry.grid(row=2, column=1, columnspan=2, sticky="ew", pady=(10, 0), ipady=6)
        self.mmproj_button = self.make_button(model_panel, "Browse", lambda: self.open_file_picker("mmproj"), width=78)
        self.mmproj_button.grid(row=2, column=3, padx=(8, 0), pady=(10, 0))
        self.draft_model_label = tk.Label(model_panel, text="Draft", bg=self.colors["panel"], fg=self.colors["muted"], font=("TkDefaultFont", 10, "bold"))
        self.draft_model_label.grid(row=3, column=0, sticky="w", padx=(0, 12), pady=(10, 0))
        self.draft_model_var = tk.StringVar()
        self.draft_model_var.trace_add("write", lambda *_: self.on_draft_model_changed())
        self.draft_model_entry = self.make_entry(model_panel, self.draft_model_var)
        self.draft_model_entry.grid(row=3, column=1, columnspan=2, sticky="ew", pady=(10, 0), ipady=6)
        self.draft_model_button = self.make_button(model_panel, "Browse", lambda: self.open_file_picker("draft"), width=78)
        self.draft_model_button.grid(row=3, column=3, padx=(8, 0), pady=(10, 0))
        self.draft_compact = tk.Frame(model_panel, bg=self.colors["panel"])
        tk.Label(self.draft_compact, text="Draft", bg=self.colors["panel"], fg=self.colors["muted"], font=("TkDefaultFont", 10, "bold")).grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.draft_compact_button = self.make_button(self.draft_compact, "Browse", lambda: self.open_file_picker("draft"), width=78)
        self.draft_compact_button.grid(row=0, column=1, sticky="w")
        Tooltip(self.draft_model_entry, "Draft model\n\nOptional smaller GGUF model for speculative decoding. It is launched with -md/--model-draft.")
        self.update_optional_model_fields()

        flags_panel = self.make_panel(right, padx=14, pady=12)
        flags_panel.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        flags_panel.columnconfigure(0, weight=1)
        flags_panel.rowconfigure(1, weight=1)
        flags_header = tk.Frame(flags_panel, bg=self.colors["panel"])
        flags_header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        flags_header.columnconfigure(0, weight=1)
        tk.Label(flags_header, text="Flags", bg=self.colors["panel"], fg=self.colors["text"], font=("TkDefaultFont", 12, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        add_flag_button = self.make_button(flags_header, "+", self.add_flag_dialog, width=34, bg=self.colors["add"], hover=self.colors["add_hover"])
        add_flag_button.grid(row=0, column=1, sticky="e", padx=(0, 6))
        Tooltip(add_flag_button, "Add flag\n\nAdd a custom server flag to the UI.")
        clear_flags_button = self.make_button(flags_header, "Clear", self.clear_flags_to_default, width=52, bg=self.colors["panel_soft"])
        clear_flags_button.grid(row=0, column=2, sticky="e")
        Tooltip(clear_flags_button, "Clear flags\n\nUntick every flag, empty all flag values, and clear Extra args.")
        self.flags_frame = tk.Frame(flags_panel, bg=self.colors["panel"])
        self.flags_frame.grid(row=1, column=0, sticky="ew")
        self.flags_frame.bind("<Configure>", self._on_flags_canvas_resize)
        self._flags_min_cols = 1
        self._flags_est_cell_width = 380  # px per flag cell
        self._flags_last_col_count = 0
        self._configure_flag_columns(3)
        extra_row = tk.Frame(flags_panel, bg=self.colors["panel"])
        extra_row.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        extra_row.columnconfigure(1, weight=1)
        tk.Label(extra_row, text="Extra args", bg=self.colors["panel"], fg=self.colors["muted"], font=("TkDefaultFont", 10, "bold")).grid(
            row=0, column=0, sticky="w", padx=(0, 12)
        )
        self.extra_args_var = tk.StringVar()
        self.extra_args_var.trace_add("write", lambda *_: (self.mark_dirty(), self.update_command_preview()))
        extra_entry = self.make_entry(extra_row, self.extra_args_var)
        extra_entry.grid(row=0, column=1, sticky="ew", ipady=6)
        Tooltip(
            extra_entry,
            "Extra server arguments\n\nUse this for advanced or less common flags that are not shown in the UI, for example --no-webui, --metrics, --log-file server.log, -cuda graphs=0, or --tensor-split 3,1.",
        )

        controls = tk.Frame(right, bg=self.colors["bg"])
        controls.grid(row=2, column=0, sticky="ew", pady=(12, 10))
        controls.columnconfigure(0, weight=0)
        controls.columnconfigure(1, weight=1)
        button_bar = tk.Frame(controls, bg=self.colors["bg"])
        button_bar.grid(row=0, column=0, sticky="w", padx=(0, 12))
        for col in range(6):
            button_bar.columnconfigure(col, weight=1, uniform="control_buttons")
        launch_button = self.make_button(button_bar, "Launch", self.launch, width=72, bg=self.colors["good"], hover=self.colors["good_hover"])
        launch_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        Tooltip(launch_button, "Launch\n\nStart the selected inferer with the current settings.")
        stop_button = self.make_button(button_bar, "Stop", self.stop_process, width=72, bg=self.colors["danger"], hover=self.colors["danger_hover"])
        stop_button.grid(row=0, column=1, sticky="ew", padx=(0, 6))
        Tooltip(stop_button, "Stop\n\nStop the running inferer process.")
        clear_button = self.make_button(button_bar, "Clear", self.clear_logs, width=72, bg=self.colors["panel_soft"])
        clear_button.grid(row=0, column=2, sticky="ew", padx=(0, 6))
        Tooltip(clear_button, "Clear output\n\nClear the output log.")
        self.auto_restart_var = tk.BooleanVar(value=False)
        self.auto_restart_button = tk.Button(
            button_bar,
            text="AUTO: OFF",
            command=self.toggle_auto_restart,
            bg=self.colors["panel_soft"],
            fg="#ffffff",
            activebackground=self.colors["accent_hover"],
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            padx=10,
            pady=5,
            width=9,
            font=("TkDefaultFont", 10, "bold"),
            anchor="center",
            justify="center",
            cursor="hand2",
        )
        self.auto_restart_button.bind("<Enter>", lambda _event: self.auto_restart_button.configure(bg=self.colors["accent_hover"] if self.auto_restart_var.get() else self.colors["border"]))
        self.auto_restart_button.bind("<Leave>", lambda _event: self.update_auto_restart_button())
        self.auto_restart_button.grid(row=0, column=3, sticky="ew", padx=(0, 6))
        Tooltip(self.auto_restart_button, "Auto-restart\n\nRestart the selected inferer automatically if it crashes. Manual Stop will not restart it.")
        copy_button = self.make_button(button_bar, "Copy", self.copy_command, width=72, bg=self.colors["panel_soft"])
        copy_button.grid(row=0, column=4, sticky="ew")
        Tooltip(copy_button, "Copy command\n\nCopy the full launch command to the clipboard.")

        # Open UI button — opens the browser to the server's web UI
        self.open_ui_button = self.make_button(button_bar, "Open UI", self.open_browser_ui, width=72, bg=self.colors["import"], hover=self.colors["import_hover"])
        self.open_ui_button.grid(row=0, column=5, sticky="ew", padx=(6, 0))
        Tooltip(self.open_ui_button, "Open UI\n\nOpen the llama.cpp web UI or API docs in your browser.\nOnly works when the server is running.")
        self.command_var = tk.StringVar()
        self.command_label = tk.Label(
            controls,
            textvariable=self.command_var,
            bg=self.colors["bg"],
            fg=self.colors["muted"],
            anchor="nw",
            justify="left",
            wraplength=760,
            font=("DejaVu Sans Mono", 9),
            height=2,
        )
        self.command_label.grid(row=0, column=1, sticky="ew")

        diagnostics_panel = self.make_panel(right, padx=14, pady=12)
        diagnostics_panel.grid(row=3, column=0, sticky="ew", pady=(0, 12))
        diagnostics_panel.columnconfigure(1, weight=1)
        tk.Label(diagnostics_panel, text="Diagnostics", bg=self.colors["panel"], fg=self.colors["text"], font=("TkDefaultFont", 12, "bold")).grid(
            row=0, column=0, sticky="w", padx=(0, 14)
        )
        diagnostics_buttons = tk.Frame(diagnostics_panel, bg=self.colors["panel"])
        diagnostics_buttons.grid(row=0, column=1, sticky="ew")
        for col in range(4):
            diagnostics_buttons.columnconfigure(col, weight=1, uniform="diagnostics_buttons")
        doctor_button = self.make_button(diagnostics_buttons, "Doctor", self.run_diagnostics_doctor, width=78, bg=self.colors["panel_soft"])
        doctor_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        Tooltip(doctor_button, "Doctor\n\nCheck executable, model paths, port, and OpenAI-compatible endpoints.")
        probe_button = self.make_button(diagnostics_buttons, "Probe", self.run_diagnostics_probe, width=78, bg=self.colors["panel_soft"])
        probe_button.grid(row=0, column=1, sticky="ew", padx=(0, 6))
        Tooltip(probe_button, "Probe\n\nSend one small chat completion request to the configured endpoint.")
        bench_button = self.make_button(diagnostics_buttons, "Bench", self.run_diagnostics_bench, width=78, bg=self.colors["panel_soft"])
        bench_button.grid(row=0, column=2, sticky="ew", padx=(0, 6))
        Tooltip(bench_button, "Bench\n\nRun one short benchmark request and save JSON results locally.")
        stress_button = self.make_button(diagnostics_buttons, "Stress", self.run_diagnostics_stress, width=78, bg=self.colors["warn"])
        stress_button.grid(row=0, column=3, sticky="ew")
        Tooltip(stress_button, "Stress\n\nSend a large prompt to fill most of the configured context window.")

        output_panel = self.make_panel(right, padx=14, pady=12)
        output_panel.grid(row=4, column=0, sticky="nsew")
        output_panel.columnconfigure(0, weight=1)
        output_panel.rowconfigure(1, weight=1)
        tk.Label(output_panel, text="Output", bg=self.colors["panel"], fg=self.colors["text"], font=("TkDefaultFont", 12, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        output_wrap = tk.Frame(output_panel, bg=self.colors["panel"])
        output_wrap.grid(row=1, column=0, sticky="nsew")
        output_wrap.columnconfigure(0, weight=1)
        output_wrap.rowconfigure(0, weight=1)
        self.output = tk.Text(
            output_wrap,
            height=14,
            wrap="word",
            bd=0,
            relief="flat",
            bg=self.colors["field"],
            fg=self.colors["text"],
            insertbackground=self.colors["text"],
            padx=10,
            pady=10,
            font=("DejaVu Sans Mono", 10),
            state="disabled",
        )
        self.output.grid(row=0, column=0, sticky="nsew")
        output_scroll = ttk.Scrollbar(output_wrap, command=self.output.yview)
        output_scroll.grid(row=0, column=1, sticky="ns")
        self.output.configure(yscrollcommand=output_scroll.set)
        self.output.tag_configure("normal", foreground="#d1d7e0")
        self.output.tag_configure("warn", foreground="#f2cc60")
        self.output.tag_configure("error", foreground="#ff7b72")

        status_panel = self.make_panel(right, padx=14, pady=12)
        status_panel.grid(row=5, column=0, sticky="ew", pady=(12, 0))
        status_panel.columnconfigure(4, weight=1)
        tk.Label(status_panel, text="Status", bg=self.colors["panel"], fg=self.colors["muted"], font=("TkDefaultFont", 10, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        self.status_var = tk.StringVar(value="Stopped")
        self.status_label = tk.Label(
            status_panel,
            textvariable=self.status_var,
            bg=self.colors["panel_soft"],
            fg=self.colors["text"],
            padx=10,
            pady=6,
            font=("TkDefaultFont", 10, "bold"),
        )
        self.status_label.grid(row=0, column=1, sticky="w", padx=(10, 20))
        self.tokps_var = tk.StringVar(value="")
        tk.Label(status_panel, textvariable=self.tokps_var, bg=self.colors["panel"], fg=self.colors["muted"], font=("TkDefaultFont", 10, "bold")).grid(
            row=0, column=2, sticky="w", padx=(0, 12)
        )
        self._session_stats_var = tk.StringVar(value="")
        tk.Label(status_panel, textvariable=self._session_stats_var, bg=self.colors["panel"], fg=self.colors["muted"], font=("TkDefaultFont", 10, "bold")).grid(
            row=0, column=3, sticky="w", padx=(0, 16)
        )
        tk.Label(status_panel, text="VRAM", bg=self.colors["panel"], fg=self.colors["muted"], font=("TkDefaultFont", 10, "bold")).grid(
            row=0, column=4, sticky="e", padx=(0, 10)
        )
        self.vram_label_var = tk.StringVar()
        tk.Label(status_panel, textvariable=self.vram_label_var, bg=self.colors["panel"], fg=self.colors["text"], font=("TkDefaultFont", 10)).grid(
            row=0, column=5, sticky="w"
        )
        self.vram_confidence_var = tk.StringVar(value="")
        self.vram_confidence_label = tk.Label(
            status_panel, textvariable=self.vram_confidence_var, bg=self.colors["panel"], font=("TkDefaultFont", 12)
        )
        self.vram_confidence_label.grid(row=0, column=6, sticky="w", padx=(4, 0))
        self.vram_bar = tk.Canvas(status_panel, height=36, bg=self.colors["field"], highlightthickness=1, highlightbackground=self.colors["border"])
        self.vram_bar.grid(row=1, column=0, columnspan=7, sticky="ew", pady=(10, 0))
        self.vram_bar.bind("<Configure>", lambda _event: self.draw_vram_bar())
        self.vram_breakdown_var = tk.StringVar()
        tk.Label(
            status_panel,
            textvariable=self.vram_breakdown_var,
            bg=self.colors["panel"],
            fg=self.colors["muted"],
            anchor="w",
            justify="left",
            font=("DejaVu Sans Mono", 9),
        ).grid(row=2, column=0, columnspan=5, sticky="ew", pady=(8, 0))

    def make_panel(self, parent: tk.Widget, padx: int = 12, pady: int = 12) -> tk.Frame:
        return tk.Frame(parent, bg=self.colors["panel"], highlightbackground=self.colors["border"], highlightthickness=1, padx=padx, pady=pady)

    def make_entry(self, parent: tk.Widget, variable: tk.StringVar) -> tk.Entry:
        return tk.Entry(
            parent,
            textvariable=variable,
            bg=self.colors["field"],
            fg=self.colors["text"],
            insertbackground=self.colors["text"],
            relief="flat",
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            highlightcolor=self.colors["accent"],
            font=("TkDefaultFont", 11),
            insertwidth=1,
        )

    def make_choice_selector(self, parent: tk.Widget, variable: tk.StringVar, choices: tuple[str, ...]) -> tk.Frame:
        selector = tk.Frame(
            parent,
            bg=self.colors["field"],
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            highlightcolor=self.colors["accent"],
            cursor="hand2",
            height=34,
        )
        selector.grid_propagate(False)
        selector.columnconfigure(0, weight=1)
        selector.rowconfigure(0, weight=1)
        value_label = tk.Label(
            selector,
            textvariable=variable,
            bg=self.colors["field"],
            fg=self.colors["text"],
            anchor="w",
            padx=8,
            font=("TkDefaultFont", 11),
            cursor="hand2",
        )
        value_label.grid(row=0, column=0, sticky="nsew")
        arrow = tk.Label(selector, text="▾", bg=self.colors["field"], fg=self.colors["text"], padx=8, font=("TkDefaultFont", 10), cursor="hand2")
        arrow.grid(row=0, column=1, sticky="ns")
        menu = tk.Menu(selector, tearoff=False, bg=self.colors["field"], fg=self.colors["text"], activebackground=self.colors["accent"], activeforeground="#ffffff")
        for choice in choices:
            menu.add_command(label=choice, command=lambda value=choice: variable.set(value))

        def show_menu(_event: tk.Event | None = None) -> None:
            menu.tk_popup(selector.winfo_rootx(), selector.winfo_rooty() + selector.winfo_height())

        for widget in (selector, value_label, arrow):
            widget.bind("<Button-1>", show_menu)
            widget.bind("<Enter>", lambda _event, w=widget: w.configure(bg=self.colors["panel_soft"]))
            widget.bind("<Leave>", lambda _event, w=widget: w.configure(bg=self.colors["field"]))
        return selector

    def make_button(
        self,
        parent: tk.Widget,
        text: str,
        command: Any,
        width: int = 80,
        bg: str | None = None,
        hover: str | None = None,
    ) -> tk.Button:
        base = bg or self.colors["accent"]
        hover_color = hover or self.colors["accent_hover"]
        button = tk.Button(
            parent,
            text=text.upper(),
            command=command,
            bg=base,
            fg="#ffffff",
            activebackground=hover_color,
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            padx=10,
            pady=5,
            width=max(1, width // 10),
            font=("TkDefaultFont", 10, "bold"),
            anchor="center",
            justify="center",
            cursor="hand2",
        )
        button.bind("<Enter>", lambda _event: button.configure(bg=hover_color))
        button.bind("<Leave>", lambda _event: button.configure(bg=base))
        return button

    def toggle_auto_restart(self) -> None:
        self.auto_restart_var.set(not self.auto_restart_var.get())
        self.update_auto_restart_button()

    def update_auto_restart_button(self) -> None:
        button = getattr(self, "auto_restart_button", None)
        if not button:
            return
        enabled = self.auto_restart_var.get()
        button.configure(text="AUTO: ON" if enabled else "AUTO: OFF", bg=self.colors["accent"] if enabled else self.colors["panel_soft"])

    def current_inferer_key(self) -> str:
        if self.inferer_var is None:
            return "llama.cpp"
        key = self.inferer_var.get()
        return key if key in self.inferers else "llama.cpp"

    def current_inferer(self) -> InfererConfig:
        return self.inferers[self.current_inferer_key()]

    def flag_supported_by_current_inferer(self, flag: str) -> bool:
        if flag in self.hidden_flags:
            return False
        cfg = self.flags.get(flag)
        if cfg is None or not cfg.inferers:
            return True
        return self.current_inferer_key() in cfg.inferers

    def on_inferer_changed(self) -> None:
        inferer = self.current_inferer()
        if self.inferer_executable_var and not self.inferer_executable_var.get().strip():
            self.inferer_executable_var.set(inferer.executable)
        if self.current_inferer_key() == "Custom" and self.inferer_executable_entry:
            self.root.after_idle(lambda: (self.inferer_executable_entry.focus_set(), self.inferer_executable_entry.select_range(0, "end")))
        self.mark_dirty()
        self.render_flags()
        self.update_command_preview()

    def snapshot_state(self) -> str:
        data = {
            "inferer": self.current_inferer_key(),
            "inferer_executable": self.inferer_executable_var.get().strip() if self.inferer_executable_var else "",
            "model_path": self.model_var.get().strip() if hasattr(self, "model_var") else "",
            "mmproj_path": self.mmproj_var.get().strip() if hasattr(self, "mmproj_var") else "",
            "draft_model_path": self.draft_model_var.get().strip() if self.draft_model_var else "",
            "extra_args": self.extra_args_var.get().strip() if self.extra_args_var else "",
            "hidden_flags": sorted(self.hidden_flags),
            "flags": {
                flag: {
                    "value": cfg.value,
                    "enabled": cfg.enabled,
                    "value_required": cfg.value_required,
                    "custom": cfg.custom,
                    "step_mode": cfg.step_mode,
                }
                for flag, cfg in self.flags.items()
            },
        }
        return json.dumps(data, sort_keys=True)

    def mark_dirty(self) -> None:
        if self.loading_preset:
            return
        was_dirty = self.dirty
        self.dirty = self.snapshot_state() != self.saved_snapshot if self.selected_preset else bool(self.snapshot_state())
        if self.dirty != was_dirty:
            self.render_presets()

    def normalize_custom_flag(self, flag: str) -> str:
        return flag.strip()

    def add_or_update_flag(self, flag: str, value: str = "", value_required: bool = True, enabled: bool = True, step_mode: str = "") -> None:
        normalized = self.normalize_custom_flag(flag)
        if not normalized.startswith("-") or normalized == "-":
            raise ValueError("Flag must start with - or --.")
        if any(char.isspace() for char in normalized):
            raise ValueError("Flag name cannot contain spaces.")
        if normalized in {"-m", "--model"}:
            raise ValueError("Use the Model field for the model path.")
        if normalized in {"--mmproj", "-mm"}:
            raise ValueError("Use the MMProj field for the projector path.")
        if normalized in {"-md", "--model-draft", "--spec-draft-model"}:
            raise ValueError("Use the Draft model field for speculative decoding.")
        existing = self.flags.get(normalized)
        if existing:
            existing.value = value
            existing.value_required = value_required
            existing.enabled = enabled
            existing.step_mode = step_mode
            self.hidden_flags.discard(normalized)
        else:
            label = normalized.lstrip("-") or normalized
            self.flags[normalized] = FlagConfig(label, value, enabled, value_required, custom=True, step_mode=step_mode)
        self.mark_dirty()

    def add_flag_dialog(self) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Add flag")
        dialog.configure(bg=self.colors["bg"])
        dialog.geometry("460x250")
        dialog.minsize(420, 240)
        dialog.transient(self.root)
        dialog.grab_set()

        shell = tk.Frame(dialog, bg=self.colors["bg"], padx=14, pady=14)
        shell.pack(fill="both", expand=True)
        shell.columnconfigure(1, weight=1)

        tk.Label(shell, text="Flag", bg=self.colors["bg"], fg=self.colors["muted"], font=("TkDefaultFont", 10, "bold")).grid(
            row=0, column=0, sticky="w", padx=(0, 12), pady=(0, 10)
        )
        flag_var = tk.StringVar()
        flag_entry = ttk.Combobox(
            shell,
            textvariable=flag_var,
            values=KNOWN_LLAMA_FLAGS,
            style="Dark.TCombobox",
            height=20,
        )
        flag_entry.grid(row=0, column=1, sticky="ew", pady=(0, 10), ipady=6)

        tk.Label(shell, text="Value", bg=self.colors["bg"], fg=self.colors["muted"], font=("TkDefaultFont", 10, "bold")).grid(
            row=1, column=0, sticky="w", padx=(0, 12), pady=(0, 10)
        )
        value_var = tk.StringVar()
        value_entry = self.make_entry(shell, value_var)
        value_entry.grid(row=1, column=1, sticky="ew", pady=(0, 10), ipady=6)

        needs_value_var = tk.BooleanVar(value=True)
        needs_value = tk.Checkbutton(
            shell,
            text="takes a value",
            variable=needs_value_var,
            bg=self.colors["bg"],
            fg=self.colors["text"],
            activebackground=self.colors["bg"],
            activeforeground=self.colors["text"],
            selectcolor=self.colors["field"],
            relief="flat",
            bd=0,
            highlightthickness=0,
        )
        needs_value.grid(row=2, column=1, sticky="w")
        power2_var = tk.BooleanVar(value=False)
        power2 = tk.Checkbutton(
            shell,
            text="2^n controls",
            variable=power2_var,
            bg=self.colors["bg"],
            fg=self.colors["text"],
            activebackground=self.colors["bg"],
            activeforeground=self.colors["text"],
            selectcolor=self.colors["field"],
            relief="flat",
            bd=0,
            highlightthickness=0,
        )
        power2.grid(row=3, column=1, sticky="w", pady=(6, 0))

        actions = tk.Frame(shell, bg=self.colors["bg"])
        actions.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(16, 0))
        actions.columnconfigure(0, weight=1)

        def save() -> None:
            try:
                step_mode = "power2" if needs_value_var.get() and power2_var.get() else ""
                self.add_or_update_flag(flag_var.get(), value_var.get().strip(), needs_value_var.get(), True, step_mode)
            except ValueError as exc:
                messagebox.showerror("Cannot add flag", str(exc), parent=dialog)
                return
            dialog.destroy()
            self.render_flags()
            self.recalculate_vram()

        self.make_button(actions, "Add", save, width=76, bg=self.colors["add"], hover=self.colors["add_hover"]).grid(row=0, column=1, padx=(0, 8))
        self.make_button(actions, "Cancel", dialog.destroy, width=76, bg=self.colors["panel_soft"]).grid(row=0, column=2)
        flag_entry.focus_set()

    def remove_flag(self, flag: str) -> None:
        if flag == "--mmproj":
            return
        cfg = self.flags.get(flag)
        if not cfg:
            return
        if cfg.custom:
            del self.flags[flag]
        else:
            self.hidden_flags.add(flag)
            cfg.enabled = False
        self.mark_dirty()
        self.render_flags()
        self.recalculate_vram()

    def render_flags(self) -> None:
        for child in self.flags_frame.winfo_children():
            child.destroy()
        self.flag_vars.clear()
        cols = self._flags_col_count()
        self._configure_flag_columns(cols)
        self._flags_last_col_count = cols
        items = [(flag, cfg) for flag, cfg in self.flags.items() if flag != "--mmproj" and self.flag_supported_by_current_inferer(flag)]
        help_text = self.flag_help()
        # Sort by group order, then by flag name within group
        group_order = {
            "Model / VRAM": 0,
            "Context / KV Cache": 1,
            "Threads": 2,
            "Memory": 3,
            "Server": 4,
            "Batch": 5,
            "Speculative Decoding": 6,
            "Generation": 7,
            "Multimodal": 8,
        }
        items.sort(key=lambda fc: (group_order.get(fc[1].group, 99), fc[0]))
        for idx, (flag, cfg) in enumerate(items):
            row, col = divmod(idx, cols)
            cell = tk.Frame(self.flags_frame, bg=self.colors["panel"])
            cell.grid(row=row, column=col, sticky="ew", ipadx=7, pady=5)
            cell.columnconfigure(1, weight=1)
            cell.columnconfigure(2, minsize=26)
            enabled_var = tk.BooleanVar(value=cfg.enabled)
            check = tk.Checkbutton(
                cell,
                text=flag,
                variable=enabled_var,
                command=lambda f=flag, v=enabled_var: self.set_flag_enabled(f, v.get()),
                bg=self.colors["panel"],
                fg=self.colors["text"],
                activebackground=self.colors["panel"],
                activeforeground=self.colors["text"],
                selectcolor=self.colors["field"],
                relief="flat",
                bd=0,
                highlightthickness=0,
                font=("DejaVu Sans Mono", 10, "bold"),
            )
            check.grid(
                row=0, column=0, sticky="w", padx=(2, 8)
            )
            Tooltip(check, f"{flag} - {cfg.label}\n\n{help_text.get(flag, '')}")
            values: dict[str, tk.Variable] = {"enabled": enabled_var}
            if cfg.value_required:
                value_var = tk.StringVar(value=cfg.value)
                value_var.trace_add("write", lambda *_args, f=flag, v=value_var: self.set_flag_value(f, v.get()))
                if cfg.choices:
                    entry = self.make_choice_selector(cell, value_var, cfg.choices)
                else:
                    entry_parent = cell
                    if cfg.step_mode:
                        entry_parent = tk.Frame(cell, bg=self.colors["panel"])
                    entry = self.make_entry(entry_parent, value_var)
                    entry.configure(width=18)
                if cfg.step_mode:
                    control_frame = entry_parent
                    control_frame.grid(row=0, column=1, sticky="ew")
                    control_frame.columnconfigure(2, weight=1)
                    minus = self.make_button(control_frame, "-", lambda f=flag: self.step_numeric_flag(f, -1), width=26, bg=self.colors["panel_soft"])
                    minus.grid(row=0, column=0, sticky="ew", padx=(0, 4))
                    divide = self.make_button(control_frame, "/", lambda f=flag: self.scale_numeric_flag(f, 0.5), width=26, bg=self.colors["panel_soft"])
                    divide.grid(row=0, column=1, sticky="ew", padx=(0, 4))
                    entry.grid(row=0, column=2, sticky="ew", ipady=6)
                    multiply = self.make_button(control_frame, "2x", lambda f=flag: self.scale_numeric_flag(f, 2), width=32, bg=self.colors["panel_soft"])
                    multiply.grid(row=0, column=3, sticky="ew", padx=(4, 0))
                    plus = self.make_button(control_frame, "+", lambda f=flag: self.step_numeric_flag(f, 1), width=26, bg=self.colors["panel_soft"])
                    plus.grid(row=0, column=4, sticky="ew", padx=(4, 0))
                    if cfg.step_mode == "power2":
                        Tooltip(minus, "Previous 2^n value\n\nHalve the current value.")
                        Tooltip(plus, "Next 2^n value\n\nDouble the current value.")
                        Tooltip(entry, f"{flag} value\n\n2^n mode keeps this value on powers of two.\n\n{help_text.get(flag, '')}")
                    else:
                        Tooltip(minus, "Decrease value\n\nSubtract 1024.")
                        Tooltip(plus, "Increase value\n\nAdd 1024.")
                        Tooltip(entry, f"{flag} value\n\n-/+ change by 1024; / and x halve or double it.\n\n{help_text.get(flag, '')}")
                    Tooltip(divide, "Halve value\n\nDivide the current value by 2.")
                    Tooltip(multiply, "Double value\n\nMultiply the current value by 2.")
                else:
                    entry.grid(row=0, column=1, sticky="ew", ipady=6)
                    Tooltip(entry, f"{flag} value\n\n{help_text.get(flag, '')}")
                values["value"] = value_var
            else:
                label = tk.Label(
                    cell,
                    text=cfg.label,
                    bg=self.colors["panel"],
                    fg=self.colors["muted"],
                    font=("TkDefaultFont", 10),
                )
                label.grid(row=0, column=1, sticky="w")
                Tooltip(label, f"{flag} - {cfg.label}\n\n{help_text.get(flag, '')}")
            remove = self.make_button(cell, "x", lambda f=flag: self.remove_flag(f), width=22, bg=self.colors["panel_soft"])
            remove.grid(row=0, column=2, sticky="e", padx=(6, 0))
            Tooltip(remove, f"Remove {flag}\n\nHide this flag from the current UI. Add it again with the plus button.")
            self.flag_vars[flag] = values
        # blank row at bottom for visual breathing room
        padding_row = tk.Frame(self.flags_frame, bg=self.colors["panel"], height=12)
        padding_row.grid(row=(idx // cols) + 1, column=0, columnspan=cols, sticky="ew")

    def clear_flags_to_default(self) -> None:
        for cfg in self.flags.values():
            cfg.enabled = False
            cfg.value = ""
        if self.extra_args_var:
            self.extra_args_var.set("")
        self.render_flags()
        self.recalculate_vram()

    def render_presets(self) -> None:
        self.preset_list.delete(0, "end")
        for preset in self.history.presets:
            name = preset.get("preset_name", "Unnamed")
            suffix = " *" if self.selected_preset == name and self.dirty else ""
            self.preset_list.insert("end", f"{name}{suffix}")
        if self.selected_preset:
            for idx, preset in enumerate(self.history.presets):
                if preset.get("preset_name") == self.selected_preset:
                    self.preset_list.selection_set(idx)
                    self.preset_list.activate(idx)
                    break

    def on_preset_selected(self, _event: tk.Event) -> None:
        return

    def update_preset_tooltip(self, event: tk.Event) -> None:
        idx = self.preset_list.nearest(event.y)
        if idx < 0 or idx >= len(self.history.presets):
            self.preset_tooltip.text = ""
            return
        self.preset_tooltip.text = self.history.presets[idx].get("preset_name", "Unnamed")

    def load_selected_preset(self, _event: tk.Event | None = None) -> None:
        selection = self.preset_list.curselection()
        if not selection:
            return
        preset = self.history.presets[selection[0]]
        self.load_preset(preset)

    def new_preset(self) -> None:
        self.selected_preset = None
        self.saved_snapshot = ""
        self.dirty = False
        self.model_var.set("")
        self.mmproj_var.set("")
        if self.draft_model_var:
            self.draft_model_var.set("")
        if self.extra_args_var:
            self.extra_args_var.set("")
        self.flags = self.default_flags()
        self.hidden_flags = set()
        self.model_meta = None
        self.draft_model_meta = None
        self.render_flags()
        self.render_presets()
        self.recalculate_vram()

    def save_current_preset(self) -> None:
        if not self.selected_preset:
            messagebox.showwarning("No preset selected", "Select a preset to save, or use Save as to create a new one.", parent=self.root)
            return
        name = self.selected_preset
        preset = self.current_preset_payload(name)
        self.history.upsert_preset(preset)
        self.saved_snapshot = self.snapshot_state()
        self.dirty = False
        self.render_presets()

    def save_as_preset(self) -> None:
        default = Path(self.model_var.get()).stem if self.model_var.get() else ""
        name = simpledialog.askstring("Save preset as", "Preset name:", initialvalue=default, parent=self.root)
        if not name:
            return
        preset = self.current_preset_payload(name)
        self.history.upsert_preset(preset)
        self.selected_preset = name.strip()
        self.saved_snapshot = self.snapshot_state()
        self.dirty = False
        self.render_presets()

    def current_preset_payload(self, name: str | None = None) -> dict[str, Any]:
        preset_name = (name or self.selected_preset or Path(self.model_var.get()).stem or "Unsaved").strip()
        payload = {
            "format_version": 1,
            "preset_name": preset_name,
            "inferer": self.current_inferer_key(),
            "inferer_executable": self.inferer_executable_var.get().strip() if self.inferer_executable_var else self.current_inferer().executable,
            "model_path": self.model_var.get().strip(),
            "mmproj_path": self.flags["--mmproj"].value,
            "draft_model_path": self.draft_model_var.get().strip() if self.draft_model_var else "",
            "extra_args": self.extra_args_var.get().strip() if self.extra_args_var else "",
            "hidden_flags": sorted(self.hidden_flags),
            "flags": {
                flag: {
                    "value": cfg.value,
                    "enabled": cfg.enabled,
                    "value_required": cfg.value_required,
                    "custom": cfg.custom,
                    "step_mode": cfg.step_mode,
                }
                for flag, cfg in self.flags.items()
            },
        }
        for existing in self.history.presets:
            if existing.get("preset_name") != preset_name:
                continue
            if "session_stats" in existing:
                payload["session_stats"] = existing["session_stats"]
            if "vram_calibration" in existing:
                payload["vram_calibration"] = existing["vram_calibration"]
            break
        return payload

    def import_command_dialog(self) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Import server command")
        dialog.configure(bg=self.colors["bg"])
        dialog.geometry("760x360")
        dialog.minsize(560, 280)
        dialog.transient(self.root)
        dialog.grab_set()

        shell = tk.Frame(dialog, bg=self.colors["bg"], padx=14, pady=14)
        shell.pack(fill="both", expand=True)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(1, weight=1)

        tk.Label(
            shell,
            text="Paste server command",
            bg=self.colors["bg"],
            fg=self.colors["text"],
            font=("TkDefaultFont", 12, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))
        text = tk.Text(
            shell,
            height=8,
            wrap="word",
            bd=0,
            relief="flat",
            bg=self.colors["field"],
            fg=self.colors["text"],
            insertbackground=self.colors["text"],
            padx=10,
            pady=10,
            font=("DejaVu Sans Mono", 10),
        )
        text.grid(row=1, column=0, sticky="nsew")
        actions = tk.Frame(shell, bg=self.colors["bg"])
        actions.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        actions.columnconfigure(0, weight=1)

        def apply_import() -> None:
            command = text.get("1.0", "end").strip()
            try:
                changed, skipped = self.import_command(command)
            except ValueError as exc:
                messagebox.showerror("Import failed", str(exc), parent=dialog)
                return
            dialog.destroy()
            summary = f"Imported {changed} setting{'s' if changed != 1 else ''}."
            if skipped:
                summary += f"\nSkipped: {', '.join(skipped[:8])}"
                if len(skipped) > 8:
                    summary += f", +{len(skipped) - 8} more"
            messagebox.showinfo("Command imported", summary, parent=self.root)

        self.make_button(actions, "Import", apply_import, width=86, bg=self.colors["import"], hover=self.colors["import_hover"]).grid(row=0, column=1, padx=(0, 8))
        self.make_button(actions, "Cancel", dialog.destroy, width=86, bg=self.colors["panel_soft"]).grid(row=0, column=2)
        text.focus_set()

    def import_command(self, command_text: str) -> tuple[int, list[str]]:
        if not command_text.strip():
            raise ValueError("Paste a server command first.")
        try:
            tokens = shlex.split(command_text)
        except ValueError as exc:
            raise ValueError(f"Could not read the command: {exc}") from exc
        if not tokens:
            raise ValueError("Paste a server command first.")
        if tokens[0] and not tokens[0].startswith("-"):
            executable = tokens[0]
            exe_name = Path(executable).name
            if self.inferer_executable_var:
                self.inferer_executable_var.set(executable)
            if self.inferer_var:
                self.inferer_var.set("llama.cpp" if exe_name == "llama-server" else "Custom")
            tokens = tokens[1:]

        alias_to_flag = {
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
        valueless_flags = {flag for flag, cfg in self.flags.items() if not cfg.value_required}
        changed = 0
        skipped: list[str] = []
        extra_tokens: list[str] = []
        idx = 0
        while idx < len(tokens):
            token = tokens[idx]
            if token.startswith("--") and "=" in token:
                token, inline_value = token.split("=", 1)
            else:
                inline_value = None

            if token in {"-m", "--model"}:
                value, idx = self.consume_import_value(tokens, idx, inline_value, token)
                self.model_var.set(value)
                changed += 1
            elif token in alias_to_flag:
                flag = alias_to_flag[token]
                if flag in valueless_flags:
                    value = inline_value or ""
                else:
                    value, idx = self.consume_import_value(tokens, idx, inline_value, token)
                if flag == "--mmproj":
                    self.mmproj_var.set(value)
                elif flag == "-md":
                    if self.draft_model_var:
                        self.draft_model_var.set(value)
                elif flag in self.flags:
                    self.flags[flag].value = value
                    self.flags[flag].enabled = True
                    self.hidden_flags.discard(flag)
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
                try:
                    self.add_or_update_flag(token, value, value_required, True)
                    changed += 1
                except ValueError:
                    skipped.append(token)
                    extra_tokens.append(token if inline_value is None else f"{token}={inline_value}")
                    if value_required and value:
                        extra_tokens.append(value)
            idx += 1

        if extra_tokens and self.extra_args_var:
            current = shlex.split(self.extra_args_var.get()) if self.extra_args_var.get().strip() else []
            self.extra_args_var.set(shlex.join(current + extra_tokens))
        self.render_flags()
        self.recalculate_vram()
        return changed, skipped

    def consume_import_value(self, tokens: list[str], idx: int, inline_value: str | None, flag: str) -> tuple[str, int]:
        if inline_value is not None:
            return inline_value, idx
        if idx + 1 >= len(tokens):
            raise ValueError(f"{flag} needs a value.")
        return tokens[idx + 1], idx + 1

    def delete_selected_preset(self) -> None:
        if not self.selected_preset:
            messagebox.showwarning("No preset selected", "Select a preset first.")
            return
        self.history.delete_preset(self.selected_preset)
        self.selected_preset = None
        self.render_presets()



    def load_preset(self, preset: dict[str, Any]) -> None:
        self.loading_preset = True
        try:
            self.selected_preset = preset.get("preset_name")
            self.flags = self.default_flags()
            self.hidden_flags = set()
            saved_flags = preset.get("flags", {})
            for flag, saved in saved_flags.items():
                if flag not in self.flags and bool(saved.get("custom", False)):
                    self.flags[flag] = FlagConfig(
                        flag.lstrip("-") or flag,
                        str(saved.get("value", "") or ""),
                        bool(saved.get("enabled", True)),
                        bool(saved.get("value_required", True)),
                        custom=True,
                        step_mode=str(saved.get("step_mode", "") or ""),
                    )
            for flag, cfg in self.flags.items():
                saved = saved_flags.get(flag, {})
                cfg.value = str(saved.get("value", cfg.value) or "")
                cfg.enabled = bool(saved.get("enabled", cfg.enabled))
                if cfg.custom:
                    cfg.value_required = bool(saved.get("value_required", cfg.value_required))
                    cfg.step_mode = str(saved.get("step_mode", cfg.step_mode) or "")
            self.hidden_flags = {str(flag) for flag in preset.get("hidden_flags", []) if str(flag) in self.flags}
            inferer = str(preset.get("inferer") or "llama.cpp")
            if self.inferer_var:
                self.inferer_var.set(inferer if inferer in self.inferers else "llama.cpp")
            if self.inferer_executable_var:
                self.inferer_executable_var.set(str(preset.get("inferer_executable") or self.current_inferer().executable))
            self.model_var.set(str(preset.get("model_path") or saved_flags.get("-m", {}).get("value") or ""))
            self.mmproj_var.set(str(preset.get("mmproj_path") or saved_flags.get("--mmproj", {}).get("value") or ""))
            if self.draft_model_var:
                self.draft_model_var.set(str(preset.get("draft_model_path") or saved_flags.get("-md", {}).get("value") or ""))
            if self.extra_args_var:
                self.extra_args_var.set(str(preset.get("extra_args") or ""))
        finally:
            self.loading_preset = False
        self.saved_snapshot = self.snapshot_state()
        self.dirty = False
        self.render_flags()
        self.render_presets()
        self.recalculate_vram()
        stats = preset.get("session_stats", {})
        if stats:
            parts = []
            if "avg_ttft_ms" in stats:
                parts.append(f"{stats['avg_ttft_ms']}ms TTFT")
            if "avg_tok_s" in stats:
                parts.append(f"{stats['avg_tok_s']} tok/s")
            if stats.get("auto_restarts", 0):
                parts.append(f"{stats['auto_restarts']} restarts")
            self._session_stats_var.set("Last session: " + " | ".join(parts))
        else:
            self._session_stats_var.set("")
        if hasattr(self, "right_canvas"):
            self.root.after_idle(lambda: self.right_canvas.yview_moveto(0.0))

    def open_file_picker(self, target: str) -> None:
        if target == "model":
            start_path = self.model_var.get().strip()
        elif target == "draft":
            start_path = self.draft_model_var.get().strip() if self.draft_model_var else ""
        else:
            start_path = self.flags["--mmproj"].value
        remembered = str(self.history.settings.get("last_gguf_dir") or "")
        initial_dir = Path(start_path).expanduser().parent if start_path else Path(remembered).expanduser() if remembered else Path.home()
        if not initial_dir.exists():
            initial_dir = Path.home()
        result = self.choose_gguf_file(initial_dir)
        if result:
            self.history.set_setting("last_gguf_dir", str(Path(result).expanduser().parent))
            if target == "model":
                self.model_var.set(result)
            elif target == "draft":
                if self.draft_model_var:
                    self.draft_model_var.set(result)
            else:
                self.mmproj_var.set(result)

    def choose_gguf_file(self, initial_dir: Path) -> str | None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Select GGUF file")
        dialog.configure(bg=self.colors["bg"])
        dialog.geometry("820x560")
        dialog.minsize(640, 420)
        dialog.transient(self.root)
        dialog.grab_set()

        result: dict[str, str | None] = {"path": None}
        current_dir = {"path": initial_dir.expanduser().resolve()}
        entries: list[Path] = []

        shell = tk.Frame(dialog, bg=self.colors["bg"], padx=14, pady=14)
        shell.pack(fill="both", expand=True)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(2, weight=1)

        header = tk.Frame(shell, bg=self.colors["bg"])
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)
        tk.Label(header, text="Directory", bg=self.colors["bg"], fg=self.colors["muted"], font=("TkDefaultFont", 10, "bold")).grid(
            row=0, column=0, sticky="w", padx=(0, 10)
        )
        dir_var = tk.StringVar(value=str(current_dir["path"]))
        dir_entry = self.make_entry(header, dir_var)
        dir_entry.grid(row=0, column=1, sticky="ew", ipady=6)
        self.make_button(header, "Go", lambda: navigate(Path(dir_var.get())), width=44, bg=self.colors["panel_soft"]).grid(row=0, column=2, padx=(8, 0))
        self.make_button(header, "Home", lambda: navigate(Path.home()), width=56, bg=self.colors["panel_soft"]).grid(row=0, column=3, padx=(6, 0))

        file_var = tk.StringVar()
        hint = tk.Label(shell, text="Double-click a folder to open it, or a .gguf file to select it.", bg=self.colors["bg"], fg=self.colors["muted"])
        hint.grid(row=1, column=0, sticky="w", pady=(10, 8))

        list_wrap = tk.Frame(shell, bg=self.colors["panel"], highlightbackground=self.colors["border"], highlightthickness=1)
        list_wrap.grid(row=2, column=0, sticky="nsew")
        list_wrap.columnconfigure(0, weight=1)
        list_wrap.rowconfigure(0, weight=1)
        file_list = tk.Listbox(
            list_wrap,
            activestyle="none",
            bd=0,
            bg=self.colors["field"],
            fg=self.colors["text"],
            selectbackground=self.colors["accent"],
            selectforeground="#ffffff",
            highlightthickness=0,
            font=("DejaVu Sans Mono", 10),
            relief="flat",
        )
        file_list.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(list_wrap, command=file_list.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        file_list.configure(yscrollcommand=scroll.set)

        footer = tk.Frame(shell, bg=self.colors["bg"])
        footer.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        footer.columnconfigure(1, weight=1)
        tk.Label(footer, text="File", bg=self.colors["bg"], fg=self.colors["muted"], font=("TkDefaultFont", 10, "bold")).grid(
            row=0, column=0, sticky="w", padx=(0, 10)
        )
        file_entry = self.make_entry(footer, file_var)
        file_entry.grid(row=0, column=1, sticky="ew", ipady=6)
        self.make_button(footer, "Open", lambda: accept(), width=68).grid(row=0, column=2, padx=(8, 0))
        self.make_button(footer, "Cancel", dialog.destroy, width=72, bg=self.colors["panel_soft"]).grid(row=0, column=3, padx=(6, 0))

        def render() -> None:
            nonlocal entries
            file_list.delete(0, "end")
            dir_var.set(str(current_dir["path"]))
            rows: list[Path] = []
            if current_dir["path"].parent != current_dir["path"]:
                rows.append(current_dir["path"].parent)
                file_list.insert("end", "../")
            try:
                children = sorted(current_dir["path"].iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            except OSError as exc:
                messagebox.showerror("Cannot open directory", str(exc), parent=dialog)
                children = []
            for child in children:
                if child.is_dir() or child.suffix.lower() == ".gguf":
                    rows.append(child)
                    suffix = "/" if child.is_dir() else ""
                    file_list.insert("end", f"{child.name}{suffix}")
            entries = rows

        def navigate(path: Path) -> None:
            expanded = path.expanduser()
            if expanded.is_file():
                expanded = expanded.parent
            if not expanded.exists() or not expanded.is_dir():
                messagebox.showerror("Cannot open directory", f"{expanded} is not a directory.", parent=dialog)
                return
            current_dir["path"] = expanded.resolve()
            file_var.set("")
            render()

        def selected_path() -> Path | None:
            selection = file_list.curselection()
            if not selection:
                return None
            idx = selection[0]
            if idx >= len(entries):
                return None
            return entries[idx]

        def on_select(_event: tk.Event | None = None) -> None:
            path = selected_path()
            if path and path.is_file():
                file_var.set(path.name)

        def on_open(_event: tk.Event | None = None) -> None:
            path = selected_path()
            if not path:
                return
            if path.is_dir():
                navigate(path)
            else:
                file_var.set(path.name)
                accept()

        def accept() -> None:
            value = file_var.get().strip()
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = current_dir["path"] / value
            if not path.exists() or not path.is_file():
                messagebox.showerror("Select a file", "Choose an existing .gguf file.", parent=dialog)
                return
            if path.suffix.lower() != ".gguf":
                messagebox.showerror("Select a GGUF file", "The selected file must end with .gguf.", parent=dialog)
                return
            result["path"] = str(path)
            dialog.destroy()

        file_list.bind("<<ListboxSelect>>", on_select)
        file_list.bind("<Double-Button-1>", on_open)
        file_list.bind("<Return>", on_open)
        dir_entry.bind("<Return>", lambda _event: navigate(Path(dir_var.get())))
        file_entry.bind("<Return>", lambda _event: accept())
        render()
        file_list.focus_set()
        self.root.wait_window(dialog)
        return result["path"]

    def on_model_changed(self) -> None:
        self.mark_dirty()
        value = self.model_var.get().strip()
        if value and Path(value).expanduser().exists():
            self.model_meta = parse_gguf(value)
            for warning in self.model_meta.warnings:
                self.append_log("warn", warning)
        else:
            self.model_meta = None
        self.recalculate_vram()

    def update_optional_model_fields(self) -> None:
        if not hasattr(self, "mmproj_entry") or not self.draft_model_var:
            return
        mmproj_selected = bool(self.mmproj_var.get().strip())
        draft_selected = bool(self.draft_model_var.get().strip())
        self.mmproj_entry.grid_remove()
        self.draft_model_entry.grid_remove()
        self.draft_compact.grid_remove()
        if not mmproj_selected and not draft_selected:
            self.mmproj_label.grid(row=2, column=0, sticky="w", padx=(0, 12), pady=(10, 0))
            self.mmproj_button.grid(row=2, column=1, sticky="w", pady=(10, 0))
            self.draft_model_label.grid_remove()
            self.draft_model_button.grid_remove()
            self.draft_compact.grid(row=2, column=2, sticky="w", padx=(14, 0), pady=(10, 0))
            return
        self.mmproj_label.grid(row=2, column=0, sticky="w", padx=(0, 12), pady=(10, 0))
        self.draft_model_label.grid(row=3, column=0, sticky="w", padx=(0, 12), pady=(10, 0))
        if mmproj_selected:
            self.mmproj_entry.grid(row=2, column=1, columnspan=2, sticky="ew", pady=(10, 0), ipady=6)
            self.mmproj_button.grid(row=2, column=3, padx=(8, 0), pady=(10, 0))
        else:
            self.mmproj_button.grid(row=2, column=1, sticky="w", pady=(10, 0))
        if draft_selected:
            self.draft_model_entry.grid(row=3, column=1, columnspan=2, sticky="ew", pady=(10, 0), ipady=6)
            self.draft_model_button.grid(row=3, column=3, padx=(8, 0), pady=(10, 0))
        else:
            self.draft_model_button.grid(row=3, column=1, sticky="w", pady=(10, 0))

    def on_mmproj_changed(self) -> None:
        self.update_optional_model_fields()
        self.mark_dirty()
        value = self.mmproj_var.get().strip()
        self.flags["--mmproj"].value = value
        self.flags["--mmproj"].enabled = bool(value)
        self.recalculate_vram()

    def on_draft_model_changed(self) -> None:
        self.update_optional_model_fields()
        self.mark_dirty()
        value = self.draft_model_var.get().strip() if self.draft_model_var else ""
        if value and Path(value).expanduser().exists():
            self.draft_model_meta = parse_gguf(value)
            for warning in self.draft_model_meta.warnings:
                self.append_log("warn", warning)
        else:
            self.draft_model_meta = None
        self.recalculate_vram()

    def set_flag_value(self, flag: str, value: str) -> None:
        self.flags[flag].value = value
        self.mark_dirty()
        if flag == "--mmproj" and hasattr(self, "mmproj_var") and self.mmproj_var.get() != value:
            self.mmproj_var.set(value)
        self.recalculate_vram()

    def set_flag_enabled(self, flag: str, enabled: bool) -> None:
        self.flags[flag].enabled = enabled
        self.mark_dirty()
        self.recalculate_vram()

    def step_context_size(self, direction: int) -> None:
        self.step_numeric_flag("-c", direction)

    def scale_context_size(self, multiplier: float) -> None:
        self.scale_numeric_flag("-c", multiplier)

    def set_context_size(self, value: int) -> None:
        self.set_numeric_flag_value("-c", value)

    def step_numeric_flag(self, flag: str, direction: int) -> None:
        cfg = self.flags.get(flag)
        if not cfg:
            return
        current = self.get_int_flag(flag, 0)
        if cfg.step_mode == "power2":
            base = max(1, current or 1)
            next_value = base * 2 if direction > 0 else max(1, base // 2)
        else:
            next_value = max(1024, max(0, current) + (direction * 1024))
        self.set_numeric_flag_value(flag, next_value)

    def scale_numeric_flag(self, flag: str, multiplier: float) -> None:
        cfg = self.flags.get(flag)
        if not cfg:
            return
        current = max(1, self.get_int_flag(flag, 1))
        if cfg.step_mode == "power2":
            next_value = current * 2 if multiplier >= 1 else max(1, current // 2)
        else:
            next_value = max(1024, int(round((current * multiplier) / 1024)) * 1024)
        self.set_numeric_flag_value(flag, next_value)

    def set_numeric_flag_value(self, flag: str, value: int) -> None:
        if flag not in self.flags:
            return
        self.flags[flag].value = str(value)
        self.mark_dirty()
        value_var = self.flag_vars.get(flag, {}).get("value")
        if isinstance(value_var, tk.StringVar):
            value_var.set(str(value))
        else:
            self.recalculate_vram()

    def get_int_flag(self, flag: str, default: int) -> int:
        try:
            return int(str(self.flags[flag].value).strip())
        except Exception:
            return default

    def get_gpu_layers_for_estimate(self, total_layers: int) -> int:
        raw = str(self.flags["-ngl"].value).strip().lower()
        if raw in {"auto", "all", "-1"}:
            return total_layers
        try:
            return int(raw)
        except ValueError:
            return total_layers

    def describe_gpu_layers_for_estimate(self, total_layers: int, ngl: int) -> str:
        raw = str(self.flags["-ngl"].value).strip().lower()
        if raw == "auto":
            return f"auto -> estimating full offload ({min(ngl, total_layers)}/{total_layers} layers)"
        if raw in {"all", "-1"}:
            return f"{raw} -> full offload ({min(ngl, total_layers)}/{total_layers} layers)"
        return f"{min(ngl, total_layers)}/{total_layers} layers"

    def estimate_confidence(self) -> tuple[str, str]:
        """Returns (label, color) where color is a hex color for the badge dot."""
        if self.estimate.get("calibrated"):
            return ("calibrated", self.colors["conf_green"])
        if not self.model_meta:
            return ("no model selected", self.colors["conf_red"])
        if self.model_meta.n_layers and self.model_meta.n_embd_k_gqa and self.model_meta.n_embd_v_gqa:
            return ("full metadata", self.colors["conf_green"])
        if self.model_meta.n_layers and self.model_meta.n_embd and self.model_meta.n_kv_heads:
            return ("partial metadata", self.colors["conf_yellow"])
        return ("file-size only", self.colors["conf_red"])

    def matching_vram_calibration(self) -> dict[str, Any] | None:
        if not self.selected_preset:
            return None
        preset = None
        for candidate in self.history.presets:
            if candidate.get("preset_name") == self.selected_preset:
                preset = candidate
                break
        if not preset:
            return None
        calibration = preset.get("vram_calibration")
        if not self.vram_calibration_is_plausible(calibration):
            return None
        try:
            command = self.build_command(validate_model=False)
        except Exception:
            return None
        if core.calibration_matches(calibration, command, self.model_var.get().strip()):
            return calibration
        return None

    def vram_calibration_is_plausible(self, calibration: Any) -> bool:
        if not isinstance(calibration, dict):
            return False
        observed = calibration.get("observed_bytes")
        estimated = calibration.get("estimated_bytes")
        try:
            observed_value = float(observed)
            estimated_value = float(estimated)
        except (TypeError, ValueError):
            return False
        if observed_value <= 0 or estimated_value <= 0:
            return False
        log_allocations = calibration.get("log_allocations") if isinstance(calibration.get("log_allocations"), dict) else {}
        if log_allocations:
            return True
        ratio = observed_value / estimated_value
        return ratio >= 0.20

    def recalculate_vram(self) -> None:
        meta = self.model_meta
        model_size = float(meta.size) if meta else 0.0
        total_layers = max(1, meta.n_layers if meta else self.get_int_flag("-ngl", 80))
        ngl = max(0, self.get_gpu_layers_for_estimate(total_layers))
        context = max(1, self.get_int_flag("-c", 32768))
        n_layers = meta.n_layers if meta and meta.n_layers else total_layers
        n_embd = meta.n_embd if meta and meta.n_embd else 0
        n_kv_heads = meta.n_kv_heads if meta and meta.n_kv_heads else 1
        ctk_value = self.flags["-ctk"].value or "q4_0"
        ctv_value = self.flags["-ctv"].value or ctk_value
        k_bytes = KV_BYTES.get(ctk_value, 0.5)
        v_bytes = KV_BYTES.get(ctv_value, 0.5)
        k_dim = meta.n_embd_k_gqa if meta and meta.n_embd_k_gqa else 0
        v_dim = meta.n_embd_v_gqa if meta and meta.n_embd_v_gqa else 0

        # Sliding Window Attention: if the model has a sliding window, only
        # that many tokens are cached per SWA layer instead of the full context.
        # We use it as the effective context size for the estimate.
        sw = meta.sliding_window if meta and meta.sliding_window else 0
        effective_ctx = sw if sw > 0 else context

        if k_dim and v_dim:
            kv_cache = effective_ctx * n_layers * ((k_dim * k_bytes) + (v_dim * v_bytes))
        elif n_embd:
            fallback_dim = n_embd / max(1, n_kv_heads)
            kv_cache = effective_ctx * n_layers * ((fallback_dim * k_bytes) + (fallback_dim * v_bytes))
        else:
            fallback_dim = 128.0
            k_dim = int(fallback_dim)
            v_dim = int(fallback_dim)
            kv_cache = effective_ctx * n_layers * ((fallback_dim * k_bytes) + (fallback_dim * v_bytes))
        # -- MTP speculative KV cache --
        mtp_kv = 0.0
        mtp_active = False
        spec_type_val = str(self.flags.get("--spec-type", object()).value).strip().lower() if "--spec-type" in self.flags and self.flags["--spec-type"].enabled else ""
        for token in spec_type_val.split(","):
            token = token.strip()
            if token in ("mtp", "draft-mtp"):
                mtp_active = True
                break
        if mtp_active:
            n_draft = self.get_int_flag("--spec-draft-n-max", 3)
            n_draft = max(1, min(n_draft, 16))
            if k_dim and v_dim:
                mtp_kv = n_draft * n_layers * ((k_dim * k_bytes) + (v_dim * v_bytes))
            else:
                mtp_kv = n_draft * n_layers * ((fallback_dim * k_bytes) + (fallback_dim * v_bytes))
        mmproj_size = 0.0
        mmproj_path = self.flags["--mmproj"].value
        if self.flags["--mmproj"].enabled and mmproj_path and Path(mmproj_path).expanduser().exists():
            mmproj_size = float(Path(mmproj_path).expanduser().stat().st_size)
        # Draft model size (always counted if present, even under MTP when using -md)
        draft_size = float(self.draft_model_meta.size) if self.draft_model_meta else 0.0
        weights = (min(ngl, total_layers) / total_layers) * model_size
        overhead = 0.0
        if model_size:
            overhead = (1.0 * GB) + (weights * 0.08) + (kv_cache * 0.05)
        estimated = weights + draft_size + mmproj_size + kv_cache + mtp_kv + overhead if model_size else 0.0
        self.estimate = {
            "weights": weights,
            "kv": kv_cache,
            "mtp_kv": mtp_kv,
            "mmproj": mmproj_size,
            "draft": draft_size,
            "overhead": overhead,
            "estimated": estimated,
            "ngl": float(ngl),
            "total_layers": float(total_layers),
            "context": float(context),
            "k_dim": float(k_dim),
            "v_dim": float(v_dim),
        }
        calibration = self.matching_vram_calibration()
        if calibration and calibration.get("observed_bytes"):
            calibrated = int(calibration["observed_bytes"])
            self.estimate.update(
                {
                    "calibrated": calibrated,
                    "calibration_ratio": calibration.get("correction_ratio"),
                    "calibration_source": calibration.get("observed_source", "runtime"),
                    "calibration_created_at": calibration.get("created_at", ""),
                }
            )
        self.update_command_preview()
        self.update_vram_ui()

    def update_command_preview(self) -> None:
        try:
            command = self.build_command(validate_model=False)
            self.command_var.set(shlex.join(command))
        except Exception as exc:
            self.command_var.set(f"Command incomplete: {exc}")

    def copy_command(self) -> None:
        try:
            command_text = shlex.join(self.build_command(validate_model=False))
        except Exception as exc:
            messagebox.showerror("Cannot copy command", str(exc))
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(command_text)
        self.root.update()
        self.command_var.set(command_text)

    def _run_diagnostics_task(self, label: str, worker: Any) -> None:
        if self.diagnostics_busy:
            self.append_log("warn", "Diagnostics already running.")
            return
        preset = self.current_preset_payload()
        self.diagnostics_busy = True
        self.append_log("normal", f"── {label} ──")

        def run() -> None:
            try:
                ok, lines = worker(preset)
                for line in lines:
                    self.append_log("error" if line.startswith("[FAIL]") else "normal", line)
                if not ok:
                    self.append_log("error", f"{label} failed.")
            except Exception as exc:
                self.append_log("error", f"{label} failed: {exc}")
            finally:
                self.root.after(0, self._finish_diagnostics_task)

        threading.Thread(target=run, daemon=True).start()

    def _finish_diagnostics_task(self) -> None:
        self.diagnostics_busy = False

    def run_diagnostics_doctor(self) -> None:
        self._run_diagnostics_task("Doctor", lambda preset: core.doctor_report(preset))

    def run_diagnostics_probe(self) -> None:
        self._run_diagnostics_task("Probe", lambda preset: core.probe_report(preset))

    def run_diagnostics_bench(self) -> None:
        def worker(preset: dict[str, Any]) -> tuple[bool, list[str]]:
            row, _paths, lines = core.run_benchmark(preset)
            return row.get("status") == "pass", lines

        self._run_diagnostics_task("Bench", worker)

    def run_diagnostics_stress(self) -> None:
        self._run_diagnostics_task("Stress", lambda preset: core.context_stress_report(preset))

    def update_vram_ui(self) -> None:
        process_running = self.process is not None and self.process.poll() is None
        has_runtime = process_running and self.vram_used is not None
        idle_estimate = self.estimate.get("calibrated", self.estimate.get("estimated", 0.0))
        used = self.vram_used if has_runtime else idle_estimate
        total = self.vram_total
        total_used = self.vram_total_used
        percent = min(100.0, (used / total) * 100) if used and total else 0.0
        self.draw_vram_bar(percent)
        prefix = "Server" if has_runtime else "Calibrated" if self.estimate.get("calibrated") else "Estimated"
        bits = [f"{prefix} {human_bytes(used)}"]
        if total_used and total:
            bits.append(f"| GPU {human_bytes(total_used)} / {human_bytes(total)}")
        elif total:
            bits.append(f"of {human_bytes(total)}")
        bits.append(f"({self.vram_source})")
        self.vram_label_var.set(" ".join(bits))
        # Confidence badge
        conf_label, conf_color = self.estimate_confidence()
        self.vram_confidence_var.set("\u25cf")
        self.vram_confidence_label.configure(fg=conf_color)
        self.update_vram_breakdown()

    def _configure_flag_columns(self, n: int) -> None:
        """Set up equal-weight column configs for the flags grid."""
        existing_cols = self.flags_frame.grid_size()[0]
        for col in range(max(existing_cols, self._flags_last_col_count, n)):
            self.flags_frame.columnconfigure(col, weight=0, uniform="")
        for col in range(n):
            self.flags_frame.columnconfigure(col, weight=1, uniform="flag_columns")

    def _flags_col_count(self) -> int:
        """Compute how many flag columns fit in the current flags panel width."""
        width = self.flags_frame.winfo_width()
        if width <= 0:
            return self._flags_min_cols
        return max(self._flags_min_cols, width // self._flags_est_cell_width)

    def _on_flags_canvas_resize(self, _event: tk.Event) -> None:
        new_cols = self._flags_col_count()
        if new_cols != self._flags_last_col_count:
            self._flags_last_col_count = new_cols
            self.render_flags()

    def _on_right_canvas_resize(self, _event: tk.Event) -> None:
        self.right_canvas.itemconfigure(self.right_window, width=max(1, self.right_canvas.winfo_width()))
        self.update_right_scrollregion()

    def update_right_scrollregion(self) -> None:
        if not hasattr(self, "right_canvas") or not hasattr(self, "right_content"):
            return
        width = max(1, self.right_canvas.winfo_width())
        height = max(self.right_content.winfo_reqheight(), self.right_canvas.winfo_height())
        self.right_canvas.configure(scrollregion=(0, 0, width, height))
        if self.right_canvas.yview()[0] < 0.001:
            self.right_canvas.yview_moveto(0.0)

    def on_mousewheel(self, event: tk.Event) -> str | None:
        if not hasattr(self, "right_canvas"):
            return None
        x1 = self.right_canvas.winfo_rootx()
        y1 = self.right_canvas.winfo_rooty()
        x2 = x1 + self.right_canvas.winfo_width()
        y2 = y1 + self.right_canvas.winfo_height()
        if not (x1 <= event.x_root <= x2 and y1 <= event.y_root <= y2):
            return None
        if getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        else:
            delta = int(-event.delta / 120) if event.delta else 0
        if delta:
            self.right_canvas.yview_scroll(delta, "units")
            return "break"
        return None

    def draw_vram_bar(self, percent: float | None = None) -> None:
        if not hasattr(self, "vram_bar"):
            return
        canvas = self.vram_bar
        canvas.delete("all")
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        total = self.vram_total or self.estimate.get("calibrated", self.estimate.get("estimated", 0.0))
        if not total:
            return
        base_segments = [
            ("weights", max(0.0, float(self.estimate.get("weights", 0.0))), "#5b8ec9", "MODEL", "#ffffff"),
            ("kv", max(0.0, float(self.estimate.get("kv", 0.0))), "#c9a04e", "KV", "#0a0d14"),
            ("mtp_kv", max(0.0, float(self.estimate.get("mtp_kv", 0.0))), "#c4608a", "MTP", "#ffffff"),
            ("draft", max(0.0, float(self.estimate.get("draft", 0.0))), "#8a6cc7", "DRAFT", "#ffffff"),
            ("mmproj", max(0.0, float(self.estimate.get("mmproj", 0.0))), "#5a9060", "MMPROJ", "#ffffff"),
            ("overhead", max(0.0, float(self.estimate.get("overhead", 0.0))), "#c95a4e", "OVERHEAD", "#ffffff"),
        ]
        formula_total = sum(size for _key, size, _color, _short, _text_color in base_segments)
        calibrated = max(0.0, float(self.estimate.get("calibrated", 0.0) or 0.0))
        segments = base_segments
        if calibrated and formula_total > 0 and calibrated < formula_total:
            scale = calibrated / formula_total
            segments = [
                (key, size * scale, color, short, text_color)
                for key, size, color, short, text_color in base_segments
            ]
        elif calibrated and calibrated > formula_total:
            segments = [*base_segments, ("calibrated_delta", calibrated - formula_total, "#6b7280", "CAL", "#ffffff")]
        elif calibrated and formula_total <= 0:
            segments = [("calibrated", calibrated, "#6b7280", "CAL", "#ffffff")]
        x = 0.0
        label_y = height // 2
        for _key, size, color, short, text_color in segments:
            segment_width = (size / total) * width
            if segment_width <= 0:
                continue
            x2 = min(width, x + segment_width)
            canvas.create_rectangle(x, 0, x2, height, fill=color, outline="")
            # Inside label — short name + compact GiB value if wide enough
            if segment_width > 80:
                compact = f"{human_bytes(size)}".replace(" ", "")
                label = f"{short} {compact}"
                canvas.create_text(x + 6, label_y, text=label, fill=text_color, font=("DejaVu Sans Mono", 10, "bold"), anchor="w")
            x = x2
        # Total GPU usage marker (dimmer line)
        if self.vram_total_used is not None and total:
            gpu_percent = min(100.0, (self.vram_total_used / total) * 100)
            gpu_width = min(width, (gpu_percent / 100.0) * width)
            canvas.create_line(gpu_width, 0, gpu_width, height, fill="#8b949e", width=1)
        # Server usage marker (bright white line)
        if percent is not None and percent > 0:
            used_width = min(width, (percent / 100.0) * width)
            canvas.create_line(used_width, 0, used_width, height, fill="#ffffff", width=2)

    def update_vram_breakdown(self) -> None:
        if not hasattr(self, "vram_breakdown_var"):
            return
        total_layers = int(self.estimate.get("total_layers", 0))
        ngl = int(self.estimate.get("ngl", 0))
        conf_label, conf_color = self.estimate_confidence()
        layer_text = self.describe_gpu_layers_for_estimate(total_layers, ngl) if total_layers else "no model layers available"
        mtp_kv_bytes = self.estimate.get("mtp_kv", 0.0)
        parts = [
            f"Model {human_bytes(self.estimate.get('weights', 0.0))}",
            f"Context/KV {human_bytes(self.estimate.get('kv', 0.0))}",
        ]
        if mtp_kv_bytes > 0:
            parts.append(f"MTP KV {human_bytes(mtp_kv_bytes)}")
        parts.extend([
            f"Draft {human_bytes(self.estimate.get('draft', 0.0))}",
            f"MMProj {human_bytes(self.estimate.get('mmproj', 0.0))}",
            f"Overhead {human_bytes(self.estimate.get('overhead', 0.0))}",
        ])
        if self.estimate.get("calibrated"):
            parts.append(f"Calibrated {human_bytes(self.estimate.get('calibrated', 0.0))}")
        legend = "Model blue + Context gold + MTP rose + Draft violet + MMProj green + Overhead coral\n"
        meta = self.model_meta
        sw_text = f" | SWA: {meta.sliding_window} (KV est. uses SWA window)" if meta and meta.sliding_window else ""
        self.vram_breakdown_var.set(
            legend
            + " + ".join(parts)
            + f" = {human_bytes(self.estimate.get('calibrated', self.estimate.get('estimated', 0.0)))}\n"
            + f"-c: {int(self.estimate.get('context', 0.0))} tok | K {int(self.estimate.get('k_dim', 0.0))}, V {int(self.estimate.get('v_dim', 0.0))} | -ngl: {layer_text} | confidence: {conf_label}{sw_text}"
        )

    def build_command(self, validate_model: bool = True) -> list[str]:
        preset = self.current_preset_payload()
        for flag, cfg in preset.get("flags", {}).items():
            if not self.flag_supported_by_current_inferer(flag):
                cfg["enabled"] = False
        return core.build_command_from_preset(preset, validate_paths=validate_model)

    def launch(self) -> None:
        try:
            command = self.build_command()
        except Exception as exc:
            messagebox.showerror("Cannot launch", str(exc))
            return
        executable = command[0]
        executable_exists = Path(executable).expanduser().exists() if any(sep in executable for sep in ("/", "\\")) else bool(shutil.which(executable))
        if not executable_exists:
            messagebox.showerror("Cannot launch", f"{executable} was not found. Check the inferer executable field.")
            return
        self.intentional_stop = False
        self.clear_logs()
        self.start_process(command, add_separator=False)

    def start_process(self, command: list[str], add_separator: bool = False) -> None:
        self.model_loaded = False
        self._server_ready = False
        self._stop_logged = False
        self.reset_session_stats()
        self.reset_vram_calibration_session(command)
        self.kill_existing_on_port(self.get_int_flag("--port", DEFAULT_SERVER_PORT))
        if add_separator:
            self.append_log("warn", "Auto-restarting inferer...")
        self.append_log("normal", f"$ {shlex.join(command)}")
        self.set_status("Launching")
        try:
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                start_new_session=(os.name != "nt"),
            )
            self.history.add_run(self.model_var.get().strip(), command)
            threading.Thread(target=self.stream_pipe, args=(self.process.stdout,), daemon=True).start()
            threading.Thread(target=self.stream_pipe, args=(self.process.stderr,), daemon=True).start()
        except Exception as exc:
            self.set_status("Error")
            self.append_log("error", f"Launch failed: {exc}")

    def reset_vram_calibration_session(self, command: list[str]) -> None:
        self._vram_command = list(command)
        self._vram_command_hash = core.command_hash(command)
        self._vram_log_allocations = {}
        self._vram_calibration_saved_for_hash = ""
        self._vram_calibration_wait_logged = False
        try:
            _used, total_used, _total, _source = self.gpu.usage(None)
        except Exception:
            total_used = None
        self._vram_baseline_total_used = total_used

    def kill_existing_on_port(self, port: int) -> None:
        if os.name == "nt" or not shutil.which("lsof"):
            return
        try:
            output = subprocess.check_output(["lsof", f"-tiTCP:{port}", "-sTCP:LISTEN"], text=True, stderr=subprocess.DEVNULL, timeout=3)
        except Exception:
            return
        for pid_text in output.splitlines():
            try:
                pid = int(pid_text)
            except ValueError:
                continue
            self.append_log("warn", f"Stopping process on port {port}: pid {pid}")
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                continue

    def stream_pipe(self, pipe: Any) -> None:
        if pipe is None:
            return
        for line in iter(pipe.readline, ""):
            self.append_log(self.classify_log(line), line.rstrip())

    def stop_process(self) -> None:
        self.intentional_stop = True
        self._stop_health_check()
        if self.process and self.process.poll() is None:
            self.append_log("warn", "Stopping inferer...")
            try:
                if os.name != "nt":
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                else:
                    self.process.terminate()
            except Exception:
                self.process.terminate()
            self.set_status("Stopping")
        else:
            self.set_status("Stopped")
            if not self._stop_logged:
                self.append_log("normal", "Server already stopped.")
                self._stop_logged = True
        self._server_ready = False
        self._save_session_stats_to_preset()

    def refresh_process_state(self) -> None:
        if self.process and self.process.poll() is not None and self.status_var.get() in {"Launching", "Running", "Ready", "Stopping"}:
            code = self.process.returncode
            self._stop_health_check()
            self._server_ready = False
            if code == 0 or self.intentional_stop:
                self.set_status("Stopped")
                if self.intentional_stop and not self._stop_logged:
                    self.append_log("normal", "Server stopped.")
                    self._stop_logged = True
            else:
                self.set_status("Crashed")
                self.append_log("error", f"inferer exited with code {code}")
                if hasattr(self, "auto_restart_var") and self.auto_restart_var.get():
                    try:
                        command = self.build_command()
                    except Exception as exc:
                        self.append_log("error", f"Auto-restart skipped: {exc}")
                    else:
                        self._auto_restart_count += 1
                        self.root.after(800, lambda cmd=command: self.start_process(cmd, add_separator=True))
            self._save_session_stats_to_preset()
        self.root.after(600, self.refresh_process_state)

    def refresh_gpu_usage(self) -> None:
        process_running = self.process is not None and self.process.poll() is None
        target_pid = self.process.pid if process_running else None
        self.vram_used, self.vram_total_used, self.vram_total, self.vram_source = self.gpu.usage(target_pid)
        if process_running and self.model_loaded:
            self.maybe_save_vram_calibration()
        self.update_vram_ui()
        # Poll every 2s while server is running, every 30s when idle
        interval = 2000 if process_running else 30000
        self.root.after(interval, self.refresh_gpu_usage)

    def set_status(self, status: str) -> None:
        if status == "Running" and self._server_ready:
            display = "Ready"
        elif status == "Running":
            display = "Running"
        else:
            display = status
        self.status_var.set(display)
        if hasattr(self, "status_label"):
            colors = {
                "Stopped": self.colors["panel_soft"],
                "Launching": self.colors["warn"],
                "Stopping": self.colors["warn"],
                "Running": self.colors["accent"],
                "Ready": self.colors["good"],
                "Error": self.colors["danger"],
                "Crashed": self.colors["danger"],
            }
            self.status_label.configure(bg=colors.get(display, self.colors["panel_soft"]))

    def _get_server_url(self) -> str:
        """Build the server URL from host and port flags."""
        host = self.flags.get("--host", FlagConfig("", "127.0.0.1", False)).value.strip() or "127.0.0.1"
        port = self.get_int_flag("--port", DEFAULT_SERVER_PORT)
        return f"http://{host}:{port}"

    def open_browser_ui(self) -> None:
        """Open the llama.cpp web UI in the default browser."""
        url = self._get_server_url()
        webbrowser.open(url)
        self.append_log("normal", f"Opening browser at {url}")

    def _start_health_check(self) -> None:
        """Start polling the /health endpoint to detect when the server is ready."""
        self._server_ready = False
        self._health_check_tick()

    def _stop_health_check(self) -> None:
        """Cancel any pending health check timer."""
        if self._health_check_timer:
            self.root.after_cancel(self._health_check_timer)
            self._health_check_timer = None

    def _health_check_tick(self) -> None:
        """Poll /health once. On success, mark server as ready. On failure, retry."""
        if not self.process or self.process.poll() is not None:
            self._server_ready = False
            self.set_status("Stopped")
            self._health_check_timer = None
            return
        url = self._get_server_url() + "/health"
        try:
            resp = urllib_req.urlopen(url, timeout=2)
            resp.read()
        except Exception:
            # Server not ready yet — poll again in 500ms
            self._health_check_timer = self.root.after(500, self._health_check_tick)
            return
        # Health check passed
        self._server_ready = True
        self._health_check_timer = None
        self.set_status("Ready")
        self.maybe_save_vram_calibration()
        self.append_log("normal", f"Server ready at {self._get_server_url()}")

    def classify_log(self, line: str) -> str:
        lower = line.lower()
        if any(token in lower for token in ("error", "exception", "failed", "fatal")):
            return "error"
        if any(token in lower for token in ("warn", "warning", "fallback")):
            return "warn"
        return "normal"

    def clear_logs(self) -> None:
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.configure(state="disabled")
        self.last_tokens_per_second = None
        if self.tokps_var:
            self.tokps_var.set("")
        # Reset restart counter on manual launch
        self._auto_restart_count = 0

    def reset_session_stats(self) -> None:
        self._session_ttft_ms.clear()
        self._session_gen_tokens = 0
        self._session_gen_time_ms = 0.0
        if self._session_stats_var:
            self._session_stats_var.set("")

    def capture_vram_allocation(self, text: str) -> None:
        event = core.parse_llama_vram_log_line(text)
        if not event:
            return
        category, bytes_used = event
        self._vram_log_allocations[category] = self._vram_log_allocations.get(category, 0) + bytes_used

    def observed_vram_for_calibration(self, *, fresh: bool = False) -> tuple[int, str] | None:
        vram_used = self.vram_used
        total_used = self.vram_total_used
        source = self.vram_source
        if fresh and self.process and self.process.poll() is None:
            try:
                vram_used, total_used, self.vram_total, source = self.gpu.usage(self.process.pid)
                self.vram_used = vram_used
                self.vram_total_used = total_used
                self.vram_source = source
            except Exception:
                pass
        if vram_used and vram_used > 0:
            source = self.vram_source or "GPU process usage"
            if "AMD" not in source.upper():
                return int(vram_used), source
        if total_used is not None and self._vram_baseline_total_used is not None:
            delta = total_used - self._vram_baseline_total_used
            if delta > 0:
                return int(delta), f"{source} total delta"
        log_total = sum(self._vram_log_allocations.values())
        if log_total > 0:
            return int(log_total), "llama-server allocation logs"
        return None

    def maybe_save_vram_calibration(self) -> None:
        if not self.selected_preset or not self._vram_command_hash:
            return
        observed = self.observed_vram_for_calibration(fresh=True)
        estimated = float(self.estimate.get("estimated", 0.0))
        if not observed or estimated <= 0:
            return
        observed_bytes, observed_source = observed
        log_total = sum(self._vram_log_allocations.values())
        if not log_total and observed_bytes < estimated * 0.20:
            if not self._vram_calibration_wait_logged:
                self.append_log(
                    "warn",
                    f"VRAM calibration waiting for stable GPU reading: observed {human_bytes(observed_bytes)} vs estimate {human_bytes(estimated)}.",
                )
                self._vram_calibration_wait_logged = True
            return
        preset = None
        for candidate in self.history.presets:
            if candidate.get("preset_name") == self.selected_preset:
                preset = candidate
                break
        if preset is None:
            return
        existing = preset.get("vram_calibration") if isinstance(preset.get("vram_calibration"), dict) else None
        if existing and core.calibration_matches(existing, self._vram_command, self.model_var.get().strip()):
            existing_observed = int(existing.get("observed_bytes") or 0)
            if self._vram_calibration_saved_for_hash == self._vram_command_hash and observed_bytes <= existing_observed * 1.05:
                return
            if observed_bytes <= existing_observed and self.vram_calibration_is_plausible(existing):
                return
        calibration = core.make_vram_calibration(
            preset_name=self.selected_preset,
            command=self._vram_command,
            model_path=self.model_var.get().strip(),
            estimated_bytes=estimated,
            observed_bytes=observed_bytes,
            observed_source=observed_source,
            log_allocations=self._vram_log_allocations,
        )
        preset["vram_calibration"] = calibration
        self.history.upsert_preset(preset)
        self._vram_calibration_saved_for_hash = self._vram_command_hash
        self.recalculate_vram()
        self.append_log(
            "normal",
            f"VRAM calibrated: observed {human_bytes(observed_bytes)} from {observed_source}; "
            f"estimate ratio {calibration.get('correction_ratio')}",
        )

    def _update_session_display(self) -> None:
        if not self._session_stats_var:
            return
        parts = []
        if self._session_ttft_ms:
            avg_ttft = sum(self._session_ttft_ms) / len(self._session_ttft_ms)
            parts.append(f"{avg_ttft:.0f}ms TTFT")
        if self._session_gen_tokens > 0 and self._session_gen_time_ms > 0:
            avg_tps = (self._session_gen_tokens / self._session_gen_time_ms) * 1000.0
            parts.append(f"{avg_tps:.2f} tok/s")
        self._session_stats_var.set("  ".join(parts))

    def _save_session_stats_to_preset(self) -> None:
        """Save current session averages to the preset in history.json."""
        if not self.selected_preset or not self.history.presets:
            return
        avg_ttft = 0.0
        avg_tok_s = 0.0
        if self._session_ttft_ms:
            avg_ttft = sum(self._session_ttft_ms) / len(self._session_ttft_ms)
        if self._session_gen_tokens > 0 and self._session_gen_time_ms > 0:
            avg_tok_s = (self._session_gen_tokens / self._session_gen_time_ms) * 1000.0
        preset = None
        for p in self.history.presets:
            if p.get("preset_name") == self.selected_preset:
                preset = p
                break
        if preset is None:
            return
        preset["session_stats"] = {
            "avg_ttft_ms": round(avg_ttft, 1),
            "avg_tok_s": round(avg_tok_s, 2),
            "auto_restarts": self._auto_restart_count,
        }
        self.history.upsert_preset(preset)

    def append_log(self, level: str, text: str) -> None:
        if threading.current_thread() is not threading.main_thread():
            self.log_queue.put((level, text))
            return
        self.capture_vram_allocation(text)
        self.capture_tokens_per_second(text)
        self.update_status_from_log(text)
        self.output.configure(state="normal")
        # Drop oldest lines if log is too large
        lines = int(self.output.index("end-1c").split(".")[0])
        if lines > 5000:
            self.output.delete("1.0", "3.0")
        self.output.insert("end", text + "\n", level)
        self.output.see("end")
        self.output.configure(state="disabled")

    def update_status_from_log(self, text: str) -> None:
        if self.model_loaded:
            return
        lower = text.lower()
        if "main: model loaded" in lower or "server is listening on" in lower or "all slots are idle" in lower:
            self.model_loaded = True
            if self.process and self.process.poll() is None:
                self.set_status("Running")
                # Kick off health check polling
                self._start_health_check()

    def capture_tokens_per_second(self, text: str) -> None:
        try:
            # Prompt eval time → TTFT (Time To First Token)
            ttft_match = re.search(
                r"prompt\s+eval\s+(?:took|time\s*=\s*)\s*([0-9]+(?:\.[0-9]+)?)\s*ms\s*/\s*(\d+)\s*tokens",
                text,
                re.IGNORECASE,
            )
            if ttft_match:
                ms = float(ttft_match.group(1))
                self._session_ttft_ms.append(ms)

            # Generation eval time → average tok/s accumulator
            # Match "eval" (took|time =) but not when preceded by "prompt"
            gen_match = re.search(
                r"(?<!prompt\s)eval\s+(?:took|time\s*=\s*)\s*([0-9]+(?:\.[0-9]+)?)\s*ms\s*/\s*(\d+)\s*tokens",
                text,
                re.IGNORECASE,
            )
            if gen_match:
                ms = float(gen_match.group(1))
                tokens = int(gen_match.group(2))
                self._session_gen_tokens += tokens
                self._session_gen_time_ms += ms

            # Legacy: keep last tok/s for display as well
            match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:tokens/s|tok/s|t/s)", text, re.IGNORECASE)
            if match:
                try:
                    self.last_tokens_per_second = float(match.group(1))
                except ValueError:
                    pass
                if self.tokps_var:
                    self.tokps_var.set(f"{self.last_tokens_per_second:.2f} tok/s")

            self._update_session_display()
        except Exception:
            pass  # future llama-server log format change — stats just stop updating, don't crash

    def drain_log_queue(self) -> None:
        while True:
            try:
                level, text = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.append_log(level, text)
        self.root.after(100, self.drain_log_queue)

    def on_close(self) -> None:
        if self.process and self.process.poll() is None:
            if messagebox.askyesno("Server running", "The inferer server is still running. Stop it before closing?", parent=self.root):
                self.stop_process()
        self.root.destroy()

    def run(self) -> None:
        # Handle Ctrl+C to gracefully stop server and save session stats
        self._sigint_received = False
        signal.signal(signal.SIGINT, lambda s, f: setattr(self, "_sigint_received", True))
        self.root.after(200, self._check_sigint)
        self.root.mainloop()

    def _check_sigint(self) -> None:
        if self._sigint_received:
            self._sigint_received = False
            if self.process and self.process.poll() is None:
                self.append_log("warn", "SIGINT received, stopping inferer...")
                try:
                    if os.name != "nt":
                        os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                    else:
                        self.process.terminate()
                except Exception:
                    self.process.terminate()
            self._save_session_stats_to_preset()
            self.root.destroy()
            return
        self.root.after(200, self._check_sigint)


CLI_COMMANDS = {
    "list", "create", "import", "show", "set", "enable", "disable",
    "rmflag", "rename", "delete", "run", "doctor", "probe", "bench",
    "stress", "export-presets", "import-presets", "help",
}


def parse_launch_mode(argv: list[str]) -> tuple[str, list[str]]:
    if not argv:
        return "auto", []
    if argv[0] == "--gui":
        return "gui", argv[1:]
    if argv[0] == "--cli":
        return "cli", argv[1:]
    if argv[0] == "--mode":
        if len(argv) < 2 or argv[1] not in {"auto", "gui", "cli"}:
            print("usage: llamawrap.py [--gui | --cli | --mode auto|gui|cli] [cli args...]", file=sys.stderr)
            raise SystemExit(1)
        return argv[1], argv[2:]
    return "auto", argv


def run_cli(argv: list[str]) -> None:
    cli_path = APP_DIR / "llamawrap-cli.py"
    if not cli_path.exists():
        cli_path = Path(__file__).resolve().with_name("llamawrap-cli.py")
    if not cli_path.exists():
        print("error: llamawrap-cli.py not found next to launcher", file=sys.stderr)
        raise SystemExit(1)

    old_argv = sys.argv[:]
    try:
        sys.argv = [str(cli_path), *argv]
        runpy.run_path(str(cli_path), run_name="__main__")
    finally:
        sys.argv = old_argv


def run_gui() -> None:
    if tk is None:
        print(f"{APP_TITLE} GUI is unavailable: {TK_IMPORT_ERROR}", file=sys.stderr)
        raise SystemExit(1)
    LauncherApp().run()


def launched_from_terminal() -> bool:
    return bool(getattr(sys.stdin, "isatty", lambda: False)() and getattr(sys.stdout, "isatty", lambda: False)())


def main(argv: list[str] | None = None) -> None:
    mode, remaining = parse_launch_mode(list(sys.argv[1:] if argv is None else argv))

    if mode == "cli":
        run_cli(remaining)
        return
    if mode == "gui":
        if remaining:
            print("error: --gui does not accept CLI command arguments", file=sys.stderr)
            raise SystemExit(1)
        run_gui()
        return

    if remaining and (remaining[0] in CLI_COMMANDS or remaining[0] in {"-h", "--help"}):
        run_cli(remaining)
        return

    if not remaining and launched_from_terminal():
        run_cli([])
        return

    try:
        run_gui()
    except SystemExit:
        if tk is not None:
            raise
        run_cli(remaining)
    except TK_TCL_ERROR as exc:
        print(f"{APP_TITLE} GUI is unavailable, falling back to CLI: {exc}", file=sys.stderr)
        run_cli(remaining)


if __name__ == "__main__":
    main()
