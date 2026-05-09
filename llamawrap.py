from __future__ import annotations

import json
import os
import queue
import re
import shlex
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk
from typing import Any


APP_TITLE = "llama-wrap"
DEFAULT_SERVER_PORT = 8123
APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
HISTORY_FILE = APP_DIR / "history.json"
GB = 1024**3
KV_BYTES = {"q4_0": 0.5, "q4_1": 0.5, "q5_0": 0.625, "q5_1": 0.625, "q8_0": 1.0, "f16": 2.0}


@dataclass
class FlagConfig:
    label: str
    value: str = ""
    enabled: bool = True
    value_required: bool = True
    choices: tuple[str, ...] = ()
    inferers: tuple[str, ...] = ()
    custom: bool = False


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
    warnings: list[str] = field(default_factory=list)


class HistoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.presets: list[dict[str, Any]] = []
        self.runs: list[dict[str, Any]] = []
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.save()
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.presets = list(data.get("presets", []))
            self.runs = list(data.get("runs", []))
        except Exception:
            backup = self.path.with_suffix(f".corrupt-{int(time.time())}.json")
            self.path.rename(backup)
            self.presets = []
            self.runs = []
            self.save()

    def save(self) -> None:
        data = {"presets": self.presets, "runs": self.runs[-100:]}
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

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
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024 or unit == "TB":
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
        0: "<?",
        1: "<b",
        2: "<B",
        3: "<h",
        4: "<H",
        5: "<i",
        6: "<I",
        7: "<f",
        10: "<q",
        11: "<Q",
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
        if count > 128:
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


def parse_gguf(path: str) -> GGUFMetadata:
    model = Path(path).expanduser()
    meta = GGUFMetadata(path=str(model), size=model.stat().st_size)
    try:
        with model.open("rb") as fh:
            data = fh.read(min(max(meta.size, 128), 64 * 1024 * 1024))
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
            value, offset = read_gguf_value(data, offset, value_type)
            if metadata_key_matches(key, ("block_count", "n_layers", "n_layer")):
                meta.n_layers = int(value)
            elif metadata_key_matches(key, ("embedding_length", "n_embd")):
                meta.n_embd = int(value)
            elif metadata_key_matches(key, ("attention.head_count_kv", "n_head_kv", "n_kv_heads")):
                meta.n_kv_heads = int(value)
            if meta.n_layers and meta.n_embd and meta.n_kv_heads:
                break
    except Exception:
        meta.warnings.append("The launcher could not read all model metadata. Launching should still work, but the VRAM estimate may be less accurate.")
    return meta


def find_mmproj(model_path: str) -> str:
    model = Path(model_path).expanduser()
    if not model.exists():
        return ""
    candidates: list[Path] = []
    for path in model.parent.glob("*.gguf"):
        lower = path.name.lower()
        if path != model and ("mmproj" in lower or "vision" in lower or "projector" in lower):
            candidates.append(path)
    if not candidates:
        return ""
    model_tokens = set(re.split(r"[-_.\s]+", model.stem.lower()))
    candidates.sort(key=lambda p: len(model_tokens.intersection(re.split(r"[-_.\s]+", p.stem.lower()))), reverse=True)
    return str(candidates[0])


class GPUMonitor:
    def usage(self) -> tuple[int | None, int | None, str]:
        nvidia = self._nvidia_usage()
        if nvidia[2] != "not found":
            return nvidia
        amd = self._amd_usage()
        if amd[2] != "not found":
            return amd
        return None, None, "GPU not detected"

    def _nvidia_usage(self) -> tuple[int | None, int | None, str]:
        if not shutil.which("nvidia-smi"):
            return None, None, "not found"
        try:
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
            return used_mib * 1024 * 1024, total_mib * 1024 * 1024, "NVIDIA"
        except Exception:
            return None, None, "NVIDIA"

    def _amd_usage(self) -> tuple[int | None, int | None, str]:
        if not shutil.which("rocm-smi"):
            return None, None, "not found"
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
            return used, total, "AMD"
        except Exception:
            return None, None, "AMD"


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
            bg="#0b1018",
            fg="#e6edf3",
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
        self.process: subprocess.Popen[str] | None = None
        self.selected_preset: str | None = None
        self.inferers = self.default_inferers()
        self.flags = self.default_flags()
        self.hidden_flags: set[str] = set()
        self.flag_vars: dict[str, dict[str, tk.Variable]] = {}
        self.log_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.estimate: dict[str, float] = {}
        self.vram_used: int | None = None
        self.vram_total: int | None = None
        self.vram_source = "GPU not detected"
        self.model_loaded = False
        self.intentional_stop = False
        self.extra_args_var: tk.StringVar | None = None
        self.inferer_var: tk.StringVar | None = None
        self.inferer_executable_var: tk.StringVar | None = None

        self.root = tk.Tk()
        self.root.title(APP_TITLE)
        self.root.geometry("1060x720")
        self.root.minsize(900, 620)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.configure_style()
        self.build_ui()
        self.render_presets()
        self.render_flags()
        self.recalculate_vram()
        self.refresh_process_state()
        self.refresh_gpu_usage()
        self.drain_log_queue()

    def default_inferers(self) -> dict[str, InfererConfig]:
        return {
            "llama.cpp": InfererConfig(
                "llama-server",
            ),
            "ik_llama.cpp": InfererConfig(
                "llama-server",
            ),
            "Custom": InfererConfig(
                "llama-server",
            ),
        }

    def default_flags(self) -> dict[str, FlagConfig]:
        cpu_threads = max(1, os.cpu_count() or 1)
        return {
            "-ngl": FlagConfig("GPU layers", "auto"),
            "-c": FlagConfig("Context size", "32768"),
            "-t": FlagConfig("Threads", str(cpu_threads)),
            "-tb": FlagConfig("Batch threads", "", False),
            "-fa": FlagConfig("Flash attention", "auto", True, True, ("auto", "on", "off")),
            "-ctk": FlagConfig("KV cache K", "q4_0", True, True, ("f16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1")),
            "-ctv": FlagConfig("KV cache V", "q4_0", True, True, ("f16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1")),
            "--port": FlagConfig("Port", str(DEFAULT_SERVER_PORT)),
            "--host": FlagConfig("Host", "127.0.0.1"),
            "-b": FlagConfig("Batch", "2048"),
            "-ub": FlagConfig("UBatch", "512"),
            "-np": FlagConfig("Parallel slots", "-1"),
            "--threads-http": FlagConfig("HTTP threads", "-1", False),
            "-a": FlagConfig("Alias", "", False),
            "-to": FlagConfig("Timeout", "600", False),
            "--jinja": FlagConfig("Jinja tools", "", False, False),
            "--fit": FlagConfig("Auto fit VRAM", "", False, False, (), ("ik_llama.cpp",)),
            "--fit-margin": FlagConfig("Fit margin MiB", "1024", False, True, (), ("ik_llama.cpp",)),
            "-mla": FlagConfig("MLA mode", "3", False, True, ("0", "1", "2", "3"), ("ik_llama.cpp",)),
            "-fmoe": FlagConfig("Fused MoE", "", False, False, (), ("ik_llama.cpp",)),
            "-cram": FlagConfig("RAM prompt cache", "8192", False, True, (), ("ik_llama.cpp",)),
            "-khad": FlagConfig("K Hadamard", "", False, False, (), ("ik_llama.cpp",)),
            "-vhad": FlagConfig("V Hadamard", "", False, False, (), ("ik_llama.cpp",)),
            "--mmproj": FlagConfig("MMProj", "", False),
        }

    def flag_help(self) -> dict[str, str]:
        return {
            "-ngl": "GPU layers to keep in VRAM. Use 'auto' for llama.cpp's automatic fit, 'all' to try full offload, 0 for CPU-only, or a number for a fixed layer count.",
            "-c": "Context size in tokens. Larger values allow longer prompts/conversations but use more KV-cache memory. 0 uses the model default.",
            "-t": "CPU generation threads. Usually set near your physical CPU core count; -1 lets llama.cpp choose.",
            "-tb": "CPU threads used for prompt/batch processing. Leave disabled to use the same value as generation threads.",
            "-fa": "Flash Attention mode. auto lets llama.cpp decide, on forces it, off disables it. Use off if your GPU/backend reports flash-attention errors.",
            "-ctk": "KV cache data type for K. f16 is highest compatibility; q8_0/q5/q4 use less memory with possible quality or speed tradeoffs.",
            "-ctv": "KV cache data type for V. Match K for simplicity; quantized values reduce VRAM/RAM for long contexts.",
            "--port": "HTTP port for llama-server. Use this in clients as http://host:port, for example 8123.",
            "--host": "Bind address. 127.0.0.1 is local-only. 0.0.0.0 exposes the server to other devices on reachable networks.",
            "-b": "Logical batch size. Higher can improve prompt processing throughput but uses more memory. Current llama.cpp default is 2048.",
            "-ub": "Physical micro-batch size. Lower if you hit memory errors during prompt processing. Current llama.cpp default is 512.",
            "-np": "Number of parallel server slots. -1 lets llama.cpp choose. Higher supports more simultaneous requests but increases memory use.",
            "--threads-http": "HTTP worker threads for serving requests. -1 lets llama.cpp choose.",
            "-a": "Model alias shown to API clients. Useful when clients expect a specific model name.",
            "-to": "Server read/write timeout in seconds.",
            "--jinja": "Enable Jinja chat templates. Needed by some tool/function-calling clients and supported by llama.cpp-style servers.",
            "--fit": "ik_llama.cpp only. Automatically fits as many tensors as possible into available VRAM instead of choosing a fixed layer count.",
            "--fit-margin": "ik_llama.cpp only. Safety VRAM margin in MiB used with --fit. Increase if model loading hits CUDA out-of-memory.",
            "-mla": "ik_llama.cpp only. MLA mode for DeepSeek-style MLA models. Leave disabled unless the model/docs recommend it.",
            "-fmoe": "ik_llama.cpp only. Enable fused MoE kernels for mixture-of-experts models when supported by your build.",
            "-cram": "ik_llama.cpp only. Prompt/KV cache kept in host RAM, in MiB. 0 disables, -1 removes the limit.",
            "-khad": "ik_llama.cpp only. Hadamard transform for K cache, useful when experimenting with aggressive KV quantization.",
            "-vhad": "ik_llama.cpp only. Hadamard transform for V cache, useful when experimenting with aggressive KV quantization.",
        }

    def configure_style(self) -> None:
        self.colors = {
            "bg": "#0e1117",
            "panel": "#161b22",
            "panel_soft": "#1c222c",
            "field": "#0f141c",
            "border": "#2b3442",
            "text": "#e6edf3",
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

    def build_ui(self) -> None:
        top_bar = tk.Frame(self.root, bg="#090c10", height=54)
        top_bar.pack(fill="x")
        top_bar.pack_propagate(False)
        tk.Label(top_bar, text=APP_TITLE, bg="#090c10", fg=self.colors["text"], font=("TkDefaultFont", 15, "bold")).pack(side="left", padx=18)

        body = tk.Frame(self.root, bg=self.colors["bg"], padx=16, pady=16)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, minsize=280)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left = tk.Frame(body, bg=self.colors["panel"], highlightbackground=self.colors["border"], highlightthickness=1)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        left.rowconfigure(2, weight=1)
        left.columnconfigure(0, weight=1)

        preset_header = tk.Frame(left, bg=self.colors["panel"])
        preset_header.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 8))
        preset_header.columnconfigure(0, weight=1)
        tk.Label(preset_header, text="Presets", bg=self.colors["panel"], fg=self.colors["text"], font=("TkDefaultFont", 13, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        preset_actions = tk.Frame(left, bg=self.colors["panel"])
        preset_actions.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 10))
        for col in range(4):
            preset_actions.columnconfigure(col, weight=1, uniform="preset_actions")
        add_button = self.make_button(preset_actions, "➕", self.add_preset, width=46, bg=self.colors["add"], hover=self.colors["add_hover"])
        add_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        Tooltip(add_button, "New preset\n\nSave the current launcher settings as a new preset.")
        delete_button = self.make_button(preset_actions, "➖", self.delete_selected_preset, width=46, bg=self.colors["remove"], hover=self.colors["remove_hover"])
        delete_button.grid(row=0, column=1, sticky="ew", padx=(0, 6))
        Tooltip(delete_button, "Delete preset\n\nRemove the selected preset.")
        save_button = self.make_button(preset_actions, "💾", self.save_current_preset, width=46, bg=self.colors["good"], hover=self.colors["good_hover"])
        save_button.grid(row=0, column=2, sticky="ew", padx=(0, 6))
        Tooltip(save_button, "Save preset\n\nUpdate the selected preset, or save the current settings as a preset.")
        import_button = self.make_button(preset_actions, "📋", self.import_command_dialog, width=46, bg=self.colors["import"], hover=self.colors["import_hover"])
        import_button.grid(row=0, column=3, sticky="ew")
        Tooltip(import_button, "Import command\n\nPaste a server command and load recognized arguments into the UI.")

        self.preset_list = tk.Listbox(
            left,
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
        )
        self.preset_list.grid(row=2, column=0, sticky="nsew", padx=14, pady=(0, 14))
        self.preset_list.bind("<<ListboxSelect>>", self.on_preset_selected)

        right = tk.Frame(body, bg=self.colors["bg"])
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(4, weight=1)

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
        Tooltip(inferer_selector, "Inferer\n\nChoose which llama-server-compatible backend profile to use. ik_llama.cpp exposes extra ik-only tuning flags.")
        self.inferer_executable_var = tk.StringVar(value=self.current_inferer().executable)
        self.inferer_executable_var.trace_add("write", lambda *_: self.update_command_preview())
        executable_entry = self.make_entry(model_panel, self.inferer_executable_var)
        executable_entry.grid(row=0, column=2, sticky="ew", padx=(8, 0), ipady=6)
        Tooltip(executable_entry, "Executable\n\nCommand or path to the server binary. For ik_llama.cpp this is often /path/to/ik_llama.cpp/build/bin/llama-server.")
        tk.Label(model_panel, text="Model", bg=self.colors["panel"], fg=self.colors["muted"], font=("TkDefaultFont", 10, "bold")).grid(
            row=1, column=0, sticky="w", padx=(0, 12), pady=(10, 0)
        )
        self.model_var = tk.StringVar()
        self.model_var.trace_add("write", lambda *_: self.on_model_changed())
        self.model_entry = self.make_entry(model_panel, self.model_var)
        self.model_entry.grid(row=1, column=1, columnspan=2, sticky="ew", pady=(10, 0), ipady=6)
        self.make_button(model_panel, "Browse", lambda: self.open_file_picker("model"), width=78).grid(row=1, column=3, padx=(8, 0), pady=(10, 0))
        tk.Label(model_panel, text="MMProj", bg=self.colors["panel"], fg=self.colors["muted"], font=("TkDefaultFont", 10, "bold")).grid(
            row=2, column=0, sticky="w", padx=(0, 12), pady=(10, 0)
        )
        self.mmproj_var = tk.StringVar()
        self.mmproj_var.trace_add("write", lambda *_: self.on_mmproj_changed())
        self.mmproj_entry = self.make_entry(model_panel, self.mmproj_var)
        self.mmproj_entry.grid(row=2, column=1, columnspan=2, sticky="ew", pady=(10, 0), ipady=6)
        self.make_button(model_panel, "Browse", lambda: self.open_file_picker("mmproj"), width=78).grid(row=2, column=3, padx=(8, 0), pady=(10, 0))

        flags_panel = self.make_panel(right, padx=14, pady=12)
        flags_panel.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        flags_panel.columnconfigure(0, weight=1)
        flags_header = tk.Frame(flags_panel, bg=self.colors["panel"])
        flags_header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        flags_header.columnconfigure(0, weight=1)
        tk.Label(flags_header, text="Flags", bg=self.colors["panel"], fg=self.colors["text"], font=("TkDefaultFont", 12, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        add_flag_button = self.make_button(flags_header, "➕", self.add_flag_dialog, width=34, bg=self.colors["add"], hover=self.colors["add_hover"])
        add_flag_button.grid(row=0, column=1, sticky="e", padx=(0, 6))
        Tooltip(add_flag_button, "Add flag\n\nAdd a custom server flag to the UI.")
        clear_flags_button = self.make_button(flags_header, "🧹", self.clear_flags_to_default, width=34, bg=self.colors["panel_soft"])
        clear_flags_button.grid(row=0, column=2, sticky="e")
        Tooltip(clear_flags_button, "Clear flags\n\nUntick every flag, empty all flag values, and clear Extra args.")
        self.flags_frame = tk.Frame(flags_panel, bg=self.colors["panel"])
        self.flags_frame.grid(row=1, column=0, sticky="ew")
        for col in range(3):
            self.flags_frame.columnconfigure(col, weight=1, uniform="flag_columns")
        extra_row = tk.Frame(flags_panel, bg=self.colors["panel"])
        extra_row.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        extra_row.columnconfigure(1, weight=1)
        tk.Label(extra_row, text="Extra args", bg=self.colors["panel"], fg=self.colors["muted"], font=("TkDefaultFont", 10, "bold")).grid(
            row=0, column=0, sticky="w", padx=(0, 12)
        )
        self.extra_args_var = tk.StringVar()
        self.extra_args_var.trace_add("write", lambda *_: self.update_command_preview())
        extra_entry = self.make_entry(extra_row, self.extra_args_var)
        extra_entry.grid(row=0, column=1, sticky="ew", ipady=6)
        Tooltip(
            extra_entry,
            "Extra server arguments\n\nUse this for advanced or less common flags that are not shown in the UI, for example --no-webui, --metrics, --log-file server.log, -cuda graphs=0, or --tensor-split 3,1.",
        )

        controls = tk.Frame(right, bg=self.colors["bg"])
        controls.grid(row=2, column=0, sticky="ew", pady=(12, 10))
        controls.columnconfigure(5, weight=1)
        launch_button = self.make_button(controls, "▶", self.launch, width=52, bg=self.colors["good"], hover=self.colors["good_hover"])
        launch_button.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        Tooltip(launch_button, "Launch\n\nStart the selected inferer with the current settings.")
        stop_button = self.make_button(controls, "■", self.stop_process, width=52, bg=self.colors["danger"], hover=self.colors["danger_hover"])
        stop_button.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        Tooltip(stop_button, "Stop\n\nStop the running inferer process.")
        clear_button = self.make_button(controls, "🧹", self.clear_logs, width=52, bg=self.colors["panel_soft"])
        clear_button.grid(row=0, column=2, sticky="ew", padx=(0, 8))
        Tooltip(clear_button, "Clear output\n\nClear the output log.")
        self.auto_restart_var = tk.BooleanVar(value=False)
        auto_restart = tk.Checkbutton(
            controls,
            text="↻",
            variable=self.auto_restart_var,
            bg=self.colors["bg"],
            fg=self.colors["text"],
            activebackground=self.colors["bg"],
            activeforeground=self.colors["text"],
            selectcolor=self.colors["field"],
            relief="flat",
            bd=0,
            highlightthickness=0,
            font=("TkDefaultFont", 13, "bold"),
        )
        auto_restart.grid(row=0, column=3, sticky="w")
        Tooltip(auto_restart, "Auto-restart\n\nRestart the selected inferer automatically if it crashes. Manual Stop will not restart it.")
        self.command_var = tk.StringVar()
        self.command_label = tk.Label(
            controls,
            textvariable=self.command_var,
            bg=self.colors["bg"],
            fg=self.colors["muted"],
            anchor="w",
            justify="left",
            font=("DejaVu Sans Mono", 9),
            wraplength=760,
        )
        self.command_label.grid(row=1, column=0, columnspan=6, sticky="ew", pady=(8, 0))

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
        status_panel.columnconfigure(2, weight=1)
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
        tk.Label(status_panel, text="VRAM", bg=self.colors["panel"], fg=self.colors["muted"], font=("TkDefaultFont", 10, "bold")).grid(
            row=0, column=2, sticky="e", padx=(0, 10)
        )
        self.vram_label_var = tk.StringVar()
        tk.Label(status_panel, textvariable=self.vram_label_var, bg=self.colors["panel"], fg=self.colors["text"], font=("TkDefaultFont", 10)).grid(
            row=0, column=3, sticky="w"
        )
        self.vram_bar = ttk.Progressbar(status_panel, mode="determinate", maximum=100)
        self.vram_bar.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        self.vram_breakdown_var = tk.StringVar()
        tk.Label(
            status_panel,
            textvariable=self.vram_breakdown_var,
            bg=self.colors["panel"],
            fg=self.colors["muted"],
            anchor="w",
            justify="left",
            font=("DejaVu Sans Mono", 9),
        ).grid(row=2, column=0, columnspan=4, sticky="ew", pady=(8, 0))

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
        )
        selector.columnconfigure(0, weight=1)
        value_label = tk.Label(
            selector,
            textvariable=variable,
            bg=self.colors["field"],
            fg=self.colors["text"],
            anchor="w",
            padx=8,
            pady=6,
            font=("TkDefaultFont", 11),
            cursor="hand2",
        )
        value_label.grid(row=0, column=0, sticky="ew")
        arrow = tk.Label(selector, text="▾", bg=self.colors["field"], fg=self.colors["text"], padx=8, font=("TkDefaultFont", 10), cursor="hand2")
        arrow.grid(row=0, column=1, sticky="e")
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
            text=text,
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
            cursor="hand2",
        )
        button.bind("<Enter>", lambda _event: button.configure(bg=hover_color))
        button.bind("<Leave>", lambda _event: button.configure(bg=base))
        return button

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
        self.render_flags()
        self.update_command_preview()

    def normalize_custom_flag(self, flag: str) -> str:
        return flag.strip()

    def add_or_update_flag(self, flag: str, value: str = "", value_required: bool = True, enabled: bool = True) -> None:
        normalized = self.normalize_custom_flag(flag)
        if not normalized.startswith("-") or normalized == "-":
            raise ValueError("Flag must start with - or --.")
        if any(char.isspace() for char in normalized):
            raise ValueError("Flag name cannot contain spaces.")
        if normalized in {"-m", "--model"}:
            raise ValueError("Use the Model field for the model path.")
        if normalized in {"--mmproj", "-mm"}:
            raise ValueError("Use the MMProj field for the projector path.")
        existing = self.flags.get(normalized)
        if existing:
            existing.value = value
            existing.value_required = value_required
            existing.enabled = enabled
            self.hidden_flags.discard(normalized)
        else:
            label = normalized.lstrip("-") or normalized
            self.flags[normalized] = FlagConfig(label, value, enabled, value_required, custom=True)

    def add_flag_dialog(self) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Add flag")
        dialog.configure(bg=self.colors["bg"])
        dialog.geometry("460x220")
        dialog.minsize(420, 210)
        dialog.transient(self.root)
        dialog.grab_set()

        shell = tk.Frame(dialog, bg=self.colors["bg"], padx=14, pady=14)
        shell.pack(fill="both", expand=True)
        shell.columnconfigure(1, weight=1)

        tk.Label(shell, text="Flag", bg=self.colors["bg"], fg=self.colors["muted"], font=("TkDefaultFont", 10, "bold")).grid(
            row=0, column=0, sticky="w", padx=(0, 12), pady=(0, 10)
        )
        flag_var = tk.StringVar()
        flag_entry = self.make_entry(shell, flag_var)
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

        actions = tk.Frame(shell, bg=self.colors["bg"])
        actions.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(16, 0))
        actions.columnconfigure(0, weight=1)

        def save() -> None:
            try:
                self.add_or_update_flag(flag_var.get(), value_var.get().strip(), needs_value_var.get(), True)
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
        self.render_flags()
        self.recalculate_vram()

    def render_flags(self) -> None:
        for child in self.flags_frame.winfo_children():
            child.destroy()
        self.flag_vars.clear()
        items = [(flag, cfg) for flag, cfg in self.flags.items() if flag != "--mmproj" and self.flag_supported_by_current_inferer(flag)]
        help_text = self.flag_help()
        for idx, (flag, cfg) in enumerate(items):
            row, col = divmod(idx, 3)
            cell = tk.Frame(self.flags_frame, bg=self.colors["panel"])
            cell.grid(row=row, column=col, sticky="ew", padx=(0, 14) if col < 2 else (0, 0), pady=5)
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
                    if flag == "-c":
                        entry_parent = tk.Frame(cell, bg=self.colors["panel"])
                    entry = self.make_entry(entry_parent, value_var)
                    entry.configure(width=18)
                if flag == "-c":
                    ctx_frame = entry_parent
                    ctx_frame.grid(row=0, column=1, sticky="ew")
                    ctx_frame.columnconfigure(2, weight=1)
                    minus = self.make_button(ctx_frame, "➖", lambda: self.step_context_size(-1), width=26, bg=self.colors["panel_soft"])
                    minus.grid(row=0, column=0, sticky="ew", padx=(0, 4))
                    divide = self.make_button(ctx_frame, "/", lambda: self.scale_context_size(0.5), width=26, bg=self.colors["panel_soft"])
                    divide.grid(row=0, column=1, sticky="ew", padx=(0, 4))
                    entry.grid(row=0, column=2, sticky="ew", ipady=6)
                    multiply = self.make_button(ctx_frame, "x", lambda: self.scale_context_size(2), width=26, bg=self.colors["panel_soft"])
                    multiply.grid(row=0, column=3, sticky="ew", padx=(4, 0))
                    plus = self.make_button(ctx_frame, "➕", lambda: self.step_context_size(1), width=26, bg=self.colors["panel_soft"])
                    plus.grid(row=0, column=4, sticky="ew", padx=(4, 0))
                    Tooltip(minus, "Decrease context size\n\nSubtract 1024 tokens.")
                    Tooltip(divide, "Halve context size\n\nDivide the current context size by 2.")
                    Tooltip(multiply, "Double context size\n\nMultiply the current context size by 2.")
                    Tooltip(plus, "Increase context size\n\nAdd 1024 tokens.")
                    Tooltip(entry, f"{flag} value\n\nShown as the final context size in tokens. -/+ change by 1024; / and x halve or double it.\n\n{help_text.get(flag, '')}")
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
            remove = self.make_button(cell, "×", lambda f=flag: self.remove_flag(f), width=22, bg=self.colors["panel_soft"])
            remove.grid(row=0, column=2, sticky="e", padx=(6, 0))
            Tooltip(remove, f"Remove {flag}\n\nHide this flag from the current UI. Add it again with the plus button.")
            self.flag_vars[flag] = values

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
            self.preset_list.insert("end", preset.get("preset_name", "Unnamed"))
        if self.selected_preset:
            for idx, preset in enumerate(self.history.presets):
                if preset.get("preset_name") == self.selected_preset:
                    self.preset_list.selection_set(idx)
                    self.preset_list.activate(idx)
                    break

    def on_preset_selected(self, _event: tk.Event) -> None:
        selection = self.preset_list.curselection()
        if not selection:
            return
        preset = self.history.presets[selection[0]]
        self.load_preset(preset)

    def add_preset(self) -> None:
        self.selected_preset = None
        self.save_current_preset()

    def save_current_preset(self) -> None:
        default = self.selected_preset or (Path(self.model_var.get()).stem if self.model_var.get() else "")
        name = simpledialog.askstring("Save preset", "Preset name:", initialvalue=default, parent=self.root)
        if not name:
            return
        preset = {
            "preset_name": name.strip(),
            "inferer": self.current_inferer_key(),
            "inferer_executable": self.inferer_executable_var.get().strip() if self.inferer_executable_var else self.current_inferer().executable,
            "model_path": self.model_var.get().strip(),
            "mmproj_path": self.flags["--mmproj"].value,
            "extra_args": self.extra_args_var.get().strip() if self.extra_args_var else "",
            "hidden_flags": sorted(self.hidden_flags),
            "flags": {
                flag: {
                    "value": cfg.value,
                    "enabled": cfg.enabled,
                    "value_required": cfg.value_required,
                    "custom": cfg.custom,
                }
                for flag, cfg in self.flags.items()
            },
        }
        self.history.upsert_preset(preset)
        self.selected_preset = name.strip()
        self.render_presets()

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
            if "ik_llama" in executable.lower():
                self.inferer_var.set("ik_llama.cpp")
            elif exe_name == "llama-server" and self.inferer_var and self.inferer_var.get() not in self.inferers:
                self.inferer_var.set("llama.cpp")
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
            "--threads-http": "--threads-http",
            "--alias": "-a",
            "-a": "-a",
            "--timeout": "-to",
            "-to": "-to",
            "--mmproj": "--mmproj",
            "-mm": "--mmproj",
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
                )
        for flag, cfg in self.flags.items():
            saved = saved_flags.get(flag, {})
            cfg.value = str(saved.get("value", cfg.value) or "")
            cfg.enabled = bool(saved.get("enabled", cfg.enabled))
            cfg.value_required = bool(saved.get("value_required", cfg.value_required))
        self.hidden_flags = {str(flag) for flag in preset.get("hidden_flags", []) if str(flag) in self.flags}
        inferer = str(preset.get("inferer") or "llama.cpp")
        if self.inferer_var:
            self.inferer_var.set(inferer if inferer in self.inferers else "llama.cpp")
        if self.inferer_executable_var:
            self.inferer_executable_var.set(str(preset.get("inferer_executable") or self.current_inferer().executable))
        self.model_var.set(str(preset.get("model_path") or saved_flags.get("-m", {}).get("value") or ""))
        self.mmproj_var.set(str(preset.get("mmproj_path") or saved_flags.get("--mmproj", {}).get("value") or ""))
        if self.extra_args_var:
            self.extra_args_var.set(str(preset.get("extra_args") or ""))
        self.render_flags()
        self.render_presets()
        self.recalculate_vram()

    def open_file_picker(self, target: str) -> None:
        start_path = self.model_var.get().strip() if target == "model" else self.flags["--mmproj"].value
        initial_dir = Path(start_path).expanduser().parent if start_path else Path.home()
        if not initial_dir.exists():
            initial_dir = Path.home()
        result = self.choose_gguf_file(initial_dir)
        if result:
            if target == "model":
                self.model_var.set(result)
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
        value = self.model_var.get().strip()
        if value and Path(value).expanduser().exists():
            self.model_meta = parse_gguf(value)
            mmproj = find_mmproj(value)
            if mmproj and not self.flags["--mmproj"].value:
                self.mmproj_var.set(mmproj)
                self.render_flags()
            for warning in self.model_meta.warnings:
                self.append_log("warn", warning)
        else:
            self.model_meta = None
        self.recalculate_vram()

    def on_mmproj_changed(self) -> None:
        value = self.mmproj_var.get().strip()
        self.flags["--mmproj"].value = value
        self.flags["--mmproj"].enabled = bool(value)
        self.recalculate_vram()

    def set_flag_value(self, flag: str, value: str) -> None:
        self.flags[flag].value = value
        if flag == "--mmproj" and hasattr(self, "mmproj_var") and self.mmproj_var.get() != value:
            self.mmproj_var.set(value)
        self.recalculate_vram()

    def set_flag_enabled(self, flag: str, enabled: bool) -> None:
        self.flags[flag].enabled = enabled
        self.recalculate_vram()

    def step_context_size(self, direction: int) -> None:
        current = self.get_int_flag("-c", 0)
        if current <= 0:
            current = 0
        next_value = max(1024, current + (direction * 1024))
        self.set_context_size(next_value)

    def scale_context_size(self, multiplier: float) -> None:
        current = max(1024, self.get_int_flag("-c", 1024))
        next_value = max(1024, int(round((current * multiplier) / 1024)) * 1024)
        self.set_context_size(next_value)

    def set_context_size(self, value: int) -> None:
        self.flags["-c"].value = str(value)
        value_var = self.flag_vars.get("-c", {}).get("value")
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

    def estimate_confidence(self) -> str:
        if not self.model_meta:
            return "no model selected"
        if self.model_meta.n_layers and self.model_meta.n_embd and self.model_meta.n_kv_heads:
            return "full metadata"
        return "file-size only"

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
        kv_bytes = max(KV_BYTES.get(ctk_value, 0.5), KV_BYTES.get(ctv_value, 0.5))
        kv_cache = context * 2 * (n_layers + 1) * (n_embd / max(1, n_kv_heads)) * kv_bytes if n_embd else 0
        mmproj_size = 0.0
        mmproj_path = self.flags["--mmproj"].value
        if self.flags["--mmproj"].enabled and mmproj_path and Path(mmproj_path).expanduser().exists():
            mmproj_size = float(Path(mmproj_path).expanduser().stat().st_size)
        weights = (min(ngl, total_layers) / total_layers) * model_size
        overhead = 1.25 * GB
        estimated = weights + mmproj_size + kv_cache + overhead if model_size else 0.0
        self.estimate = {
            "weights": weights,
            "kv": kv_cache,
            "mmproj": mmproj_size,
            "overhead": overhead,
            "estimated": estimated,
            "ngl": float(ngl),
            "total_layers": float(total_layers),
        }
        self.update_command_preview()
        self.update_vram_ui()

    def update_command_preview(self) -> None:
        try:
            command = self.build_command(validate_model=False)
            self.command_var.set(shlex.join(command))
        except Exception as exc:
            self.command_var.set(f"Command incomplete: {exc}")

    def update_vram_ui(self) -> None:
        process_running = self.process is not None and self.process.poll() is None
        used = self.vram_used if process_running and self.vram_used is not None else self.estimate.get("estimated", 0.0)
        total = self.vram_total
        percent = min(100.0, (used / total) * 100) if used and total else 0.0
        self.vram_bar.configure(value=percent)
        prefix = "Runtime" if process_running and self.vram_used is not None else "Estimated"
        bits = [f"{prefix} {human_bytes(used)}"]
        if total:
            bits.append(f"of {human_bytes(total)}")
        bits.append(f"({self.vram_source})")
        self.vram_label_var.set(" ".join(bits))
        self.update_vram_breakdown()

    def update_vram_breakdown(self) -> None:
        if not hasattr(self, "vram_breakdown_var"):
            return
        total_layers = int(self.estimate.get("total_layers", 0))
        ngl = int(self.estimate.get("ngl", 0))
        confidence = self.estimate_confidence()
        layer_text = self.describe_gpu_layers_for_estimate(total_layers, ngl) if total_layers else "no model layers available"
        parts = [
            f"Weights {human_bytes(self.estimate.get('weights', 0.0))}",
            f"KV {human_bytes(self.estimate.get('kv', 0.0))}",
            f"MMProj {human_bytes(self.estimate.get('mmproj', 0.0))}",
            f"Overhead {human_bytes(self.estimate.get('overhead', 0.0))}",
        ]
        self.vram_breakdown_var.set(
            " + ".join(parts)
            + f" = {human_bytes(self.estimate.get('estimated', 0.0))}\n"
            + f"-ngl: {layer_text} | confidence: {confidence}"
        )

    def build_command(self, validate_model: bool = True) -> list[str]:
        model_path = self.model_var.get().strip()
        if not model_path:
            raise ValueError("model path is required")
        if validate_model and not Path(model_path).expanduser().exists():
            raise ValueError("model path does not exist")
        executable = self.inferer_executable_var.get().strip() if self.inferer_executable_var else self.current_inferer().executable
        if not executable:
            raise ValueError("inferer executable is required")
        try:
            command = shlex.split(executable)
        except ValueError as exc:
            raise ValueError(f"inferer executable is not valid shell-style text: {exc}") from exc
        if not command:
            raise ValueError("inferer executable is required")
        command.extend(["-m", model_path])
        for flag, cfg in self.flags.items():
            if not cfg.enabled:
                continue
            if not self.flag_supported_by_current_inferer(flag):
                continue
            if flag == "--mmproj" and not cfg.value.strip():
                continue
            if cfg.value_required:
                if cfg.value.strip():
                    command.extend([flag, cfg.value.strip()])
            else:
                command.append(flag)
        if self.extra_args_var and self.extra_args_var.get().strip():
            try:
                command.extend(shlex.split(self.extra_args_var.get().strip()))
            except ValueError as exc:
                raise ValueError(f"extra args are not valid shell-style arguments: {exc}") from exc
        return command

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
        if self.process and self.process.poll() is None:
            self.append_log("warn", "Stopping inferer...")
            try:
                if os.name != "nt":
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                else:
                    self.process.terminate()
            except Exception:
                self.process.terminate()
        self.set_status("Stopped")

    def refresh_process_state(self) -> None:
        if self.process and self.process.poll() is not None and self.status_var.get() in {"Launching", "Running"}:
            code = self.process.returncode
            if code == 0 or self.intentional_stop:
                self.set_status("Stopped")
            else:
                self.set_status("Crashed")
                self.append_log("error", f"inferer exited with code {code}")
                if hasattr(self, "auto_restart_var") and self.auto_restart_var.get():
                    try:
                        command = self.build_command()
                    except Exception as exc:
                        self.append_log("error", f"Auto-restart skipped: {exc}")
                    else:
                        self.root.after(800, lambda cmd=command: self.start_process(cmd, add_separator=True))
        self.root.after(600, self.refresh_process_state)

    def refresh_gpu_usage(self) -> None:
        self.vram_used, self.vram_total, self.vram_source = self.gpu.usage()
        self.update_vram_ui()
        self.root.after(2000, self.refresh_gpu_usage)

    def set_status(self, status: str) -> None:
        self.status_var.set(status)
        if hasattr(self, "status_label"):
            colors = {
                "Stopped": self.colors["panel_soft"],
                "Launching": self.colors["warn"],
                "Running": self.colors["good"],
                "Error": self.colors["danger"],
                "Crashed": self.colors["danger"],
            }
            self.status_label.configure(bg=colors.get(status, self.colors["panel_soft"]))

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

    def append_log(self, level: str, text: str) -> None:
        if threading.current_thread() is not threading.main_thread():
            self.log_queue.put((level, text))
            return
        self.update_status_from_log(text)
        self.output.configure(state="normal")
        self.output.insert("end", text + "\n", level)
        self.output.see("end")
        self.output.configure(state="disabled")

    def update_status_from_log(self, text: str) -> None:
        if self.model_loaded:
            return
        if "main: model loaded" in text.lower():
            self.model_loaded = True
            if self.process and self.process.poll() is None:
                self.set_status("Running")

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
            self.stop_process()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    try:
        LauncherApp().run()
    except tk.TclError as exc:
        print(f"{APP_TITLE} could not start: {exc}", file=sys.stderr)
        raise
