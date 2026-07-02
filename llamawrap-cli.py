#!/usr/bin/env python3
"""
Interactive preset browser and CLI for llama-wrap history.json.

Usage:
    llamawrap-cli              Interactive preset browser (default)
    llamawrap-cli create <name> <model-path> [options]
    llamawrap-cli import <name> <command-or-args>
    llamawrap-cli list
    llamawrap-cli show <name>
    llamawrap-cli set <name> <flag> <value>
    llamawrap-cli enable <name> <flag>
    llamawrap-cli disable <name> <flag>
    llamawrap-cli rmflag <name> <flag>
    llamawrap-cli rename <name> <new-name>
    llamawrap-cli run <name>
    llamawrap-cli delete <name>
"""

from __future__ import annotations

import difflib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

try:
    import readline
except ImportError:  # Not available in the standard Windows Python build.
    readline = None

import llamawrap_core as core


def find_history() -> Path:
    """Locate history.json (env override, app dir, cwd, then per-user data dir)."""
    return core.find_history(sys.argv[0])


def load_history(path: Path) -> dict:
    """Shared core loader with CLI-style error exits."""
    try:
        return core.load_history(path)
    except FileNotFoundError:
        print(f"error: {path} not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"error: cannot read {path}: {e}", file=sys.stderr)
        sys.exit(1)


def load_or_init_history(path: Path) -> dict:
    if path.exists():
        return load_history(path)
    return core.load_or_init_history(path)


def save_history(path: Path, data: dict) -> None:
    core.save_history(path, data)


def find_preset(data: dict, name: str) -> dict | None:
    return core.find_preset(data, name)


def error(msg: str) -> None:
    print(f"{red('error:')} {msg}", file=sys.stderr)
    sys.exit(1)


# ───────────────────────────────────────
#  Color / quiet output mode
#
#  Plain ANSI escapes, no dependency. Disabled automatically when stdout
#  is not a terminal or NO_COLOR is set; can also be forced off with
#  --no-color. --quiet trims non-essential informational lines.
# ───────────────────────────────────────

COLOR_ENABLED = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
QUIET = False


def set_color_enabled(enabled: bool) -> None:
    global COLOR_ENABLED
    COLOR_ENABLED = enabled


def set_quiet(enabled: bool) -> None:
    global QUIET
    QUIET = enabled


def _c(text: str, code: str) -> str:
    if not COLOR_ENABLED or not text:
        return text
    return f"\033[{code}m{text}\033[0m"


def bold(text: str) -> str:
    return _c(text, "1")


def dim(text: str) -> str:
    return _c(text, "2")


def red(text: str) -> str:
    return _c(text, "31")


def green(text: str) -> str:
    return _c(text, "32")


def yellow(text: str) -> str:
    return _c(text, "33")


def cyan(text: str) -> str:
    return _c(text, "36")


_STATUS_COLOR = {"PASS": green, "FAIL": red, "WARN": yellow, "SKIP": dim}
_STATUS_LINE_RE = re.compile(r"^\[(PASS|FAIL|WARN|SKIP)\]\s*([^:]+):?\s*(.*)$")
_RESULT_LINE_RE = re.compile(r"^(.+ result): (PASS|FAIL)$")
_HEADER_LINE_RE = re.compile(r"^(Doctor|Probe|Bench|Stress): (.+)$")


def print_diagnostic_lines(lines: list[str]) -> None:
    """Print Doctor/Probe/Bench/Stress report lines as an aligned, color-coded list.

    Falls back to printing the line unmodified when it doesn't match a
    recognized status/result/header shape, so unusual output is never lost.
    """
    parsed = [_STATUS_LINE_RE.match(line.strip()) for line in lines]
    label_width = min(max((len(m.group(2).strip()) for m in parsed if m), default=0), 32)

    for line, m in zip(lines, parsed):
        stripped = line.strip()
        if m:
            status, label, detail = m.group(1), m.group(2).strip(), m.group(3).strip()
            if QUIET and status in ("PASS", "SKIP"):
                continue
            color = _STATUS_COLOR.get(status, lambda t: t)
            tag = color(bold(f"[{status}]"))
            if detail:
                print(f"  {tag} {label:<{label_width}}  {detail}")
            else:
                print(f"  {tag} {label}")
            continue
        rm = _RESULT_LINE_RE.match(stripped)
        if rm:
            label, status = rm.group(1), rm.group(2)
            color = _STATUS_COLOR.get(status, lambda t: t)
            print(f"{label}: {color(bold(status))}")
            continue
        hm = _HEADER_LINE_RE.match(stripped)
        if hm:
            if not QUIET:
                print(bold(stripped))
            continue
        if QUIET and stripped.startswith("saved:"):
            continue
        print(line)


def resolve_preset(data: dict, name: str) -> dict:
    """Find a preset by exact name, unique substring, or suggest close matches.

    Exits via error() (no return) when nothing usable is found, matching the
    existing call-site pattern of `preset = find_preset(...); if not preset: error(...)`.
    """
    preset = find_preset(data, name)
    if preset:
        return preset

    presets = data.get("presets", [])
    lname = name.lower()
    substring_matches = [p for p in presets if lname in p.get("preset_name", "").lower()]
    if len(substring_matches) == 1:
        match = substring_matches[0]
        if not QUIET:
            print(dim(f"  (matched '{name}' -> '{match.get('preset_name')}')"))
        return match
    if len(substring_matches) > 1:
        names = ", ".join(repr(p.get("preset_name", "")) for p in substring_matches)
        error(f"preset '{name}' is ambiguous, matches: {names}")

    all_names = [p.get("preset_name", "") for p in presets]
    suggestions = difflib.get_close_matches(name, all_names, n=3, cutoff=0.5)
    if suggestions:
        hint = ", ".join(repr(s) for s in suggestions)
        error(f"preset '{name}' not found. did you mean: {hint}?")
    error(f"preset '{name}' not found")


# ───────────────────────────────────────
#  Commands
# ───────────────────────────────────────

def default_cli_flags() -> dict:
    return core.default_cli_flags()


def parse_create_args(args: list[str]) -> dict:
    if len(args) < 3:
        error("usage: llamawrap-cli create <preset-name> <model-path> [options]")

    options = {
        "name": args[1],
        "model_path": args[2],
        "inferer": "llama.cpp",
        "executable": "llama-server",
        "mmproj_path": "",
        "draft_model_path": "",
        "extra_args": "",
        "sets": [],
        "toggles": [],
        "force": False,
    }
    i = 3
    while i < len(args):
        arg = args[i]
        if arg == "--force":
            options["force"] = True
            i += 1
        elif arg in ("--inferer", "--executable", "--mmproj", "--draft-model", "--extra-args"):
            if i + 1 >= len(args):
                error(f"usage: {arg} requires a value")
            key = {
                "--inferer": "inferer",
                "--executable": "executable",
                "--mmproj": "mmproj_path",
                "--draft-model": "draft_model_path",
                "--extra-args": "extra_args",
            }[arg]
            options[key] = args[i + 1]
            i += 2
        elif arg == "--set":
            if i + 2 >= len(args):
                error("usage: --set requires <flag> <value>")
            options["sets"].append((args[i + 1], args[i + 2]))
            i += 3
        elif arg == "--toggle":
            if i + 1 >= len(args):
                error("usage: --toggle requires <flag>")
            options["toggles"].append(args[i + 1])
            i += 2
        else:
            error(f"unknown create option: {arg}")
    return options


def cmd_create(data: dict, path: Path, args: list[str]) -> None:
    options = parse_create_args(args)
    name = options["name"].strip()
    model_path = options["model_path"].strip()
    if not name:
        error("preset name cannot be empty")
    if not model_path:
        error("model path cannot be empty")
    if find_preset(data, name) and not options["force"]:
        error(f"preset '{name}' already exists (use --force to replace it)")

    flags = default_cli_flags()
    mmproj_path = options["mmproj_path"].strip()
    if mmproj_path:
        flags["--mmproj"]["value"] = mmproj_path
        flags["--mmproj"]["enabled"] = True

    for flag, value in options["sets"]:
        flags[flag] = {
            "value": value,
            "enabled": True,
            "value_required": True,
            "custom": flag not in flags,
            "step_mode": flags.get(flag, {}).get("step_mode", ""),
        }
    for flag in options["toggles"]:
        flags[flag] = {
            "value": flags.get(flag, {}).get("value", ""),
            "enabled": True,
            "value_required": False,
            "custom": flag not in flags,
            "step_mode": flags.get(flag, {}).get("step_mode", ""),
        }

    preset = {
        "format_version": 1,
        "preset_name": name,
        "inferer": options["inferer"].strip() or "llama.cpp",
        "inferer_executable": options["executable"].strip() or "llama-server",
        "model_path": model_path,
        "mmproj_path": mmproj_path,
        "draft_model_path": options["draft_model_path"].strip(),
        "extra_args": options["extra_args"].strip(),
        "hidden_flags": [],
        "flags": flags,
    }
    presets = [p for p in data.get("presets", []) if p.get("preset_name") != name]
    presets.append(preset)
    data["presets"] = sorted(presets, key=lambda p: p.get("preset_name", "").lower())
    save_history(path, data)
    print(f"  created preset '{name}'")


def preset_from_command(name: str, command_text: str) -> tuple[dict, int, list[str]]:
    return core.preset_from_command(name, command_text)


def cmd_import(data: dict, path: Path, name: str, command_text: str, *, force: bool = False) -> None:
    name = name.strip()
    if not name:
        error("preset name cannot be empty")
    if find_preset(data, name) and not force:
        error(f"preset '{name}' already exists (use --force to replace it)")
    try:
        preset, changed, skipped = preset_from_command(name, command_text)
    except ValueError as exc:
        error(str(exc))
    presets = [p for p in data.get("presets", []) if p.get("preset_name") != name]
    presets.append(preset)
    data["presets"] = sorted(presets, key=lambda p: p.get("preset_name", "").lower())
    save_history(path, data)
    summary = f"  imported preset '{name}' ({changed} setting{'s' if changed != 1 else ''})"
    if skipped:
        summary += f"; skipped: {', '.join(skipped[:8])}"
        if len(skipped) > 8:
            summary += f", +{len(skipped) - 8} more"
    print(summary)


def _path_matches(text: str, *, files_only: bool = False, suffixes: tuple[str, ...] = ()) -> list[str]:
    expanded = os.path.expanduser(text)
    if not expanded:
        expanded = "."
    directory, prefix = os.path.split(expanded)
    if not directory:
        directory = "."
    try:
        entries = os.listdir(directory)
    except OSError:
        return []

    matches = []
    for entry in entries:
        if not entry.startswith(prefix):
            continue
        full = os.path.join(directory, entry)
        if os.path.isdir(full):
            candidate = os.path.join(directory, entry) + os.sep
        else:
            if files_only and suffixes and not entry.lower().endswith(suffixes):
                continue
            candidate = os.path.join(directory, entry)
        if text.startswith("~"):
            home = os.path.expanduser("~")
            if candidate == home:
                candidate = "~"
            elif candidate.startswith(home + os.sep):
                candidate = "~" + candidate[len(home):]
        elif not os.path.isabs(text) and candidate.startswith("." + os.sep):
            candidate = candidate[2:]
        matches.append(candidate)
    return sorted(matches)


def input_path(prompt: str, *, optional: bool = False, suffixes: tuple[str, ...] = ()) -> str:
    if readline is None:
        return input(prompt).strip()

    old_completer = readline.get_completer()
    old_delims = readline.get_completer_delims()

    def completer(text: str, state: int) -> str | None:
        matches = _path_matches(text, files_only=bool(suffixes), suffixes=suffixes)
        return matches[state] if state < len(matches) else None

    readline.set_completer(completer)
    readline.set_completer_delims(" \t\n")
    readline.parse_and_bind("tab: complete")
    try:
        return input(prompt).strip()
    finally:
        readline.set_completer(old_completer)
        readline.set_completer_delims(old_delims)


def select_path(
    title: str,
    *,
    start_dir: str = ".",
    optional: bool = False,
    suffixes: tuple[str, ...] = (),
) -> str:
    current = Path(os.path.expanduser(start_dir)).resolve()
    while True:
        print(f"\n── {title} ──")
        print(f"  {current}")
        print("  0. ..")
        try:
            entries = sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError as e:
            print(f"  cannot read directory: {e}")
            current = current.parent
            continue

        visible: list[Path] = []
        for entry in entries:
            if entry.name.startswith("."):
                continue
            if entry.is_dir() or not suffixes or entry.name.lower().endswith(suffixes):
                visible.append(entry)

        for idx, entry in enumerate(visible, 1):
            marker = "/" if entry.is_dir() else ""
            print(f"  {idx:>2}. {entry.name}{marker}")

        commands = "number to open/select, p <path> to paste path, q to cancel"
        if optional:
            commands += ", blank to skip"
        print(f"\n{commands}")
        try:
            choice = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return ""

        if optional and not choice:
            return ""
        if choice.lower() == "q":
            return ""
        if choice == "0":
            current = current.parent
            continue
        if choice.startswith("p "):
            pasted = choice[2:].strip()
            if pasted:
                return pasted
            continue
        try:
            idx = int(choice) - 1
        except ValueError:
            print("  enter a number, p <path>, or q")
            continue
        if idx < 0 or idx >= len(visible):
            print("  invalid number")
            continue

        selected = visible[idx]
        if selected.is_dir():
            current = selected
            continue
        return str(selected)


def prompt_create_preset(data: dict, history_path: Path) -> dict | None:
    print("\n── Create preset ──")
    try:
        name = input("Preset name: ").strip()
        if not name:
            print("  cancelled")
            return None
        model_path = select_path("Select model", suffixes=(".gguf",))
        if not model_path:
            print("  cancelled")
            return None
        executable = input_path("Executable [llama-server]: ")
        mmproj_path = select_path("Select MMProj (optional)", optional=True, suffixes=(".gguf",))
        draft_model_path = select_path("Select draft model (optional)", optional=True, suffixes=(".gguf",))
        extra_args = input("Extra args [optional]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  cancelled")
        return None

    create_args = ["create", name, model_path]
    if executable:
        create_args.extend(["--executable", executable])
    if mmproj_path:
        create_args.extend(["--mmproj", mmproj_path])
    if draft_model_path:
        create_args.extend(["--draft-model", draft_model_path])
    if extra_args:
        create_args.extend(["--extra-args", extra_args])

    print("\nAdd initial flags as '<flag> <value>', 'toggle <flag>', or blank when done.")
    while True:
        try:
            line = input("flag> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        try:
            parts = shlex.split(line)
        except ValueError as e:
            print(f"  invalid input: {e}")
            continue
        if len(parts) == 2 and parts[0] == "toggle":
            create_args.extend(["--toggle", parts[1]])
        elif len(parts) >= 2:
            create_args.extend(["--set", parts[0], " ".join(parts[1:])])
        else:
            print("  enter '<flag> <value>' or 'toggle <flag>'")

    try:
        cmd_create(data, history_path, create_args)
    except SystemExit:
        return None
    reloaded = load_history(history_path)
    data.clear()
    data.update(reloaded)
    return find_preset(data, name)


def prompt_import_preset(data: dict, history_path: Path) -> dict | None:
    print("\n── Import command ──")
    try:
        name = input("Preset name: ").strip()
        if not name:
            print("  cancelled")
            return None
        print("Paste launch command or args. End with a blank line.")
        lines: list[str] = []
        while True:
            line = input("> ")
            if not line.strip():
                break
            lines.append(line)
    except (EOFError, KeyboardInterrupt):
        print("\n  cancelled")
        return None

    command_text = " ".join(lines).strip()
    if not command_text:
        print("  cancelled")
        return None
    try:
        cmd_import(data, history_path, name, command_text)
    except SystemExit:
        return None
    reloaded = load_history(history_path)
    data.clear()
    data.update(reloaded)
    return find_preset(data, name)


def cmd_list(data: dict) -> None:
    presets = data.get("presets", [])
    if not presets:
        print("No presets.")
        return
    width = max(len(p.get("preset_name", "")) for p in presets)
    for p in presets:
        name = p.get("preset_name", "?")
        model = Path(p.get("model_path", "")).name or "(no model)"
        print(f"  {cyan(f'{name:<{width}}')}  {dim(model)}")


def cmd_show(data: dict, name: str) -> None:
    preset = resolve_preset(data, name)

    print(f"Preset: {bold(preset['preset_name'])}")
    print(f"  Inferer:         {preset.get('inferer', 'llama.cpp')}")
    print(f"  Executable:      {preset.get('inferer_executable', 'llama-server')}")
    print(f"  Model:           {preset.get('model_path', '')}")
    mmproj = preset.get("mmproj_path", "") or ""
    if mmproj:
        print(f"  MMProj:          {mmproj}")
    draft = preset.get("draft_model_path", "") or ""
    if draft:
        print(f"  Draft model:     {draft}")
    extra = preset.get("extra_args", "") or ""
    if extra:
        print(f"  Extra args:      {extra}")
    hidden = preset.get("hidden_flags", [])
    if hidden:
        print(f"  Hidden flags:    {', '.join(hidden)}")
    stats = preset.get("session_stats", {})
    if stats:
        parts = []
        if "avg_ttft_ms" in stats:
            parts.append(f"{stats['avg_ttft_ms']}ms TTFT")
        if "avg_tok_s" in stats:
            parts.append(f"{stats['avg_tok_s']} tok/s")
        if stats.get("auto_restarts", 0):
            parts.append(f"{stats['auto_restarts']} restarts")
        print(f"  Last session:    {' | '.join(parts)}")
    print()
    flags = preset.get("flags", {})
    enabled_flags = {f: c for f, c in flags.items() if isinstance(c, dict) and c.get("enabled", False)}
    if not enabled_flags:
        print("  (no enabled flags)")
    else:
        max_flag_len = max(len(f) for f in enabled_flags)
        for fname, cfg in sorted(enabled_flags.items()):
            if not isinstance(cfg, dict):
                continue
            value = cfg.get("value", "")
            if cfg.get("value_required", True):
                print(f"    [+] {fname:<{max_flag_len}}  {value}")
            else:
                print(f"    [+] {fname:<{max_flag_len}}  (toggle)")


def cmd_set(data: dict, path: Path, name: str, flag: str, value: str) -> None:
    preset = resolve_preset(data, name)
    flags = preset.setdefault("flags", {})
    if flag not in flags:
        flags[flag] = {
            "value": value,
            "enabled": True,
            "value_required": True,
            "custom": True,
            "step_mode": "",
        }
    else:
        flags[flag]["value"] = value
        flags[flag]["enabled"] = True
    save_history(path, data)
    print(f"  set {flag}={value} (enabled)")


def cmd_enable(data: dict, path: Path, name: str, flag: str) -> None:
    preset = resolve_preset(data, name)
    flags = preset.setdefault("flags", {})
    if flag not in flags:
        flags[flag] = {
            "value": "",
            "enabled": True,
            "value_required": False,
            "custom": True,
            "step_mode": "",
        }
    else:
        flags[flag]["enabled"] = True
    # Remove from hidden_flags if present
    hidden = preset.get("hidden_flags", [])
    if flag in hidden:
        hidden.remove(flag)
        preset["hidden_flags"] = hidden
    save_history(path, data)
    print(f"  enabled {flag}")


def cmd_disable(data: dict, path: Path, name: str, flag: str) -> None:
    preset = resolve_preset(data, name)
    flags = preset.setdefault("flags", {})
    if flag in flags:
        flags[flag]["enabled"] = False
    save_history(path, data)
    print(f"  disabled {flag}")


def cmd_rmflag(data: dict, path: Path, name: str, flag: str) -> None:
    preset = resolve_preset(data, name)
    flags = preset.get("flags", {})
    if flag not in flags:
        error(f"flag '{flag}' not in preset '{name}'")
    del flags[flag]
    hidden = preset.get("hidden_flags", [])
    if flag not in hidden:
        hidden.append(flag)
        preset["hidden_flags"] = hidden
    save_history(path, data)
    print(f"  removed {flag}")


def cmd_rename(data: dict, path: Path, name: str, new_name: str) -> None:
    preset = resolve_preset(data, name)
    if find_preset(data, new_name):
        error(f"preset '{new_name}' already exists")
    preset["preset_name"] = new_name
    save_history(path, data)
    print(f"  renamed '{name}' -> '{new_name}'")


def cmd_delete(data: dict, path: Path, name: str) -> None:
    preset = resolve_preset(data, name)
    actual_name = preset.get("preset_name", name)
    presets = data.get("presets", [])
    for i, p in enumerate(presets):
        if p is preset:
            presets.pop(i)
            save_history(path, data)
            print(f"  deleted '{actual_name}'")
            return
    error(f"preset '{name}' not found")


def build_command_from_preset(preset: dict) -> list[str]:
    """Build the server command list from a preset dict (same logic as the GUI)."""
    try:
        return core.build_command_from_preset(preset)
    except ValueError as exc:
        error(str(exc))


def _fetch_metrics(port: int, host: str = "127.0.0.1", timeout: float = 3.0) -> dict[str, float]:
    """Fetch /metrics from the running server and extract cumulative stats."""
    result: dict[str, float] = {}
    url = f"http://{host}:{port}/metrics"
    try:
        resp = urllib.request.urlopen(url, timeout=timeout)
        text = resp.read().decode("utf-8")
    except Exception:
        return result
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        if "{" in line:
            name = line.split("{")[0]
            # Extract value (last field after space)
            val_str = line.rsplit(None, 1)[-1]
        else:
            name = line.split()[0]
            val_str = line.rsplit(None, 1)[-1]
        try:
            val = float(val_str)
        except ValueError:
            continue
        # Store by metric name + type label if present
        # e.g. llama_eval_time_ms{type="generation"} -> key = "llama_eval_time_ms\tgeneration"
        type_match = re.search(r'type="([^"]+)"', line)
        key = f"{name}\t{type_match.group(1)}" if type_match else name
        result[key] = val
    return result


def run_process(
    command: list[str],
    auto: bool = False,
    preset_name: str | None = None,
    history_path: Path | None = None,
    port: int | None = None,
) -> None:
    """Launch a process, stream output, and optionally auto-restart on crash.\n
    If *preset_name* and *history_path* are given, session stats are parsed from\n    server log output and /metrics endpoint, then saved after the process stops.\n    """
    ttft_ms: list[float] = []
    gen_tokens = 0
    gen_time_ms = 0.0
    restart_count = 0

    def parse_stats(line: str) -> None:
        nonlocal gen_tokens, gen_time_ms
        # TTFT: prompt eval (took|time =) X ms / Y tokens
        m = re.search(
            r"prompt\s+eval\s+(?:took|time\s*=\s*)\s*([0-9]+(?:\.[0-9]+)?)\s*ms\s*/\s*(\d+)\s*tokens",
            line, re.IGNORECASE,
        )
        if m:
            ttft_ms.append(float(m.group(1)))
        # Gen throughput: eval (took|time =) X ms / Y tokens (not preceded by "prompt")
        m = re.search(
            r"(?<!prompt\s)eval\s+(?:took|time\s*=\s*)\s*([0-9]+(?:\.[0-9]+)?)\s*ms\s*/\s*(\d+)\s*tokens",
            line, re.IGNORECASE,
        )
        if m:
            gen_tokens += int(m.group(2))
            gen_time_ms += float(m.group(1))

    while True:
        print(f"$ {shlex.join(command)}\n")
        try:
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                start_new_session=(os.name != "nt"),
            )
        except Exception as e:
            error(f"launch failed: {e}")

        # Stream output and parse stats
        def stream(pipe, label: str) -> None:
            for line in iter(pipe.readline, ""):
                print(f"[{label}] {line}", end="")
                parse_stats(line)
            pipe.close()

        import threading
        t1 = threading.Thread(target=stream, args=(proc.stdout, "out"), daemon=True)
        t2 = threading.Thread(target=stream, args=(proc.stderr, "err"), daemon=True)
        t1.start()
        t2.start()

        try:
            proc.wait()
        except KeyboardInterrupt:
            print("\n  stopping...")
            # Fetch metrics BEFORE killing the server
            metrics = _fetch_metrics(port) if port else {}
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            print("  stopped.")
            _save_cli_session_stats(history_path, preset_name, ttft_ms, gen_tokens, gen_time_ms, metrics, restart_count)
            return

        returncode = proc.poll()
        metrics = _fetch_metrics(port) if port else {}
        _save_cli_session_stats(history_path, preset_name, ttft_ms, gen_tokens, gen_time_ms, metrics, restart_count)
        if returncode != 0 and auto:
            restart_count += 1
            print(f"\n  process exited with code {returncode}, restart #{restart_count} in 2s... (Ctrl+C to stop)\n")
            time.sleep(2)
        else:
            return


def _save_cli_session_stats(
    history_path: Path | None,
    preset_name: str | None,
    ttft_ms: list[float],
    gen_tokens: int,
    gen_time_ms: float,
    metrics: dict[str, float] | None = None,
    restart_count: int = 0,
) -> None:
    """Save accumulated session stats to the preset in history.json (CLI).\n
    Uses /metrics data (more accurate) if available, otherwise falls back\n    to log-parsed stats."""
    if not history_path or not preset_name:
        return
    avg_ttft = 0.0
    avg_tok_s = 0.0

    # Prefer /metrics data if available (cumulative counters are accurate)
    if metrics:
        gen_tok = metrics.get("llama_eval_tokens\tgeneration", 0.0) or metrics.get("llama_eval_tokens_total\tgeneration", 0.0)
        gen_t_ms = metrics.get("llama_eval_time_ms\tgeneration", 0.0) or metrics.get("llama_eval_time_ms_total\tgeneration", 0.0)
        prompt_t_ms = metrics.get("llama_eval_time_ms\tprompt", 0.0) or metrics.get("llama_eval_time_ms_total\tprompt", 0.0)
        prompt_cnt = metrics.get("llama_prompt_eval_count", 0.0) or metrics.get("llama_prompt_eval_count_total", 0.0)
        # TTFT from cumulative: total prompt time / count
        if prompt_cnt > 0 and prompt_t_ms > 0:
            avg_ttft = prompt_t_ms / prompt_cnt
        # tok/s from cumulative: total gen tokens / total gen time
        if gen_t_ms > 0:
            avg_tok_s = (gen_tok / gen_t_ms) * 1000.0
        # Also try gauge TTFT if cumulative didn't give us anything
        if avg_ttft == 0.0:
            gauge_ttft = metrics.get("llama_ttft_ms", 0.0)
            if gauge_ttft > 0:
                avg_ttft = gauge_ttft

    # Fall back to log-parsed stats if metrics didn't have useful data
    if avg_ttft == 0.0 and avg_tok_s == 0.0:
        if ttft_ms:
            avg_ttft = sum(ttft_ms) / len(ttft_ms)
        if gen_tokens > 0 and gen_time_ms > 0:
            avg_tok_s = (gen_tokens / gen_time_ms) * 1000.0

    # Always save even if zero — marks that this preset has been run
    try:
        with open(history_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return
    for p in data.get("presets", []):
        if p.get("preset_name") == preset_name:
            p["session_stats"] = {
                "avg_ttft_ms": round(avg_ttft, 1),
                "avg_tok_s": round(avg_tok_s, 2),
                "auto_restarts": restart_count,
            }
            break
    try:
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def kill_process_on_port(port: int | None, expected_name: str = "") -> None:
    """Free the launch port without silently killing unrelated processes.

    Processes matching the inferer executable name are stopped automatically.
    Anything else prompts on a terminal, or aborts when not interactive."""
    if not port:
        return
    occupants = core.processes_on_port(port)
    if not occupants:
        return
    for pid, name in occupants:
        matches = expected_name and name != "unknown" and name.startswith(expected_name)
        if not matches:
            if sys.stdin.isatty():
                answer = input(f"  port {port} is in use by '{name}' (pid {pid}). Stop it? [y/N] ").strip().lower()
                if answer not in ("y", "yes"):
                    error(f"port {port} is in use by '{name}' (pid {pid}); stop it or change --port")
            else:
                error(f"port {port} is in use by '{name}' (pid {pid}); stop it or change --port")
        print(f"  stopping process on port {port}: {name} (pid {pid})")
        core.terminate_pid(pid)
    deadline = time.time() + 3
    while time.time() < deadline and core.processes_on_port(port):
        time.sleep(0.2)
    for pid, name in core.processes_on_port(port):
        print(f"  {name} (pid {pid}) did not exit; force stopping")
        core.terminate_pid(pid, force=True)
        time.sleep(0.3)


def get_port_from_preset(preset: dict) -> int | None:
    flags = preset.get("flags", {})
    if "--port" in flags and isinstance(flags["--port"], dict):
        try:
            return int(str(flags["--port"].get("value", "8080")).strip())
        except (ValueError, TypeError):
            return 8080
    return None


def cmd_run(data: dict, name: str, auto: bool = False, history_path: Path | None = None) -> None:
    preset = resolve_preset(data, name)

    command = build_command_from_preset(preset)

    # Check executable exists
    executable = command[0]
    exists = (
        Path(executable).expanduser().exists()
        if any(sep in executable for sep in ("/", "\\"))
        else bool(shutil.which(executable))
    )
    if not exists:
        error(f"executable '{executable}' not found")

    port = get_port_from_preset(preset)
    kill_process_on_port(port, expected_name=Path(executable).name)
    run_process(command, auto=auto, preset_name=name, history_path=history_path, port=port)


def _require_preset(data: dict, name: str) -> dict:
    preset = resolve_preset(data, name)
    return preset


def cmd_doctor(data: dict, name: str) -> None:
    preset = _require_preset(data, name)
    ok, lines = core.doctor_report(preset)
    print_diagnostic_lines(lines)
    if not ok:
        sys.exit(1)


def cmd_probe(data: dict, name: str) -> None:
    preset = _require_preset(data, name)
    ok, lines = core.probe_report(preset)
    print_diagnostic_lines(lines)
    if not ok:
        sys.exit(1)


def cmd_bench(data: dict, args: list[str]) -> None:
    if len(args) < 2:
        error("usage: llamawrap-cli bench <preset-name> [--csv] [--out-dir <dir>]")
    name = args[1]
    csv_out = False
    out_dir: Path | None = None
    idx = 2
    while idx < len(args):
        arg = args[idx]
        if arg == "--csv":
            csv_out = True
            idx += 1
        elif arg == "--out-dir":
            if idx + 1 >= len(args):
                error("usage: --out-dir requires a directory")
            out_dir = Path(args[idx + 1]).expanduser()
            idx += 2
        else:
            error(f"unknown bench option: {arg}")
    preset = _require_preset(data, name)
    try:
        row, _paths, _lines = core.run_benchmark(
            preset, out_dir=out_dir, csv_out=csv_out,
            on_line=lambda line: print_diagnostic_lines([line]),
        )
    except KeyboardInterrupt:
        print("\nbench cancelled")
        sys.exit(130)
    if row.get("status") != "pass":
        sys.exit(1)


def cmd_stress(data: dict, name: str) -> None:
    preset = _require_preset(data, name)
    try:
        ok, _lines = core.context_stress_report(
            preset,
            on_line=lambda line: print_diagnostic_lines([line]),
        )
    except KeyboardInterrupt:
        print("\nstress cancelled")
        sys.exit(130)
    if not ok:
        sys.exit(1)


def cmd_export_presets(data: dict, args: list[str]) -> None:
    if "--out" not in args:
        error("usage: llamawrap-cli export-presets --out <file> [--portable] [preset-name...]")
    out_idx = args.index("--out")
    if out_idx + 1 >= len(args):
        error("usage: --out requires a file")
    out_file = Path(args[out_idx + 1]).expanduser()
    portable = "--portable" in args
    names = [
        arg for idx, arg in enumerate(args[1:])
        if arg not in {"--out", "--portable"} and idx + 1 != out_idx + 1
    ]
    try:
        warnings = core.export_presets(data, out_file, names=names, portable=portable)
    except ValueError as exc:
        error(str(exc))
    print(f"exported {len(names) if names else len(data.get('presets', []))} preset(s) to {out_file}")
    for warning in warnings:
        print(f"warning: {warning}")


def cmd_import_presets(data: dict, path: Path, args: list[str]) -> None:
    if len(args) < 2:
        error("usage: llamawrap-cli import-presets <file> [--force]")
    in_file = Path(args[1]).expanduser()
    force = "--force" in args
    unknown = [arg for arg in args[2:] if arg != "--force"]
    if unknown:
        error(f"unknown import-presets option: {unknown[0]}")
    try:
        imported, skipped = core.import_presets(data, in_file, force=force)
    except Exception as exc:
        error(str(exc))
    save_history(path, data)
    print(f"imported {imported} preset(s) from {in_file}")
    if skipped:
        print(f"skipped existing preset(s): {', '.join(skipped)}")


# ───────────────────────────────────────
#  Main
# ───────────────────────────────────────

HELP_TEXT = """Commands:

  list                        List all presets.
  create  <name> <model>      Create a preset from a model path.
  import  <name> <command>    Import a pasted launch command as a preset.
  doctor  <name>              Check executable, paths, port, and API endpoints.
  probe   <name>              Send one OpenAI-compatible chat request.
  bench   <name>              Run repeatable streamed benchmarks and save results.
  stress  <name>              Run the same context stress suite as the GUI.
  export-presets --out <file> Export presets to a portable JSON bundle.
  import-presets <file>       Import presets from an exported JSON bundle.
  show    <name>              Show preset details, flags, and values.
  set     <name> <flag> <val> Set or add a flag value. Creates the flag if missing, enables it.
  enable  <name> <flag>       Enable (tick) a flag. Creates as toggle if missing.
  disable <name> <flag>       Disable (untick) a flag.
  rmflag  <name> <flag>       Remove a flag from the preset (adds it to hidden_flags).
  rename  <name> <new>        Rename a preset.
  delete  <name>              Delete a preset.
  run     <name> [--auto]     Build and launch the server command from a preset.
                              --auto restarts the process if it crashes.
  help    [command]           Show this help or details for a specific command.

Global flags (any command, any position):
  --no-color                  Disable ANSI color even on a terminal.
  --quiet, -q                 Trim PASS/SKIP lines and headers from diagnostic output.

Color is auto-disabled when stdout isn't a terminal or NO_COLOR is set.
"""

HELP_DETAIL = {
    "list": "list\n    List all saved presets with their model file names.",
    "create": "create <preset-name> <model-path> [options]\n    Create a new preset. Creates history.json if it does not exist.\n\n    Options:\n      --executable <path-or-command>   Server executable (default: llama-server)\n      --inferer <name>                 Inferer label (default: llama.cpp)\n      --mmproj <path>                  MMProj model path and enabled --mmproj flag\n      --draft-model <path>             Draft/speculative model path\n      --extra-args <args>              Extra server args, quoted as one value\n      --set <flag> <value>             Set and enable a valued flag; repeatable\n      --toggle <flag>                  Enable a toggle flag; repeatable\n      --force                          Replace an existing preset with same name\n\n    Example:\n      llamawrap-cli create \"My Model\" /models/model.gguf --set -ngl all --set -c 32768 --set --port 8080",
    "import": "import <preset-name> <command-or-args...> [--force]\n    Import a llama-server command or launch args as a preset. Recognized\n    flags are stored as normal preset fields; unknown flags are preserved.\n\n    Examples:\n      llamawrap-cli import \"My Model\" llama-server -m /models/model.gguf -ngl all -c 32768\n      llamawrap-cli import \"Args Only\" -m /models/model.gguf --port 8080",
    "doctor": "doctor <preset-name>\n    Check the preset executable, model paths, configured host/port,\n    /health, /v1/models, and /v1/chat/completions.",
    "probe": "probe <preset-name>\n    Send one small OpenAI-compatible /v1/chat/completions request to the\n    configured running endpoint and print pass/fail details.",
    "bench": "bench <preset-name> [--csv] [--out-dir <dir>]\n    Run a warmup plus several streamed benchmark iterations against the\n    configured running endpoint, reporting median TTFT, generation tok/s,\n    and prefill tok/s when the server provides timings. JSON results are\n    saved under the llama-wrap data directory by default (a portable\n    .llama-wrap folder next to the app is used when present).\n    --csv saves a CSV copy too.",
    "stress": "stress <preset-name>\n    Run the same percentage-based context stress suite as the GUI Stress\n    button: runtime context detection, fill/decode stages, sustained synthetic\n    coding-agent turns, boundary probes, error classification, and a practical\n    working-limit summary.",
    "export-presets": "export-presets --out <file> [--portable] [preset-name...]\n    Export all presets, or selected preset names, to a JSON bundle.\n    Absolute paths are preserved and reported as portability warnings.",
    "import-presets": "import-presets <file> [--force]\n    Import presets from an export-presets JSON bundle. Existing preset names\n    are skipped unless --force is supplied. This does not change the older\n    import <name> <command> command.",
    "show": "show <preset-name>\n    Display the preset's inferer, executable, model paths, all flags,\n    their enabled/disabled status, and current values.",
    "set": "set <preset-name> <flag> <value>\n    Set a flag's value and enable it. If the flag doesn't exist in the\n    preset it is added automatically. Example:\n      llamawrap-cli set \"My Model\" --port 8080",
    "enable": "enable <preset-name> <flag>\n    Enable (check/tick) a flag so it is included when building the\n    server command. Creates the flag as a toggle (no value) if missing.\n    Example:\n      llamawrap-cli enable \"My Model\" --jinja",
    "disable": "disable <preset-name> <flag>\n    Disable (uncheck) a flag so it is skipped when building the command.\n    The flag and its value are preserved, just not emitted.",
    "rmflag": "rmflag <preset-name> <flag>\n    Remove a flag from the preset entirely. The flag is added to\n    hidden_flags so the GUI won't show it either.",
    "rename": "rename <preset-name> <new-name>\n    Rename a preset. Fails if a preset with the new name already exists.",
    "delete": "delete <preset-name>\n    Permanently delete a preset from history.json.",
    "run": "run <preset-name> [--auto]\n    Build the full server command from the preset, free the configured\n    port (stopping a matching inferer automatically; asking first for\n    anything else), launch the server, and stream its output to the\n    terminal. Press Ctrl+C to stop.\n\n    --auto  Restart the process automatically if it crashes (non-zero exit).\n            Press Ctrl+C once to stop gracefully.\n\n    Example:\n      llamawrap-cli run \"My Model\" --auto",
}


def cmd_help(args: list[str]) -> None:
    if len(args) >= 2:
        topic = args[1]
        if topic in HELP_DETAIL:
            print(HELP_DETAIL[topic])
        else:
            print(f"unknown command: {topic}", file=sys.stderr)
            print(HELP_TEXT.strip())
        return
    print(HELP_TEXT.strip())


# ───────────────────────────────────────
#  Interactive browser
# ───────────────────────────────────────

def interactive_browse(history_path: Path) -> None:
    """Interactive preset browser — default mode when no command is given."""
    data = load_or_init_history(history_path)
    presets = data.get("presets", [])
    show_help = True

    while True:
        print(f"\n── {bold('Presets')} ──")
        if not presets:
            print("  (no presets)")
        else:
            for i, p in enumerate(presets, 1):
                name = p.get("preset_name", "?")
                model = Path(p.get("model_path", "")).name or "(no model)"
                print(f"  {cyan(f'{i:>2}.')} {name}  {dim(f'({model})')}")
        if show_help:
            print()
            print("  c     create a new preset")
            print("  i     import a pasted launch command")
            print("  r     reload presets")
            print("  q     quit")
            print("  ?     show this help again")
            show_help = False
        print()
        try:
            choice = input("Select a preset number, or [c/i/r/q/?]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if choice == "?":
            show_help = True
            continue
        if choice == "q":
            break
        if choice == "c":
            created = prompt_create_preset(data, history_path)
            presets = data.get("presets", [])
            if created:
                interactive_preset_shell(created, history_path, data)
                data = load_history(history_path)
                presets = data.get("presets", [])
            continue
        if choice == "i":
            imported = prompt_import_preset(data, history_path)
            presets = data.get("presets", [])
            if imported:
                interactive_preset_shell(imported, history_path, data)
                data = load_history(history_path)
                presets = data.get("presets", [])
            continue
        if choice == "r":
            data = load_or_init_history(history_path)
            presets = data.get("presets", [])
            continue

        try:
            idx = int(choice) - 1
            if idx < 0 or idx >= len(presets):
                print("  invalid number")
                continue
        except ValueError:
            print("  enter a number, or c/i/r/q/?")
            continue

        preset = presets[idx]
        interactive_preset_shell(preset, history_path, data)
        # Reload after mutations
        data = load_history(history_path)
        presets = data.get("presets", [])


def interactive_preset_shell(preset: dict, history_path: Path, data: dict) -> None:
    """Interactive shell for a selected preset."""
    name = preset.get("preset_name", "?")
    actions = (
        ("s", "show full details"),
        ("f", "edit flags"),
        ("r", "run (launch server)"),
        ("a", "run with auto-restart on crash"),
        ("d", "delete this preset"),
        ("b", "back to list"),
    )
    show_help = True
    while True:
        print(f"\n── {bold(name)} ──")
        model = preset.get("model_path", "") or "(no model)"
        print(f"  Model:  {model}")
        flags = preset.get("flags", {})
        enabled_count = sum(1 for c in flags.values() if isinstance(c, dict) and c.get("enabled", False))
        print(f"  Flags:  {enabled_count} enabled, {len(flags)} total")
        stats = preset.get("session_stats", {})
        if stats:
            parts = []
            if "avg_ttft_ms" in stats:
                parts.append(f"{stats['avg_ttft_ms']}ms TTFT")
            if "avg_tok_s" in stats:
                parts.append(f"{stats['avg_tok_s']} tok/s")
            if stats.get("auto_restarts", 0):
                parts.append(f"{stats['auto_restarts']} restarts")
            if parts:
                print(f"  Last:   {' | '.join(parts)}")
        if show_help:
            print()
            for key, desc in actions:
                print(f"  {bold(key)}     {desc}")
            print(f"  {bold('?')}     show this help again")
            show_help = False
        try:
            action = input(f"\nAction [{'/'.join(k for k, _ in actions)}/?]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if action == "?":
            show_help = True
            continue
        if action == "b":
            break
        elif action == "s":
            cmd_show(data, name)
        elif action == "f":
            interactive_flag_editor(preset, history_path, data)
        elif action == "r":
            cmd_run(data, name, history_path=history_path)
            # Reload preset from disk so saved session_stats are visible
            reloaded = load_history(history_path)
            updated = find_preset(reloaded, name)
            if updated:
                preset.clear()
                preset.update(updated)
                data.clear()
                data.update(reloaded)
        elif action == "a":
            cmd_run(data, name, auto=True, history_path=history_path)
            reloaded = load_history(history_path)
            updated = find_preset(reloaded, name)
            if updated:
                preset.clear()
                preset.update(updated)
                data.clear()
                data.update(reloaded)
        elif action == "d":
            confirm = input(f"  delete '{name}'? (y/N): ").strip().lower()
            if confirm == "y":
                cmd_delete(data, history_path, name)
                break
        else:
            print("  unknown action (? for help)")


def _make_flag_completer(flag_names: list[str]):
    """Return a readline completer for flag names (enable/disable/rmflag/set)."""
    commands = ["enable", "disable", "rmflag", "set", "done"]

    def completer(text: str, state: int) -> str | None:
        line = readline.get_line_buffer().strip()
        # No space → completing command word
        if " " not in line:
            matches = [c for c in commands if c.startswith(text)]
            return matches[state] if state < len(matches) else None
        cmd = line.split()[0]
        if cmd in ("enable", "disable", "rmflag"):
            matches = [f for f in flag_names if f.startswith(text)]
            return matches[state] if state < len(matches) else None
        if cmd == "set":
            # After "set <flag>" no more flag completion (next word is value)
            if len(line.split()) <= 2:
                matches = [f for f in flag_names if f.startswith(text)]
                return matches[state] if state < len(matches) else None
        return None

    return completer


def interactive_flag_editor(preset: dict, history_path: Path, data: dict) -> None:
    """Interactive flag editing for a selected preset."""
    name = preset.get("preset_name", "?")
    print(f"\n── Flags: {name} ──")
    # Set up tab completion
    flag_names = sorted(preset.get("flags", {}).keys())
    old_completer = readline.get_completer() if readline is not None else None
    old_delims = readline.get_completer_delims() if readline is not None else None
    if readline is not None:
        readline.set_completer(_make_flag_completer(flag_names))
        readline.set_completer_delims(" \t\n")
        readline.parse_and_bind("tab: complete")
    while True:
        flags = preset.get("flags", {})
        enabled_flags = {f: c for f, c in flags.items() if isinstance(c, dict) and c.get("enabled", False)}
        if not enabled_flags:
            print("  (no enabled flags)")
        else:
            max_len = max(len(f) for f in enabled_flags)
            for fname, cfg in sorted(enabled_flags.items()):
                if not isinstance(cfg, dict):
                    continue
                if cfg.get("value_required", True):
                    val = cfg.get("value", "") or ""
                    print(f"    [+] {fname:<{max_len}}  {val}")
                else:
                    print(f"    [+] {fname:<{max_len}}  (toggle)")
        print()
        print("Commands: set <flag> <val>  |  enable <flag>  |  disable <flag>")
        print("          rmflag <flag>    |  done")
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line or line == "done":
            break

        parts = shlex.split(line)
        if not parts:
            continue

        sub = parts[0]
        try:
            if sub == "set" and len(parts) >= 3:
                cmd_set(data, history_path, name, parts[1], parts[2])
            elif sub == "enable" and len(parts) >= 2:
                cmd_enable(data, history_path, name, parts[1])
            elif sub == "disable" and len(parts) >= 2:
                cmd_disable(data, history_path, name, parts[1])
            elif sub == "rmflag" and len(parts) >= 2:
                cmd_rmflag(data, history_path, name, parts[1])
            else:
                print(f"  unknown: {line}")
        except SystemExit:
            pass

    if readline is not None:
        readline.set_completer(old_completer)
        readline.set_completer_delims(old_delims)


def main() -> None:
    history_path = find_history()

    args = sys.argv[1:]
    if "--no-color" in args:
        args = [a for a in args if a != "--no-color"]
        set_color_enabled(False)
    if "--quiet" in args or "-q" in args:
        args = [a for a in args if a not in ("--quiet", "-q")]
        set_quiet(True)
    if not args:
        interactive_browse(history_path)
        return
    if args[0] in ("-h", "--help"):
        cmd_help([])
        return
    if args[0] not in (
        "list", "create", "import", "show", "set", "enable", "disable",
        "rmflag", "rename", "delete", "run", "doctor", "probe", "bench",
        "stress", "export-presets", "import-presets", "help",
    ):
        print(f"unknown command: {args[0]}", file=sys.stderr)
        cmd_help([])
        sys.exit(1)

    cmd = args[0]
    if cmd == "help":
        cmd_help(args)
        return

    data = load_or_init_history(history_path) if cmd in {"create", "import", "import-presets"} else load_history(history_path)

    if cmd == "list":
        cmd_list(data)

    elif cmd == "create":
        cmd_create(data, history_path, args)

    elif cmd == "import":
        force = "--force" in args
        filtered = [a for a in args[2:] if a != "--force"]
        if len(args) < 3 or not filtered:
            error("usage: llamawrap-cli import <preset-name> <command-or-args...> [--force]")
        cmd_import(data, history_path, args[1], shlex.join(filtered), force=force)

    elif cmd == "show":
        if len(args) < 2:
            error("usage: llamawrap-cli show <preset-name>")
        cmd_show(data, args[1])

    elif cmd == "set":
        if len(args) < 4:
            error("usage: llamawrap-cli set <preset-name> <flag> <value>")
        cmd_set(data, history_path, args[1], args[2], args[3])

    elif cmd == "enable":
        if len(args) < 3:
            error("usage: llamawrap-cli enable <preset-name> <flag>")
        cmd_enable(data, history_path, args[1], args[2])

    elif cmd == "disable":
        if len(args) < 3:
            error("usage: llamawrap-cli disable <preset-name> <flag>")
        cmd_disable(data, history_path, args[1], args[2])

    elif cmd == "rmflag":
        if len(args) < 3:
            error("usage: llamawrap-cli rmflag <preset-name> <flag>")
        cmd_rmflag(data, history_path, args[1], args[2])

    elif cmd == "rename":
        if len(args) < 3:
            error("usage: llamawrap-cli rename <preset-name> <new-name>")
        cmd_rename(data, history_path, args[1], args[2])

    elif cmd == "delete":
        if len(args) < 2:
            error("usage: llamawrap-cli delete <preset-name>")
        cmd_delete(data, history_path, args[1])

    elif cmd == "run":
        auto = "--auto" in args
        name_args = [a for a in args[1:] if a != "--auto"]
        if not name_args:
            error("usage: llamawrap-cli run <preset-name> [--auto]")
        cmd_run(data, name_args[0], auto=auto, history_path=history_path)

    elif cmd == "doctor":
        if len(args) < 2:
            error("usage: llamawrap-cli doctor <preset-name>")
        cmd_doctor(data, args[1])

    elif cmd == "probe":
        if len(args) < 2:
            error("usage: llamawrap-cli probe <preset-name>")
        cmd_probe(data, args[1])

    elif cmd == "bench":
        cmd_bench(data, args)

    elif cmd == "stress":
        if len(args) < 2:
            error("usage: llamawrap-cli stress <preset-name>")
        cmd_stress(data, args[1])

    elif cmd == "export-presets":
        cmd_export_presets(data, args)

    elif cmd == "import-presets":
        cmd_import_presets(data, history_path, args)

    else:
        print(f"unknown command: {cmd}", file=sys.stderr)
        cmd_help([])
        sys.exit(1)


if __name__ == "__main__":
    main()
