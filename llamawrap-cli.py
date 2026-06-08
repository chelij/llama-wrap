#!/usr/bin/env python3
"""
Interactive preset browser and CLI for llama-wrap history.json.

Usage:
    llamawrap-cli              Interactive preset browser (default)
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

import json
import os
import re
import shlex
import readline
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def find_history() -> Path:
    """Locate history.json next to the script or in the working directory."""
    candidates = [
        Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else None,
        Path(__file__).resolve().parent,
        Path.cwd(),
    ]
    for c in candidates:
        if c is None:
            continue
        p = c / "history.json"
        if p.exists():
            return p
    # Fallback: current dir
    return Path.cwd() / "history.json"


def load_history(path: Path) -> dict:
    if not path.exists():
        print(f"error: {path} not found", file=sys.stderr)
        sys.exit(1)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"error: cannot read {path}: {e}", file=sys.stderr)
        sys.exit(1)


def save_history(path: Path, data: dict) -> None:
    data["presets"] = data.get("presets", [])
    data["runs"] = (data.get("runs") or [])[-100:]
    data["settings"] = data.get("settings", {})
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def find_preset(data: dict, name: str) -> dict | None:
    for p in data.get("presets", []):
        if p.get("preset_name") == name:
            return p
    return None


def error(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


# ───────────────────────────────────────
#  Commands
# ───────────────────────────────────────

def cmd_list(data: dict) -> None:
    presets = data.get("presets", [])
    if not presets:
        print("No presets.")
        return
    width = max(len(p.get("preset_name", "")) for p in presets)
    for p in presets:
        name = p.get("preset_name", "?")
        model = Path(p.get("model_path", "")).name or "(no model)"
        print(f"  {name:<{width}}  {model}")


def cmd_show(data: dict, name: str) -> None:
    preset = find_preset(data, name)
    if not preset:
        error(f"preset '{name}' not found")

    print(f"Preset: {preset['preset_name']}")
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
    preset = find_preset(data, name)
    if not preset:
        error(f"preset '{name}' not found")
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
    preset = find_preset(data, name)
    if not preset:
        error(f"preset '{name}' not found")
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
    preset = find_preset(data, name)
    if not preset:
        error(f"preset '{name}' not found")
    flags = preset.setdefault("flags", {})
    if flag in flags:
        flags[flag]["enabled"] = False
    save_history(path, data)
    print(f"  disabled {flag}")


def cmd_rmflag(data: dict, path: Path, name: str, flag: str) -> None:
    preset = find_preset(data, name)
    if not preset:
        error(f"preset '{name}' not found")
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
    preset = find_preset(data, name)
    if not preset:
        error(f"preset '{name}' not found")
    if find_preset(data, new_name):
        error(f"preset '{new_name}' already exists")
    preset["preset_name"] = new_name
    save_history(path, data)
    print(f"  renamed '{name}' -> '{new_name}'")


def cmd_delete(data: dict, path: Path, name: str) -> None:
    presets = data.get("presets", [])
    for i, p in enumerate(presets):
        if p.get("preset_name") == name:
            presets.pop(i)
            save_history(path, data)
            print(f"  deleted '{name}'")
            return
    error(f"preset '{name}' not found")


def build_command_from_preset(preset: dict) -> list[str]:
    """Build the server command list from a preset dict (same logic as the GUI)."""
    executable = preset.get("inferer_executable", "llama-server")
    try:
        command = shlex.split(executable)
    except ValueError as e:
        error(f"invalid executable in preset: {e}")
    if not command:
        error("executable is required")

    model_path = preset.get("model_path", "").strip()
    if not model_path:
        error("preset has no model path set")
    command.extend(["-m", model_path])

    draft_path = (preset.get("draft_model_path") or "").strip()
    if draft_path:
        command.extend(["-md", draft_path])

    flags = preset.get("flags", {})
    fit_enabled = any(
        f == "--fit" and isinstance(c, dict) and c.get("enabled", False)
        for f, c in flags.items()
    )
    ngl_enabled = any(
        f == "-ngl" and isinstance(c, dict) and c.get("enabled", False) and c.get("value", "").strip()
        for f, c in flags.items()
    )
    # Inferer check for ik-specific flags — skip if no ik support needed
    for fname, cfg in sorted(flags.items()):
        if not isinstance(cfg, dict) or not cfg.get("enabled", False):
            continue
        if fit_enabled and fname == "-ngl":
            continue
        if fname == "-ngl" and ngl_enabled and not fit_enabled:
            command.extend(["--fit", "off"])
        if fname == "--mmproj" and not cfg.get("value", "").strip():
            continue
        if cfg.get("value_required", True):
            val = cfg.get("value", "").strip()
            if val:
                command.extend([fname, val])
        else:
            command.append(fname)

    extra = (preset.get("extra_args") or "").strip()
    if extra:
        try:
            command.extend(shlex.split(extra))
        except ValueError as e:
            error(f"invalid extra_args in preset: {e}")

    # Always enable --metrics so the /metrics endpoint is available
    if "--metrics" not in command:
        command.append("--metrics")

    return command


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


def kill_process_on_port(port: int | None) -> None:
    if port and os.name != "nt" and shutil.which("lsof"):
        try:
            subprocess.check_output(
                ["lsof", f"-tiTCP:{port}", "-sTCP:LISTEN"],
                text=True, stderr=subprocess.DEVNULL, timeout=3,
            )
            print(f"  killing process on port {port}...")
            subprocess.run(
                ["kill"] + subprocess.check_output(
                    ["lsof", f"-tiTCP:{port}", "-sTCP:LISTEN"],
                    text=True, stderr=subprocess.DEVNULL, timeout=3,
                ).strip().split(),
                timeout=3,
            )
            time.sleep(0.5)
        except Exception:
            pass


def get_port_from_preset(preset: dict) -> int | None:
    flags = preset.get("flags", {})
    if "--port" in flags and isinstance(flags["--port"], dict):
        try:
            return int(str(flags["--port"].get("value", "8080")).strip())
        except (ValueError, TypeError):
            return 8080
    return None


def cmd_run(data: dict, name: str, auto: bool = False, history_path: Path | None = None) -> None:
    preset = find_preset(data, name)
    if not preset:
        error(f"preset '{name}' not found")

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
    kill_process_on_port(port)
    run_process(command, auto=auto, preset_name=name, history_path=history_path, port=port)


# ───────────────────────────────────────
#  Main
# ───────────────────────────────────────

HELP_TEXT = """Commands:

  list                        List all presets.
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
"""

HELP_DETAIL = {
    "list": "list\n    List all saved presets with their model file names.",
    "show": "show <preset-name>\n    Display the preset's inferer, executable, model paths, all flags,\n    their enabled/disabled status, and current values.",
    "set": "set <preset-name> <flag> <value>\n    Set a flag's value and enable it. If the flag doesn't exist in the\n    preset it is added automatically. Example:\n      llamawrap-cli set \"My Model\" --port 8080",
    "enable": "enable <preset-name> <flag>\n    Enable (check/tick) a flag so it is included when building the\n    server command. Creates the flag as a toggle (no value) if missing.\n    Example:\n      llamawrap-cli enable \"My Model\" --jinja",
    "disable": "disable <preset-name> <flag>\n    Disable (uncheck) a flag so it is skipped when building the command.\n    The flag and its value are preserved, just not emitted.",
    "rmflag": "rmflag <preset-name> <flag>\n    Remove a flag from the preset entirely. The flag is added to\n    hidden_flags so the GUI won't show it either.",
    "rename": "rename <preset-name> <new-name>\n    Rename a preset. Fails if a preset with the new name already exists.",
    "delete": "delete <preset-name>\n    Permanently delete a preset from history.json.",
    "run": "run <preset-name> [--auto]\n    Build the full server command from the preset, kill any existing\n    process on the configured port, launch the server, and stream its\n    output to the terminal. Press Ctrl+C to stop.\n\n    --auto  Restart the process automatically if it crashes (non-zero exit).\n            Press Ctrl+C once to stop gracefully.\n\n    Example:\n      llamawrap-cli run \"My Model\" --auto",
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
    data = load_history(history_path)
    presets = data.get("presets", [])

    while True:
        print("\n── Presets ──")
        if not presets:
            print("  (no presets — create one in the GUI first)")
            break
        for i, p in enumerate(presets, 1):
            name = p.get("preset_name", "?")
            model = Path(p.get("model_path", "")).name or "(no model)"
            print(f"  {i:>2}. {name}  ({model})")
        print()
        try:
            choice = input("Enter number to select, r to reload, q to quit: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if choice == "q":
            break
        if choice == "r":
            data = load_history(history_path)
            presets = data.get("presets", [])
            continue

        try:
            idx = int(choice) - 1
            if idx < 0 or idx >= len(presets):
                print("  invalid number")
                continue
        except ValueError:
            print("  enter a number, r, or q")
            continue

        preset = presets[idx]
        interactive_preset_shell(preset, history_path, data)
        # Reload after mutations
        data = load_history(history_path)
        presets = data.get("presets", [])


def interactive_preset_shell(preset: dict, history_path: Path, data: dict) -> None:
    """Interactive shell for a selected preset."""
    name = preset.get("preset_name", "?")
    while True:
        print(f"\n── {name} ──")
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
        print()
        print("  s     show full details")
        print("  f     edit flags")
        print("  r     run (launch server)")
        print("  a     run with auto-restart on crash")
        print("  d     delete this preset")
        print("  b     back to list")
        try:
            action = input("\nAction: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break

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
            print("  unknown action")


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
    old_completer = readline.get_completer()
    old_delims = readline.get_completer_delims()
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

    readline.set_completer(old_completer)
    readline.set_completer_delims(old_delims)


def main() -> None:
    history_path = find_history()

    args = sys.argv[1:]
    if not args:
        interactive_browse(history_path)
        return
    if args[0] in ("-h", "--help"):
        cmd_help([])
        return
    if args[0] not in (
        "list", "show", "set", "enable", "disable",
        "rmflag", "rename", "delete", "run", "help",
    ):
        print(f"unknown command: {args[0]}", file=sys.stderr)
        cmd_help([])
        sys.exit(1)

    data = load_history(history_path)
    cmd = args[0]

    if cmd == "help":
        cmd_help(args)

    elif cmd == "list":
        cmd_list(data)

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

    else:
        print(f"unknown command: {cmd}", file=sys.stderr)
        cmd_help([])
        sys.exit(1)


if __name__ == "__main__":
    main()
