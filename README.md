# llama-wrap

`llama-wrap` is a lightweight launcher for `llama-server` — available as both a **desktop GUI** (Tkinter) and an **interactive CLI** (zero dependencies).

It is not a chat UI. It is a tool for building, importing, saving, and running `llama-server`-compatible commands with GGUF models.

## Features

- **GUI & CLI** — use the desktop launcher or the terminal, whichever fits your workflow.
- **Browse** for model, draft model, and MMProj `.gguf` files.
- **Choose** the default `llama.cpp` inferer or a custom llama-server-compatible executable.
- **Edit** common server flags without typing the full command — or add custom flag rows.
- **2ⁿ step controls** for numeric flags.
- **Import** an existing server command and parse recognized values into the UI.
- **Presets** — save, reload, rename, and delete launch configurations.
- **Session stats** — average TTFT (ms) and average tok/s are tracked across the session and saved to each preset. Accurate via the auto-enabled `--metrics` endpoint.
- **Auto-restart** — optionally restart the server after a crash, with a visible restart counter.
- **Live output** — view server logs in the GUI or piped to the terminal in the CLI.
- **VRAM estimate** — model-aware breakdown of GPU memory usage.
- **Interactive CLI browser** — select presets by number, edit flags with tab completion, and launch with a single keypress.

## Requirements

- Python 3.10 or newer.
- Tkinter (for the GUI mode; the CLI does not need it).
- `llama-server` from `llama.cpp` or another compatible server executable.
- At least one GGUF model file.
- Optional smaller draft GGUF model for speculative decoding.
- Optional MMProj GGUF file for multimodal/vision models.

## Run

### GUI

```bash
python llamawrap.py
```

### CLI (interactive browser)

```bash
python llamawrap-cli.py
```

The CLI opens an interactive browser — just select a preset by number, then choose an action:

```
  s     show full details
  f     edit flags
  r     run (launch server)
  a     run with auto-restart on crash
  d     delete this preset
  b     back to list
```

### CLI (direct commands)

```bash
llamawrap-cli list
llamawrap-cli show "My Preset"
llamawrap-cli run "My Preset" --auto
llamawrap-cli set "My Preset" --port 8080
llamawrap-cli enable "My Preset" --jinja
llamawrap-cli disable "My Preset" -ngl
llamawrap-cli rename "Old Name" "New Name"
llamawrap-cli help run
```

### Release builds

Extract the archive for your platform and run:

- **Windows:** `llama-wrap.exe`
- **macOS:** open `llama-wrap.app` or run the packaged `llama-wrap` executable
- **Linux:** `./llama-wrap`

The CLI (`llamawrap-cli.py`) is not packaged into the binary — run it directly with Python from a source checkout.

## Session Stats

Every time you run a preset, `llama-wrap` tracks:

- **Average TTFT** (Time To First Token) — accumulated from each request's prompt eval time.
- **Average Tok/s** — weighted average across all generation steps.

These are computed from the server's `/metrics` endpoint (auto-enabled) and saved to the preset on stop. The `Last:` line in the CLI or "Last session:" label in the GUI shows them at a glance.

```
Last:  53.1ms TTFT | 41.51 tok/s
```

With auto-restart, stats accumulate across all restarts and a restart counter is shown:

```
Last:  53.1ms TTFT | 41.51 tok/s | 3 restarts
```

## Basic Usage

1. Choose a model `.gguf` file.
2. Optionally choose a smaller draft `.gguf` model for speculative decoding.
3. Optionally choose an MMProj `.gguf` file.
4. Adjust the common flags shown in the UI.
5. Add custom flags with the plus button beside `Flags`, or put advanced one-off arguments in `Extra args`.
6. Press launch to start the selected inferer.
7. Watch the output panel for logs.
8. Save the setup as a preset if you want to reuse it.

## Presets

Presets are stored in `history.json` next to `llamawrap.py` when running from source, or next to the launcher binary in release builds.

Each preset stores:

- model path, MMProj path, draft model path
- enabled flags and their values
- custom and hidden flag rows
- extra args
- selected inferer and executable
- **session stats** from the last run (TTFT avg, tok/s avg, restart count)

Recent run commands are also saved.

The CLI's `list` and `show` commands display everything in a compact format.

## CLI Tab Completion

In the CLI's interactive flag editor (`f`), tab completion is available:

- `ena` + Tab → `enable`
- `enable --p` + Tab → `--port`, `--presence-penalty`, etc.
- `set -n` + Tab → `-ngl`, `-np`, etc.
- Works for `enable`, `disable`, `rmflag`, and `set`.

## Importing Commands

Use the import button (GUI) to paste an existing server command:

```bash
llama-server -m /models/model.gguf -ngl all -c 32768 --host 127.0.0.1 --port 8080
```

Recognized flags are loaded into the UI. Unrecognized flags stay in `Extra args`. Custom flags are added as editable rows.

## Inferers

- `llama.cpp` uses the standard `llama-server` executable.
- `Custom` is for other llama-server-compatible executables. Selecting it auto-focuses the executable field.

## VRAM Estimate

The VRAM display is an estimate based on model file size, parsed GGUF metadata, context size (`-c`), KV cache type, GPU layers, draft model size, MMProj size, and runtime overhead. Shown in binary units (MiB, GiB).

## Notes

- This app does not download models.
- This app does not manage chat conversations.
- This app does not replace Open WebUI, LM Studio, or the built-in llama.cpp Web UI.
- It is a small local process wrapper for people who already use llama-server-compatible inferers.
- `--metrics` is automatically added to every launch for accurate session stats. The `/metrics` endpoint is available at `http://127.0.0.1:<port>/metrics`.

## Support

If this is useful to you, donations are welcome:

[https://ko-fi.com/chelib](https://ko-fi.com/chelib)
