# llama-wrap

`llama-wrap` is a lightweight desktop launcher for `llama-server` arguments.

It is not a chat UI. It is a small Tkinter app for building, importing, saving, and running `llama-server` commands with GGUF models.

## Features

- Browse for model and MMProj `.gguf` files.
- Edit common `llama-server` flags without typing the full command.
- Paste an existing `llama-server` command and import recognized values into the UI.
- Put advanced or uncommon options in `Extra args`.
- Save and reload launch presets.
- View live server output.
- Show launch status, including model-loaded detection.
- Optionally restart the server after a crash.
- Show a rough VRAM estimate and breakdown.

## Requirements

- Python 3.10 or newer.
- Tkinter for your Python installation.
- `llama-server` from `llama.cpp`.
- At least one GGUF model file.
- Optional MMProj GGUF file for multimodal/vision models.

`llama-server` must be available in your `PATH`.

```bash
llama-server --help
```

If that command works, `llama-wrap` should be able to launch it.

## Python Dependencies

There are no third-party Python package dependencies.

`llama-wrap` uses only the Python standard library. `requirements.txt` is intentionally empty except for a note.

## Run

From source:

```bash
python llamawrap.py
```

From a release download, extract the archive first, then run the executable for your platform:

- Windows: run `llama-wrap.exe`.
- macOS: open `llama-wrap.app` or run the packaged `llama-wrap` executable.
- Linux: run `./llama-wrap`.

The executable releases are built automatically on GitHub Actions. I can test the Linux build locally, but I cannot personally test the Windows and macOS release builds.

## Basic Usage

1. Choose a model `.gguf` file.
2. Optionally choose an MMProj `.gguf` file.
3. Adjust the common flags shown in the UI.
4. Add advanced flags in `Extra args` if needed.
5. Press launch to start `llama-server`.
6. Watch the output panel for logs.
7. Save the setup as a preset if you want to reuse it.

## Importing Commands

Use the import button to paste an existing command, for example:

```bash
llama-server -m /models/model.gguf -ngl auto -c 32768 --host 127.0.0.1 --port 8123 -fa auto
```

Recognized flags are loaded into the UI. Unrecognized or advanced flags are preserved in `Extra args`.

## Presets

Presets are stored in `history.json` next to `llamawrap.py` when running from source, or next to the packaged launcher when running a release build.

Presets include:

- model path
- MMProj path
- enabled flags and values
- extra args

Recent run commands are also stored in `history.json`.

## VRAM Estimate

The VRAM display is an estimate, not a guarantee.

It uses model file size, parsed GGUF metadata when available, context size, KV cache type, GPU layers, MMProj size, and a fixed overhead estimate.

If GGUF metadata cannot be read, launching can still work, but the estimate may be less accurate.

## Notes

- This app does not download models.
- This app does not manage chat conversations.
- This app does not replace Open WebUI, LM Studio, or the built-in llama.cpp Web UI.
- It is intended as a small local process wrapper for people who already use `llama-server`.

## Support

If this is useful to you, donations are welcome:

[https://ko-fi.com/chelib](https://ko-fi.com/chelib)
