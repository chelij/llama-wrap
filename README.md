# llama-wrap

`llama-wrap` is a lightweight desktop launcher for `llama-server` arguments.

It is not a chat UI. It is a small Tkinter app for building, importing, saving, and running `llama-server`-compatible commands with GGUF models.

## Features

- Browse for model, draft model, and MMProj `.gguf` files.
- Choose a `llama.cpp`, `ik_llama.cpp`, or custom llama-server-compatible inferer.
- Edit common server flags without typing the full command.
- Add or remove visible flag rows for custom or changing server arguments.
- Use optional 2^n controls for numeric flags.
- Paste an existing server command and import recognized values into the UI.
- Put advanced or uncommon options in `Extra args`.
- Save and reload launch presets.
- View live server output.
- Show launch status, including model-loaded detection.
- Show lightweight tokens/second from server log lines when available.
- Optionally restart the server after a crash.
- Show a rough VRAM estimate and breakdown.

## Requirements

- Python 3.10 or newer.
- Tkinter for your Python installation.
- `llama-server` from `llama.cpp`, `ik_llama.cpp`, or another compatible server executable.
- At least one GGUF model file.
- Optional smaller draft GGUF model for speculative decoding.
- Optional MMProj GGUF file for multimodal/vision models.

The selected executable must be available in your `PATH`, or entered as a full path in the executable field.

```bash
llama-server --help
```

If that command works, `llama-wrap` should be able to launch the default `llama.cpp` profile.

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
2. Optionally choose a smaller draft `.gguf` model for speculative decoding.
3. Optionally choose an MMProj `.gguf` file.
4. Adjust the common flags shown in the UI.
5. Add custom flags with the plus button beside `Flags`, or put advanced one-off arguments in `Extra args`.
6. Press launch to start the selected inferer.
7. Watch the output panel for logs.
8. Save the setup as a preset if you want to reuse it.

## Importing Commands

Use the import button to paste an existing command, for example:

```bash
llama-server -m /models/model.gguf -ngl auto -c 32768 --host 127.0.0.1 --port 8123 -fa auto
```

Recognized flags are loaded into the UI. Unrecognized or advanced flags are preserved in `Extra args`.

Custom flags pasted from a command are added as editable flag rows when possible.

## Draft Models

The `Draft` field adds `-md` / `--model-draft` for speculative decoding. This uses a smaller model beside the main model to propose tokens that the main model verifies.

For MTP/self-speculative models, enable `--spec-type` and use `draft-mtp` for current llama.cpp builds. Some older MTP branches used `mtp`, so the launcher keeps both values selectable.

Example:

```bash
llama-server -m /models/large.gguf -md /models/small-draft.gguf --spec-draft-n-max 16
llama-server -m /models/mtp.gguf --spec-type draft-mtp --spec-draft-n-max 3
```

## Inferers

The inferer selector controls the executable and which optional flags are shown.

- `llama.cpp` shows common `llama-server` flags.
- `ik_llama.cpp` shows common flags plus ik-specific options such as `--fit`, `--fit-margin`, `-mla`, `-fmoe`, `-cram`, `-khad`, and `-vhad`.
- `Custom` is for other llama-server-compatible executables. Put unsupported or unusual flags in `Extra args`.

`ik_llama.cpp` still builds a `llama-server` binary. If it is not in your `PATH`, set the executable field to the full path, for example:

```bash
/home/you/ik_llama.cpp/build/bin/llama-server
```

## Presets

Presets are stored in `history.json` next to `llamawrap.py` when running from source, or next to the packaged launcher when running a release build.

Presets include:

- model path
- MMProj path
- draft model path
- enabled flags and values
- custom and removed flag rows
- extra args
- selected inferer and executable

Recent run commands are also stored in `history.json`.

Loaded presets require double-click or Enter. If the selected preset has unsaved changes, `*` is shown beside its name. Saving a changed selected preset updates it without asking for the name again.

## VRAM Estimate

The VRAM display is an estimate, not a guarantee.

It uses model file size, parsed GGUF metadata when available, context size (`-c`), KV cache type, GPU layers, draft model size, MMProj size, and a runtime overhead estimate. Sizes are shown in binary units such as MiB and GiB.

If GGUF metadata cannot be read, launching can still work, but the estimate may be less accurate.

## Notes

- This app does not download models.
- This app does not manage chat conversations.
- This app does not replace Open WebUI, LM Studio, or the built-in llama.cpp Web UI.
- It is intended as a small local process wrapper for people who already use llama-server-compatible inferers.

## Support

If this is useful to you, donations are welcome:

[https://ko-fi.com/chelib](https://ko-fi.com/chelib)
