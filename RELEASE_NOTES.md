# Release Notes

## v0.1.13 - 2026-07-02 - Reliability and Benchmarking Patch

### Added

- Diagnostics output now streams into the GUI log line by line while Doctor/Probe/Bench/Stress run, with a new Cancel button; CLI `stress` and `bench` stream too and stop cleanly on Ctrl+C.
- Bench reworked: warmup request plus several streamed iterations, reporting median TTFT, generation tok/s, and prefill tok/s (from `llama-server` timings when available) instead of one request whose tok/s included prefill.
- Stress suite scales per-request timeouts from the observed prefill speed, so large-context stages on slow hardware no longer fail on a fixed cutoff; stress records now include TTFT and prefill speed when the server reports timings.
- Server output is written to rotating per-session log files under the data directory (`logs/`, last 10 sessions kept), so crash output survives closing the GUI.
- Windows support for port inspection and full process-tree stop (`netstat`/`tasklist`/`taskkill`, CTRL_BREAK for graceful shutdown).
- The bundled launcher includes its dynamically loaded CLI tab-completion dependency; Windows CLI runs without requiring the Unix-only `readline` module.
- The GUI now has preset search plus keyboard shortcuts for search (`Ctrl+F`), launch (`Ctrl+Enter`), save (`Ctrl+S`), and stop (`Escape` when focus is outside a text field).
- CLI output now uses terminal-aware colors, supports `--no-color`/`NO_COLOR` and `--quiet`, accepts unique partial preset names, and suggests close matches for misspellings.

### Changed

- Launching no longer silently kills whatever owns the configured port. Processes matching the inferer executable (or llama-wrap's own previous server) are stopped automatically; anything else prompts in the GUI, or asks/aborts in the CLI.
- Stop escalates to a force kill of the whole process tree if the server has not exited 5 seconds after a graceful stop request.
- Auto-restart is capped at 3 attempts with exponential backoff and resets once the server reaches Ready, preventing infinite crash loops.
- `history.json` for new installs lives in the per-user data directory (`~/.config/llama-wrap`, `~/Library/Application Support/llama-wrap`, or `%APPDATA%\llama-wrap`); existing files next to the app and `LLAMA_WRAP_HISTORY` keep working. All history writes are atomic.
- `--metrics` is only appended for llama.cpp/llama-server presets, so custom inferers that don't know the flag can start.
- Importing commands with unknown flags taking negative values (e.g. `--seed -1`) now parses correctly.
- The CLI reuses the shared core history helpers instead of duplicating them.
- The GUI remembers its last valid window geometry and shows an animated status pulse while the server starts or diagnostics are running.
- Interactive CLI menus are less repetitive while keeping `?` available to show their action reference on demand.

## 2026-06-24 - Ops Diagnostics Patch

This release expands `llama-wrap` from a launcher into a lightweight local-AI ops helper while keeping the existing preset format backward compatible.

### Added

- Shared core logic for GUI and CLI command building, preset parsing, endpoint checks, benchmark formatting, preset bundle portability, context stress testing, and VRAM calibration helpers.
- CLI diagnostics:
  - `doctor <preset>` checks executable/model setup, host/port, port availability, `/health`, `/v1/models`, and `/v1/chat/completions`.
  - `probe <preset>` sends a small OpenAI-compatible chat completion request.
  - `bench <preset>` runs a short benchmark prompt and saves JSON results, with optional CSV output.
  - `stress <preset>` runs the same context stress suite as the GUI.
- GUI Diagnostics row with `Doctor`, `Probe`, `Bench`, and `Stress` buttons that stream output into the existing log panel.
- Context stress suite with runtime context detection, fill/decode stages, sustained synthetic coding-agent turns, boundary probes, error classification, and practical working-limit summary.
- Runtime VRAM calibration from `llama-server` allocation logs and GPU process/total VRAM readings, stored per matching preset command/model signature.
- GUI stop logging so Stop reports both the stop request and confirmed server stop.
- CLI preset bundle portability commands:
  - `export-presets --out <file> [--portable] [preset-name...]`
  - `import-presets <file> [--force]`
- Local ignored output folder `.llama-wrap/benchmarks` for benchmark reports.

### Changed

- GUI and CLI now share command construction and command-import parsing to reduce drift.
- The GUI VRAM bar uses calibrated estimates when a matching calibration record exists.
- Generated release/local data folders stay out of source control, including new ignore coverage for `llama-wrap/` and `.llama-wrap/`.

### Not Included

- No chat UI.
- No model downloader.
- No RAG or agent framework.
- No GUI export/import buttons in the Diagnostics row.
- No Stress progress popup.
