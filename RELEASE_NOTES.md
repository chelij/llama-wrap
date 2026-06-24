# Release Notes

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
