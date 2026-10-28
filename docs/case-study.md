# Local AI operations: llama-wrap

## Problem

Launching a local model server often involves long commands, repeated configuration, and several separate checks when the server starts but a client still cannot use it. llama-wrap brings those steps into a desktop interface and CLI backed by the same operations core.

## Implementation

| Operational need | Implementation | Source |
| --- | --- | --- |
| Repeatable configuration | Import commands, preserve unknown flags, save presets, and export/import bundles. | [Shared core](../llamawrap_core.py) |
| Diagnose client compatibility | Check endpoints, inspect template capabilities, and parse a requested function tool call. | [Doctor, Probe, and Agent](../llamawrap_core.py) |
| Inspect performance and context limits | Warmup and repeated streamed requests, JSON/CSV benchmark output, staged context stress, and error classification. | [Benchmark and stress code](../llamawrap_core.py) |
| Recover saved settings | Atomic preset writes and bounded run history. | [Storage implementation](../llamawrap_core.py) |
| Support desktop and headless use | Separate GUI and CLI using the shared core. | [GUI](../llamawrap.py) · [CLI](../llamawrap-cli.py) |

## Verification

Run the existing suite from the repository root:

```sh
python -m unittest discover -s tests -v
```

The suite covers command building, preset round trips, invalid responses, timeouts, tool-call parsing, context failures, benchmark aggregation, GGUF metadata, and a CLI flow against a local HTTP fixture. It uses synthetic inputs and local fixtures rather than requiring a downloaded model.

On 29 October 2026, all **43 tests passed** in an isolated Linux copy of source commit [0d429f4](https://github.com/chelij/llama-wrap/commit/0d429f492da16c206f6fabf01d6df7986ad569fe). This was a headless source check; no GUI or real-model performance claim follows from that result.

[Automatic checks](../.github/workflows/tests.yml) run the same suite on pushes and pull requests. [Release builds](../.github/workflows/release.yml) also run it before packaging.

## Boundaries

- Agent checks establish basic endpoint/tool-call compatibility, not coding ability or a general model-quality score.
- Stress prompts are synthetic; a successful response does not establish long-context reasoning quality.
- Benchmark TTFT currently prefers server `prompt_ms` when supplied, which measures prompt evaluation rather than full client-observed time to first content.
- Without reported completion-token usage, the streaming fallback counts content chunks. Those counts are approximate and should not be treated as tokenizer-derived tokens.
- This repository does not yet publish a reproducible hardware/model comparison or measured user savings.

These limits make the current benchmark useful for local diagnosis, with care required before comparing results between servers.

## Delivery

[Download releases](https://github.com/chelij/llama-wrap/releases) · [Usage and setup](../README.md) · [Tests](../tests/test_core.py)
