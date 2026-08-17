# Colab Pretraining Workflow Readiness Record

## What is complete

The RAIQ repository now contains a separate Tesla T4 smoke configuration, reproducible text-manifest generation, absolute-path manifest verification for Google Drive, a Colab environment report, and an exact Drive-backed Colab runbook. The existing Transformer, tokenizer, trainer, checkpoint loader, and production preflight were preserved.

## Actual local dataset facts

| Field | Observed value |
|---|---|
| Train file | `datasets/technical_toy_train.txt` |
| Validation file | `datasets/technical_toy_validation.txt` |
| Format | UTF-8 plain text with CRLF line endings |
| Train bytes | 13,860 |
| Validation bytes | 14,547 |
| Total dataset size | Approximately 36 KiB on disk |
| Pre-tokenized | No |
| Existing split | Yes; train and validation files already exist |
| Manifest | `data/manifests/technical_smoke_v1.json` with SHA-256 hashes |
| Production eligibility | No; smoke corpus only |

## Model and configuration evidence

The T4 configuration preserves the RAIQ-200M-v1 architecture at **190,348,032 parameters** and reduces only the experiment sequence length to 2,048, uses micro-batch 1, gradient accumulation 32, float16 mixed precision, and gradient checkpointing. The production configuration remains unchanged.

## Validation evidence available in this environment

| Gate | Result |
|---|---|
| Full automated tests | **18/18 passed** |
| Actual dataset manifest verification | **Passed** |
| TextBlockDataset train/validation loading | **Passed** |
| 200M model instantiation and exact parameter count | **Passed** |
| Two-process distributed smoke | **Passed locally on CPU/Gloo** |
| Checkpoint-resume smoke | **Passed locally** |
| Integrated training smoke | **Passed locally on CPU with the existing trainer** |
| CUDA/T4 execution | **Not measured here**; this environment reports no CUDA device |

## Google-side measurements still required

The Colab runbook must be executed on the target T4 before claiming T4 readiness. It must record actual GPU name, VRAM, peak allocated/reserved VRAM, tokens per second, finite and decreasing validation loss, checkpoint location on Drive, and resume-after-restart evidence. No T4 measurement is fabricated in this record.

The local smoke corpus is not sufficient for useful 200M pretraining or coding/reasoning capability claims. It is only sufficient to exercise the data, tokenizer, model, checkpoint, and resume pipeline.
