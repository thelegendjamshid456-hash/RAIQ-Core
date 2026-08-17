# Local Production Preflight Record — Not Approved

**Decision:** `not_ready_do_not_start_production_pretraining`

The RAIQ production-pretraining preparation code was implemented and validated locally, but the production preflight did not pass. Per the evidence-gated push rule, these local changes have **not** been committed or pushed to GitHub.

## Checks that passed locally

| Gate | Result | Evidence |
|---|---|---|
| Unit suite | Passed | 14 tests passed |
| Production configuration | Passed | CUDA and BF16 are correctly specified in `configs/200m_production.yaml` |
| Distributed primitives | Passed | Two-process CPU Gloo smoke test reduced rank values to the expected mean of 1.5 |
| Checkpoint-resume primitive | Passed | Model weights restored equivalently and continuation loss was finite |

## Production blockers

| Gate | Observed result | Required to pass |
|---|---|---|
| Accelerator | Failed | At least 2 CUDA GPUs, each with at least 40 GiB memory; current environment has 0 CUDA GPUs |
| Storage | Failed | At least 500 GiB free; current environment has approximately 31 GiB free |
| Production data | Failed | An approved, production-eligible corpus manifest with complete source records, license approval, quality/deduplication/contamination reports, and train/validation/benchmark splits |

## Required next action

A project owner must provide or approve a suitable GPU training environment and an approved corpus manifest. After those resources exist, the production preflight must be rerun and pass before the local preparation work can be committed and pushed as production-pretraining-ready.
