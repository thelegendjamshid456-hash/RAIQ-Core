# Evidence Record: RAIQ Tiny-BPE v1

## Decision

**Engineering push approved. Capability claim denied.**

The RAIQ Tiny-BPE v1 checkpoint passed every mandatory engineering gate defined in `docs/evidence_gates.md`. The appropriate claim level is **E2 — Learns on a defined corpus**. No Code, Neural, Chem, or technical-reasoning capability claim is supported, because no contamination-controlled held-out capability benchmark has passed a predefined threshold.

## Validated checkpoint

| Field | Value |
|---|---|
| Checkpoint | `artifacts/tiny-bpe-v1/checkpoint_last.pt` |
| Model | RAIQ-Tiny-BPE |
| Parameters | 840,832 |
| Training step | 100 |
| Corpus ID | `raiq-technical-smoke-v1` |
| Tokenizer | RAIQ BPE, 384-token vocabulary |
| Evidence command | `PYTHONPATH=. python3 scripts/validate_model_evidence.py --checkpoint artifacts/tiny-bpe-v1/checkpoint_last.pt` |
| Overall engineering status | Passed |
| Supported claim level | E2 — Learns on a defined corpus |
| Capability status | Not established |

## Gate outcomes

| Gate | Result | Recorded evidence |
|---|---|---|
| Unit suite | Passed | 14 tests passed |
| Data integrity | Passed | Manifest split paths, byte counts, and SHA-256 hashes verified |
| Tokenizer integrity | Passed | Exact technical-Unicode round trip; tokenizer vocabulary 384 equals model vocabulary 384 |
| Model numerics | Passed | Finite loss 4.6864 and finite gradients during validation |
| Checkpoint continuity | Passed | Checkpoint loaded at step 100 with equivalent logits |
| Inference continuity | Passed | Checkpoint plus saved BPE tokenizer generated four new tokens without runtime failure |
| Learning signal | Passed | Full held-out validation loss 5.0647, below uniform-token baseline loss 5.9506 |
| Reproducibility record | Passed | Model, data, tokenizer, seed, device, dtype, platform, framework, and tokenizer metrics present |

## What can be said

The current RAIQ Tiny-BPE engineering stack is working end to end. It trains from random initialization, verifies corpus integrity, learns a measurable signal on the declared local validation split, saves and restores a checkpoint, and performs tokenizer-consistent inference.

## What cannot be said

The record does not prove that RAIQ can solve programming tasks, diagnose neural networks, perform chemical-engineering calculations, reason reliably, browse, use tools, or operate as a general technical AI. A loss result and runtime smoke output are not capability evidence. These claims remain unestablished until separately held-out, scored benchmarks pass their declared thresholds.

## Push criterion applied

This validation-policy commit may be pushed because all mandatory engineering gates passed. Any future commit that changes model behavior, tokenization, data, training, checkpointing, inference, or evaluation must rerun the validator and preserve its output before a push. If a gate fails, the model milestone must not be pushed as validated.
