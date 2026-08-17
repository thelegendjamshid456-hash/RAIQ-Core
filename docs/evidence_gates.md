# RAIQ Evidence Gates

## Purpose

RAIQ releases and GitHub pushes must be based on recorded evidence, not aspiration. This document defines the minimum evidence required for each claim level. A successful engineering gate does **not** imply that the model is a reliable technical assistant. A capability claim requires a separate, held-out benchmark result.

> **Push rule:** A milestone may be pushed only after every applicable engineering gate has passed and the evidence record has been generated. Capability claims may be included only when their corresponding capability gates pass.

## Claim levels

| Claim level | Permitted statement | Required evidence |
|---|---|---|
| E0 — Source only | The code is implemented. | Source review only; no claim that it runs. |
| E1 — Engineering functional | The training/inference pipeline runs end to end. | Automated tests, manifest verification, checkpoint save/load, and inference smoke output. |
| E2 — Learns on a defined corpus | The model reduces held-out loss on a stated dataset. | Fixed seed, versioned manifest, finite metrics, validation loss below a stated baseline, and saved metrics. |
| E3 — Domain capability | The model succeeds at a defined task class. | Held-out, contamination-controlled benchmark with scoring, pass threshold, failure analysis, and comparison baseline. |
| E4 — Reliable technical system | The complete RAIQ system is reliable for a declared scope. | Repeated E3 results across Code, Neural, Chem, and reasoning; tool/verification evidence; safety review; documented operating limits. |

The current RAIQ Tiny-BPE result may be evaluated for E1 and E2 only. It must not be described as E3 or E4 without the required evidence.

## Mandatory engineering gates

| Gate | Pass condition | Evidence artifact |
|---|---|---|
| Unit suite | All tests pass with no skipped required validation | `pytest` result and test count |
| Data integrity | Each configured corpus split matches its manifest path, byte count, and SHA-256 | Manifest-verification result |
| Tokenizer integrity | Tokenizer loads, exactly round-trips technical Unicode text, and its vocabulary fits the model configuration | Tokenizer test and serialized tokenizer metadata |
| Model numerics | Forward loss is finite and backward gradients exist | Core model test |
| Checkpoint continuity | A saved checkpoint reloads into an equivalent model state | Checkpoint round-trip test |
| Inference continuity | The checkpoint’s saved tokenizer and model generate without runtime failure | Inference smoke result |
| Learning signal | Fixed-run validation loss is finite and lower than the uniform-token baseline | Metrics and baseline comparison |
| Reproducibility record | Seed, configuration, corpus ID, tokenizer metadata, package versions, and metrics are recorded | Run metadata and configuration |

## Capability gates

A model may be described as capable in a task category only after a benchmark is held out from tokenizer training and model training, has deterministic scoring, and reaches its predefined threshold. The initial benchmark categories are Code, Neural, Chem, and technical reasoning. Examples, prompts, and expected answers must be versioned and separated from every training corpus.

No qualitative sample, loss decrease, or single hand-picked answer can substitute for a capability gate. A model that fails a gate remains an engineering-functional model at the highest supported claim level; it is not silently promoted.

## Enforcement behavior

The validation command must emit a structured evidence record with a `passed` field for every engineering gate and a top-level claim level. The command exits non-zero if any required gate fails. The commit/push workflow must run this command before staging the milestone. If an E3 capability benchmark is absent or fails, reports must state that capability is **not established**.
