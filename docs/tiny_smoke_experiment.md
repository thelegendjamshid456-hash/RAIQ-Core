# RAIQ Tiny Smoke Experiment

## Purpose

This experiment validates that the RAIQ Core implementation can tokenize text, construct a randomly initialized decoder-only Transformer, compute next-token loss, update model weights, evaluate held-out text, write a resumable checkpoint, reload that checkpoint, and generate through the KV-cache path. It is an engineering smoke test, not a quality demonstration of RAIQ-200M.

## Run specification

| Field | Value |
|---|---|
| Run name | `tiny-smoke` |
| Configuration | `configs/tiny.yaml` |
| Model | RAIQ-Tiny; 4 layers, 128 model width, 4 attention heads, 128-token context |
| Tokenizer | `raiq-byte-v1` byte-level UTF-8 tokenizer with RAIQ special tokens |
| Corpus | Separate slices of the user-provided RAIQ technical specification |
| Device | CPU |
| Precision | FP32 |
| Seed | 1337 |
| Optimizer steps | 20 |
| Batch size | 4 |
| Learning-rate schedule | 10-step linear warmup followed by cosine decay |
| Checkpoint | `artifacts/tiny-smoke/checkpoint_last.pt` (generated locally; not committed) |

## Results

The training loss moved from **6.1222** at the first optimizer step to **5.1230** at step 20. The held-out validation loss at step 20 was **5.2181**. The generated sample was intentionally not retained as a quality claim: with approximately twenty CPU steps, a byte-level tokenizer, and a very small corpus, it is expected to be largely incoherent.

| Metric | Observed value |
|---|---:|
| Initial training loss | 6.1222 |
| Final training loss | 5.1230 |
| Absolute loss reduction | 0.9992 |
| Validation loss at step 20 | 5.2181 |
| Unit tests | 5 passed |
| Checkpoint creation | Passed |
| Checkpoint load | Passed by test and generation command |
| KV-cache generation path | Passed |

## Test coverage

The automated suite passed five tests. It validates forward-logit shape and finite loss, backward gradients, causal masking against future-token leakage, equivalence between full and cached decoding for a final token, atomic checkpoint save/load, and byte-tokenizer round trips for Unicode technical text including chemical notation and units.

## Interpretation and limitations

The result proves that the code path is real and executable. It does **not** establish technical competence, reasoning quality, code generation ability, chemical-engineering accuracy, or a trained 200M model. A credible RAIQ-200M evaluation requires a licensed and documented technical corpus, a custom trained technical tokenizer, accelerator-backed training, a defined token budget, held-out benchmarks, and reproducible experiment records.
