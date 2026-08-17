# RAIQ Model Training Milestone v1

## What this milestone proves

This milestone advances the actual RAIQ model path beyond the original byte-token smoke run. RAIQ Core now includes a trainable, deterministic byte-pair encoding (BPE) tokenizer; a versioned corpus manifest with split hashes; tokenizer-aware training and inference; persisted tokenization metrics; validation perplexity; and a completed 100-step model-training run using these components together.

> This is a genuine, randomly initialized model-training experiment. It is not evidence of a capable technical assistant or a trained RAIQ-200M model.

## Reproducible inputs

| Item | Value |
|---|---|
| Training configuration | `configs/tiny_bpe.yaml` |
| Corpus identifier | `raiq-technical-smoke-v1` |
| Manifest | `data/manifests/technical_smoke_v1.json` |
| Corpus purpose | Engineering smoke corpus derived from the user-provided RAIQ specification |
| Corpus status | Local only, excluded from Git, not production eligible |
| Tokenizer | `raiq-technical-bpe`, version `smoke-v1` |
| Tokenizer vocabulary | 384 tokens, including 115 learned BPE merges and RAIQ special tokens |
| Model | RAIQ-Tiny-BPE, 4 layers, 128 dimensions, 4 heads, 128-token context |
| Model parameters | 840,832 |
| Device and precision | CPU, FP32 |
| Random seed | 20260817 |
| Optimization | AdamW, linear warmup then cosine learning-rate decay |

The tokenizer trainer starts with byte-level UTF-8 tokens and repeatedly merges the most frequent adjacent pair under deterministic tie-breaking. It exactly round-trips Unicode technical text while providing measurable compression. On the local training corpus it produced **7,001 tokens from 12,930 UTF-8 bytes**, or **0.5415 tokens per byte**.

## Training result

The `tiny-bpe-v1` run completed 100 optimizer steps, saved periodic and final checkpoints, verified the corpus manifest before training, wrote the trained tokenizer into the run directory, and generated through the checkpoint inference path.

| Metric | Step 20 | Step 40 | Step 60 | Step 80 | Step 100 |
|---|---:|---:|---:|---:|---:|
| Training loss | 5.4194 | 4.9846 | 4.7745 | 4.6513 | 4.5733 |
| Validation loss | 5.4897 | 5.1730 | 5.0219 | 4.9285 | 4.9012 |
| Validation perplexity | 242.19 | 176.44 | 151.70 | 138.18 | 134.46 |

Validation loss declined by **0.5885** from the first recorded evaluation at step 20 to step 100, while validation perplexity declined from **242.19** to **134.46**. The result demonstrates that the model is learning statistical structure from this small local corpus. A short BPE-checkpoint generation test completed successfully, but its output remains mostly incoherent, as expected for a sub-million-parameter model trained for only 100 CPU steps on a tiny corpus.

## Commands

```bash
PYTHONPATH=. python3 scripts/train_bpe_tokenizer.py \
  --input datasets/technical_toy_train.txt \
  --output artifacts/tokenizers/technical-smoke-bpe-v1.json \
  --vocab-size 384 \
  --name raiq-technical-bpe \
  --version smoke-v1 \
  --corpus-id raiq-technical-smoke-v1

PYTHONPATH=. python3 -m raiq.training.train \
  --config configs/tiny_bpe.yaml \
  --run-name tiny-bpe-v1

PYTHONPATH=. python3 -m raiq.inference.generate \
  --checkpoint artifacts/tiny-bpe-v1/checkpoint_last.pt \
  --prompt 'RAIQ Core model' \
  --max-new-tokens 32
```

## Current limitations and next requirement

The current corpus is intentionally too small and too narrow for capability claims. The BPE trainer prioritizes transparent reproducibility rather than high-throughput production-scale tokenization. The no-GPU baseline can validate the engineering path but cannot train or evaluate RAIQ-200M v1.

The next model milestone requires a legally documented technical corpus, corpus-quality and contamination controls, a specialist tokenizer evaluation on code/mathematics/units/formulae, accelerator-backed 20M training, and held-out Code, Neural, Chem, and reasoning benchmarks. Only after those requirements are satisfied should the project progress from RAIQ-Tiny-BPE to RAIQ-20M and eventually RAIQ-200M v1.
