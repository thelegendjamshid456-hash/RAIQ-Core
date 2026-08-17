# RAIQ Core Architecture

RAIQ Core is a configurable, from-scratch decoder-only Transformer runtime. It is the foundation-model layer of the wider RAIQ system and deliberately separates learned model behavior from the future RAIQ Agent, RAIQ Tools, RAIQ Memory, RAIQ RAG, and RAIQ Verify layers.

```text
Token IDs
    │
Token Embeddings
    │
┌──────────────────────────────────────────────┐
│ Repeated RAIQ Decoder Block                  │
│                                              │
│ RMSNorm → causal attention → residual add    │
│            └─ RoPE + grouped KV cache        │
│ RMSNorm → SwiGLU MLP → residual add          │
└──────────────────────────────────────────────┘
    │
RMSNorm → tied language-model head → logits
```

## Implemented model features

The core uses a causal decoder stack with bias-free linear projections, RMSNorm pre-normalization, RoPE positional encoding, SwiGLU feed-forward networks, residual connections, optional tied input/output embeddings, PyTorch scaled-dot-product attention, grouped key/value head support, and an incremental generation cache. A configuration loader validates dimensional constraints before model construction. Each instantiated model can report its exact parameter count and architecture metadata.

| Feature | Implementation status | Notes |
|---|---|---|
| Decoder-only causal Transformer | Implemented | `RAIQModel` and `Transformer` |
| RMSNorm | Implemented | Pre-attention, pre-MLP, and final norm |
| RoPE | Implemented | Position offset supports cached decoding |
| SwiGLU | Implemented | Configurable intermediate dimension |
| Grouped KV heads | Implemented | Validated `n_heads / n_kv_heads` grouping |
| KV cache | Implemented | Used in the generation path |
| Gradient checkpointing | Implemented | Enabled by configuration during training when cache is off |
| Flash/SDPA path | Available where PyTorch selects it | Uses PyTorch scaled-dot-product attention |
| BF16/FP16 | Configurable | Requires a compatible target environment; not demonstrated locally |
| Quantization | Not yet implemented | Scheduled after a validated trained-model baseline |
| REST API and batching | Not yet implemented | Future RAIQ Core runtime milestone |

## Scaling configurations

The same source code supports five checked-in configurations. The configurations are research starting points, not performance guarantees. Actual effectiveness depends on legal data quality, a token budget, the optimization schedule, and measured evaluation results.

| Configuration | Layers | Model dimension | Heads | Context | Vocabulary | Purpose |
|---|---:|---:|---:|---:|---:|---|
| RAIQ-Tiny | 4 | 128 | 4 | 128 | 512 | CPU-safe tests and smoke runs |
| RAIQ-20M | 12 | 384 | 6 | 1,024 | 32,768 | Initial accelerator-backed training |
| RAIQ-50M | 16 | 512 | 8 | 2,048 | 32,768 | Data and architecture validation |
| RAIQ-100M | 18 | 640 | 10 | 4,096 | 32,768 | Scaling and runtime measurement |
| RAIQ-200M-v1 | 20 | 768 | 12 | 8,192 | 32,768 | Initial foundation-model research target |

The configured `RAIQ-200M-v1` architecture instantiates **190,348,032 trainable parameters** with tied embeddings. It is therefore a legitimate approximately-200M target; it has not been trained in the current CPU-only baseline environment.

## Training and reproducibility

The training command persists the validated YAML configuration, model metadata and exact parameter count, dataset/tokenizer identifiers, Python/PyTorch/platform context, random seed, metrics, and atomic checkpoints containing the model and optimizer states. The initial schedule applies linear warmup followed by cosine decay, AdamW with decayed matrix parameters, gradient clipping, periodic validation, and resumable checkpoints.

The smoke path intentionally uses a transparent byte-level UTF-8 tokenizer with RAIQ special tokens. It provides lossless handling for code, units, formulas, and Unicode symbols at the byte level. It is not the final trained 32K–50K technical BPE/Unigram tokenizer; that work requires a licensed, curated corpus and a dedicated tokenizer evaluation report.

## Safety and system boundaries

No external pretrained language-model weights are loaded by RAIQ Core. The current code does not execute shell commands, retrieve web content, store long-term user memory, or autonomously act. These behaviors belong to future, separately constrained system layers and must be backed by explicit allow-lists, audit records, source controls, and verification before they are enabled.
