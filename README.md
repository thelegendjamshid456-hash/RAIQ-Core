# RAIQ Core

**RAIQ Core** is the transparent model-runtime foundation of **RAIQ**, a compact technical AI system. The first research target, **RAIQ-200M v1**, is a decoder-only language model trained from randomly initialized weights by this project. The system is designed to scale through RAIQ-50M, RAIQ-100M, RAIQ-200M, RAIQ-500M, RAIQ-1B, and RAIQ-7B configurations without tying the implementation to a single model size.

> This repository does not wrap or rename a pretrained external language model. RAIQ Core implements its own tokenizer, model architecture, training path, checkpoints, and inference runtime.

## Current milestone

The active milestone is a **tested tiny-model foundation**. It provides the same core interfaces intended for larger configurations: YAML architecture configurations, automatic parameter counting, a decoder-only Transformer with RMSNorm, RoPE, causal self-attention, SwiGLU, tied embeddings, checkpointing, generation, and reproducible training metadata. The tiny configuration is deliberately sized for CPU smoke tests; it is not a substitute for a trained RAIQ-200M model.

| Configuration | Intended use | Status |
|---|---|---|
| `tiny.yaml` | CPU unit tests and end-to-end smoke training | Implemented baseline |
| `20m.yaml` | First small-scale training experiment | Configuration provided |
| `50m.yaml` | Architecture and data-mixture validation | Configuration provided |
| `100m.yaml` | Scaling and runtime measurement | Configuration provided |
| `200m.yaml` | RAIQ-200M v1 research target | Configuration provided; requires suitable training resources |

## Repository layout

```text
raiq-core/
├── configs/                 # Versioned model/training configurations
├── raiq/
│   ├── core/                # Transformer model and configuration layer
│   ├── tokenizer/           # Custom tokenizer interfaces and implementation
│   ├── data/                # Data manifests and loaders
│   ├── training/            # Training loop, checkpoints, experiment records
│   ├── inference/           # Generation and runtime interfaces
│   ├── agent/               # Future RAIQ Agent orchestration layer
│   ├── tools/               # Future constrained execution tools
│   ├── memory/              # Future working and persistent memory
│   ├── rag/                 # Future retrieval-augmented generation
│   ├── verify/              # Future verification and repair checks
│   └── benchmarks/          # Domain and system evaluation suites
├── tests/                   # Deterministic unit and integration tests
├── scripts/                 # Repeatable operational scripts
├── docs/                    # Architecture, data, and experiment documentation
├── datasets/                # Ignored local data artifacts
├── checkpoints/             # Ignored local model checkpoints
└── artifacts/               # Ignored logs and generated experiment artifacts
```

## Local quick start

Install the declared development dependencies, then run the test suite and a CPU-safe smoke-training run.

```bash
python3 -m pip install -e '.[dev]'
pytest
raiq-train --config configs/tiny.yaml --max-steps 20 --run-name tiny-smoke
raiq-generate --checkpoint artifacts/tiny-smoke/checkpoint_last.pt --prompt 'RAIQ Core is' --max-new-tokens 24
```

The initial environment detected for this implementation has no CUDA-capable GPU and limited system memory. Accordingly, the checked-in experiment path is CPU-oriented and intentionally small. Training RAIQ-200M v1 will require a separately documented data and compute budget; it will not be represented as completed until a genuine run and evaluation record exist.

## Engineering commitments

Every experiment records the model configuration, parameter count, tokenizer and dataset identifiers, sequence length, optimizer, learning-rate schedule, batch settings, random seed, precision, hardware context, validation measurements, run duration, and checkpoint lineage. Checkpoints and datasets are never committed to Git. The test suite includes model-shape, causal-masking, gradient, checkpoint-resume, tokenizer round-trip, and end-to-end smoke checks.

Future work will add RAIQ Code, RAIQ Neural, RAIQ Chem, RAIQ Agent, RAIQ Tools, RAIQ Memory, RAIQ RAG, and RAIQ Verify around the validated core. Any supplied policy or prompt material will be treated as a versioned project artifact for later agent-layer review; it does not replace tested model behavior, tool safety controls, or verification.

## License

No open-source license has been selected. All rights are reserved until the repository owner chooses and adds a license.
