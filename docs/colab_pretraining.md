# RAIQ-200M Google Colab T4 Smoke Training

This runbook prepares a **short smoke experiment** on a single Google Colab NVIDIA Tesla T4. It does not start a long run and does not weaken `configs/200m_production.yaml` or `scripts/production_preflight.py`.

The local dataset inspection found two UTF-8 plain-text files: `technical_toy_train.txt` (13,860 bytes) and `technical_toy_validation.txt` (14,547 bytes). They are already split, raw text, and directly compatible with `TextBlockDataset`. They are an engineering smoke corpus, not a production corpus.

> A successful T4 smoke run proves that this 200M architecture can execute a short forward/backward/checkpoint cycle in that environment. It does not prove production-pretraining readiness, useful capability, or parity with an external model.

## 1. Prepare the Drive directory locally

Create this structure in Google Drive:

```text
MyDrive/RAIQ/
├── datasets/
│   ├── technical_toy_train.txt
│   └── technical_toy_validation.txt
├── manifests/
├── tokenizer/
├── checkpoints/
├── logs/
└── experiments/
```

Upload the two files from the repository's `datasets/` directory. Do not upload benchmark evaluation data as training data. After uploading, preserve the exact bytes so the manifest hashes remain valid.

## 2. Start a new Colab runtime

In Colab, select **Runtime → Change runtime type → T4 GPU**. Then run:

```python
from google.colab import drive
drive.mount('/content/drive')

import os, platform, subprocess, sys, torch
print('Python:', sys.version)
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is required for the T4 smoke run')
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
print('CUDA runtime:', torch.version.cuda)
```

## 3. Clone the code and install without replacing Colab's CUDA PyTorch

```bash
%cd /content
!rm -rf RAIQ-Core
!git clone https://github.com/thelegendjamshid456-hash/RAIQ-Core.git
%cd /content/RAIQ-Core
!python -m pip install -e . --no-deps
```

The `--no-deps` flag avoids replacing the CUDA-enabled PyTorch already provided by the Colab runtime. If the runtime lacks a declared non-PyTorch dependency, install only that missing dependency rather than reinstalling PyTorch.

## 4. Verify the Drive dataset and create the Drive manifest

```bash
%cd /content/RAIQ-Core
!mkdir -p /content/drive/MyDrive/RAIQ/manifests /content/drive/MyDrive/RAIQ/tokenizer
!python scripts/create_text_manifest.py \
  --train /content/drive/MyDrive/RAIQ/datasets/technical_toy_train.txt \
  --validation /content/drive/MyDrive/RAIQ/datasets/technical_toy_validation.txt \
  --output /content/drive/MyDrive/RAIQ/manifests/technical_smoke_v1.json \
  --corpus-id raiq-technical-smoke-v1 \
  --dataset-version local-inspection-v1 \
  --source-reference gdrive:MyDrive/RAIQ/datasets
```

Verify the source bytes against the expected inspected values:

```python
from pathlib import Path
assert Path('/content/drive/MyDrive/RAIQ/datasets/technical_toy_train.txt').stat().st_size == 13860
assert Path('/content/drive/MyDrive/RAIQ/datasets/technical_toy_validation.txt').stat().st_size == 14547
print('dataset byte checks: passed')
```

## 5. Train and verify the BPE tokenizer

```bash
%cd /content/RAIQ-Core
!python scripts/train_bpe_tokenizer.py \
  --input /content/drive/MyDrive/RAIQ/datasets/technical_toy_train.txt \
  --input /content/drive/MyDrive/RAIQ/datasets/technical_toy_validation.txt \
  --output /content/drive/MyDrive/RAIQ/tokenizer/raiq_code_bpe.json \
  --vocab-size 32768 \
  --name raiq-bpe \
  --version colab-trained-v1 \
  --corpus-id raiq-technical-smoke-v1 \
  --min-pair-frequency 2
```

The tokenizer must round-trip technical Unicode and have a vocabulary no larger than 32,768 before training starts.

## 6. Run data and model smoke checks

```python
import json, torch
from pathlib import Path
from raiq.core import load_config, RAIQModel
from raiq.data.manifest import verify_corpus_manifest
from raiq.data.text_dataset import TextBlockDataset
from raiq.tokenizer.loader import load_tokenizer

cfg = load_config('configs/200m_t4_smoke.yaml')
manifest = verify_corpus_manifest('/content/drive/MyDrive/RAIQ/manifests/technical_smoke_v1.json')
tok = load_tokenizer('/content/drive/MyDrive/RAIQ/tokenizer/raiq_code_bpe.json')
assert tok.vocab_size <= cfg.model.vocab_size
sample = tok.decode(tok.encode('ΔH = m·Cp·ΔT; def solve(x): return x + 1'))
print('tokenizer sample:', sample)
model = RAIQModel(cfg.model).cuda().half()
inputs = torch.randint(0, tok.vocab_size, (1, cfg.model.max_seq_len), device='cuda')
labels = inputs.roll(-1, dims=1)
with torch.autocast('cuda', dtype=torch.float16):
    out = model(inputs, labels=labels)
assert out.loss is not None and torch.isfinite(out.loss)
out.loss.backward()
print('200M forward/backward: passed; loss=', float(out.loss))
del model, inputs, labels, out
torch.cuda.empty_cache()
```

## 7. Run the short 200M T4 training smoke test

The configuration uses sequence length 2,048, micro-batch 1, gradient accumulation 32, mixed precision, gradient checkpointing, and only 8 optimizer steps. The output is written to Drive, not only to `/content`.

```bash
%cd /content/RAIQ-Core
!mkdir -p /content/drive/MyDrive/RAIQ/checkpoints /content/drive/MyDrive/RAIQ/experiments
!python -m raiq.training.train \
  --config configs/200m_t4_smoke.yaml \
  --run-name raiq-200m-t4-smoke-v1 \
  --output-dir /content/drive/MyDrive/RAIQ/checkpoints \
  --max-steps 8
```

Record actual GPU memory and throughput from the Colab output. Do not invent these values in documentation. The expected proof fields are finite loss, decreasing loss across evaluation points, checkpoint files on Drive, and no CUDA out-of-memory failure.

## 8. Test checkpoint resume after a simulated restart

Restart or reconnect the runtime, remount Drive, clone the repository again, and run:

```bash
%cd /content/RAIQ-Core
!python -m raiq.training.train \
  --config configs/200m_t4_smoke.yaml \
  --run-name raiq-200m-t4-smoke-resumed-v1 \
  --output-dir /content/drive/MyDrive/RAIQ/checkpoints \
  --max-steps 12 \
  --resume /content/drive/MyDrive/RAIQ/checkpoints/raiq-200m-t4-smoke-v1/checkpoint_last.pt
```

The resume command must be used as written, with no source edits. Confirm that the resumed run starts after the saved step and writes a new final checkpoint to Drive.

## 9. Continue a longer experiment only after the smoke gates pass

A longer run must be a separately named experiment. First record the T4's observed peak VRAM, tokens per second, loss trajectory, and checkpoint-resume evidence. Do not treat a T4 smoke run as production pretraining: the production preflight remains intentionally fail-closed for the required multi-GPU environment, approved corpus, storage, and governance evidence.
