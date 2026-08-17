# RAIQ-200M T4 Smoke-v2 Change Review

**Status:** Configuration and notebook update prepared and tested locally. **No Colab training command has been run.**

## Scope

This update creates a new, isolated `smoke-v2` experiment. It does not alter the existing T4 run, any existing checkpoint directory, or `configs/200m_production.yaml`.

## Exact Configuration Changes

The new file is `configs/200m_t4_smoke_v2.yaml`. The model architecture, tokenizer/data paths, FP16 setting, batch size, accumulation, AdamW settings, finite diagnostics, validation interval, save interval, and gradient clipping remain aligned with the existing smoke configuration.

| Setting | Existing smoke value | Fresh smoke-v2 value |
|---|---:|---:|
| `model.name` | `RAIQ-200M-v1-T4-smoke` | `RAIQ-200M-v1-T4-smoke-v2` |
| `training.max_steps` | `100` | `1000` |
| `training.learning_rate` | `0.00001` | `0.0003` |
| `training.min_learning_rate` | `0.000001` | `0.00003` |
| `training.warmup_steps` | `32` | `250` |
| `training.grad_clip_norm` | `1.0` | `1.0` |
| `training.finite_diagnostics` | `true` | `true` |

The existing trainer already applies a linear warmup followed by cosine decay. With the v2 settings, it begins at a near-zero first-step rate, reaches `0.0003` at the end of the 250-step warmup, and then follows cosine decay toward `0.00003`.

```yaml
training:
  seed: 20260817
  device: cuda
  dtype: float16
  batch_size: 1
  grad_accumulation_steps: 32
  max_steps: 1000
  learning_rate: 0.0003
  min_learning_rate: 0.00003
  warmup_steps: 250
  weight_decay: 0.1
  grad_clip_norm: 1.0
  eval_interval: 4
  save_interval: 4
  log_interval: 1
  finite_diagnostics: true
```

## Initialization Confirmation

The model implementation initializes token embeddings and ordinary linear projections using:

```python
nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

Residual attention-output and MLP-down projections are deliberately initialized at `0.02 / sqrt(2 * n_layers)`. This existing depth-scaled residual exception is preserved because it is a stability control, not an accidental deviation.

## Exact Fresh-Run Notebook Command

The updated notebook now uses the v2 configuration and a new checkpoint path. It intentionally contains **no** `--resume` argument.

```bash
python -m raiq.training.train \
  --config configs/200m_t4_smoke_v2.yaml \
  --run-name raiq-200m-t4-smoke-v2 \
  --output-dir /content/drive/MyDrive/RAIQ/checkpoints \
  --max-steps 1000
```

This command starts at step 0 and writes to:

```text
/content/drive/MyDrive/RAIQ/checkpoints/raiq-200m-t4-smoke-v2/
```

It cannot overwrite the earlier run unless that new directory name is manually reused.

## Telemetry and Acceptance Checks

The notebook’s evidence cell now asserts that every **measured post-clip gradient norm** is at or below `1.00001`. It reports initial, best, and final validation perplexity, requiring the best validation perplexity to improve on the initial validation perplexity. It deliberately does not demand an artificial point-by-point monotonic validation curve.

## Verification Completed Before Training

| Check | Result |
|---|---|
| Smoke-v2 configuration, warmup, clipping, and finite-diagnostics regression tests | Passed |
| Initialization regression test | Passed |
| Full project test suite | **26 passed** |
| Notebook JSON validation | Passed |
| Existing model-evidence validator | Passed |

## Explicit Non-Changes

The update does not change the 190M architecture, tokenizer, production configuration, distributed support, optimizer type, `grad_clip_norm: 1.0`, finite loss/gradient checks, validation, checkpointing, or resume mechanics. It does not launch training or claim that `3e-4` is stable before actual T4 evidence exists.
