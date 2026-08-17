# RAIQ-200M T4 Interrupted-Run Recovery Runbook

**Author:** Manus AI  
**Status:** Prepared from the submitted T4 training trace through step 175; not a claim of completed 1,000-step training.

## Purpose

This runbook resumes the saved RAIQ-200M T4 state toward **1,000 total optimizer steps** without restarting from step 1. It keeps the existing 190M-parameter architecture, FP16 GradScaler, finite loss/gradient checks, gradient clipping at `1.0`, validation, checkpointing, and production gates intact.

> The submitted trace contains finite values through step 175. Its learning-rate sequence matches a 1,000-step warmup-cosine horizon, so resuming with `--max-steps 1000` preserves the intended schedule.

## Codebase hardening already applied

| Git commit | Recovery improvement |
|---|---|
| `e6bf680` | Correctly records the post-clipping gradient norm. |
| `941ad64` | Records GPU identity, peak memory, token counts, and throughput after a completed run. |
| `96b8b58` | Preserves matching metric history across checkpoint resumes. |
| `5971468` | Adds read-only checkpoint inspection. |
| `a866c6c` | Atomically persists metrics beside periodic checkpoints and permits recovery when a previous interrupted run did not persist `metrics.json`. |

The codebase passed **24 tests** and the existing engineering evidence validator after the durable-recovery change. This is engineering evidence, not a model-capability claim.

## Colab recovery steps

Open the same Colab notebook with a T4 runtime and execute the following cells in order. Do not alter the model architecture, gradient clipping threshold, tokenizer, finite checks, validation, or production configuration.

### 1. Update the repository

```python
%cd /content/RAIQ-Core
!git pull --ff-only origin main
!git rev-parse --short HEAD
```

The reported revision must be `a866c6c` or a newer commit.

### 2. Locate the latest periodic checkpoint

```python
from pathlib import Path
import re

source_run = Path('/content/drive/MyDrive/RAIQ/checkpoints/raiq-200m-t4-gradient-fix-v2')
pattern = re.compile(r'checkpoint_step_(\d+)\.pt$')
candidates = []
for path in source_run.glob('checkpoint_step_*.pt'):
    match = pattern.fullmatch(path.name)
    if match:
        candidates.append((int(match.group(1)), path))

assert candidates, f'No periodic checkpoint found under {source_run}'
step, checkpoint = max(candidates)
print(f'Latest periodic checkpoint: step={step}, path={checkpoint}, bytes={checkpoint.stat().st_size}')
```

The submitted log reached step 175, so the expected latest checkpoint is usually `checkpoint_step_172.pt` because checkpoints are saved every four steps. Use the path found by this cell rather than guessing.

### 3. Inspect the checkpoint without modifying it

```python
recovered_name = 'raiq-200m-t4-recovered-v1'
recovered_run = Path('/content/drive/MyDrive/RAIQ/checkpoints') / recovered_name
recovered_run.mkdir(parents=True, exist_ok=True)
inspection = recovered_run / 'checkpoint_inspection.json'

!python scripts/inspect_checkpoint.py \
  --checkpoint "{checkpoint}" \
  --output "{inspection}"

print(inspection.read_text())
```

A passing report must include a recorded step, nonempty model state, optimizer state, run configuration, RNG state, and GradScaler state for the FP16 CUDA run.

### 4. Resume safely to 1,000 total steps

```python
!python -m raiq.training.train \
  --config configs/200m_t4_smoke.yaml \
  --run-name raiq-200m-t4-recovered-v1 \
  --output-dir /content/drive/MyDrive/RAIQ/checkpoints \
  --max-steps 1000 \
  --resume "{checkpoint}"
```

The recovery implementation starts at the checkpoint's recorded step. If the old run was interrupted before `metrics.json` was persisted, the new run records `metrics_history_status: "missing_before_resume"` rather than inventing missing history. It then persistently records new metric history every checkpoint interval.

### 5. Collect evidence after the run ends

```python
run = Path('/content/drive/MyDrive/RAIQ/checkpoints/raiq-200m-t4-recovered-v1')
print('Checkpoint files:')
for path in sorted(run.glob('checkpoint_step_*.pt'))[-5:]:
    print(path.name, path.stat().st_size)

print('\nFinal metric:')
import json
metrics = json.loads((run / 'metrics.json').read_text())
print(metrics[-1])

print('\nRuntime metadata:')
metadata = json.loads((run / 'metadata.json').read_text())
for key in [
    'start_step', 'optimizer_steps_completed', 'metrics_history_status',
    'cuda_device_name', 'cuda_total_memory_bytes',
    'cuda_peak_memory_allocated_bytes', 'cuda_peak_memory_reserved_bytes',
    'tokens_processed_per_rank', 'tokens_per_second_per_rank',
]:
    print(f'{key}: {metadata.get(key)}')
```

## Evidence boundaries

A successful resumed run provides engineering evidence of checkpoint recovery and extended numerical stability. It does **not** demonstrate coding or reasoning capability. Those claims require real licensed data, sustained pretraining, held-out evaluation, and predefined benchmark thresholds.

## Do not do the following

Do not restart from step 1 if a valid periodic checkpoint exists. Do not increase `gradient_clip_norm` above `1.0`, suppress non-finite checks, skip validation, change the 190M architecture, change the tokenizer merely to improve the smoke result, or launch production pretraining from the toy corpus.
