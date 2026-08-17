# Tesla T4 FP16 Stability Fix

## Reported failure

The initial RAIQ-200M T4 smoke run produced a finite step-1 loss followed by `NaN` losses. The prior trainer converted the full model to FP16, used CUDA autocast, and did not use gradient scaling.

## Implemented fix

For CUDA FP16 runs, RAIQ now keeps trainable master parameters in FP32 and uses CUDA autocast for forward computation. A `torch.amp.GradScaler("cuda")` is created, with a compatibility fallback for older PyTorch APIs. The loop scales the accumulated loss before backward, unscales gradients before clipping, checks gradient finiteness, clips gradients, calls `scaler.step`, and calls `scaler.update`.

The FP32 path remains an ordinary unscaled training path. BF16 is not used as the T4 solution. The production configuration and production preflight requirements were not changed.

The trainer now checks loss, gradients, and clipped gradient norms for finite values and raises a step-specific `RuntimeError` on non-finite values. Checkpoints optionally persist scaler state and restore it when available; older checkpoints without scaler state remain loadable.

## Local evidence

| Gate | Result |
|---|---|
| Full automated suite after fix | **22/22 passed** |
| Python syntax check | Passed |
| CPU FP32 training smoke | Passed; finite loss and gradient telemetry |
| CPU checkpoint resume | Passed |
| Scaler state checkpoint round trip | Passed with focused test double |
| Production configuration changed | **No** |
| Production preflight requirements changed | **No** |
| Actual Tesla T4 FP16 run | **Not yet measured in this CPU-only environment** |

The local evidence validates the implementation and backward compatibility. It does not prove that the actual T4 run is fixed. The user must rerun the 8-step Colab T4 command and provide finite step-by-step losses, validation loss, gradient norm, GradScaler scale, VRAM, checkpoint, and resume evidence before the T4 issue can be declared fixed.
