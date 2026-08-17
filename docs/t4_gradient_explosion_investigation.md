# T4 Gradient-Explosion Investigation

## Observed failure

The first GradScaler fix prevented the original step-2 NaN, but the T4 run later failed at step 22 with a non-finite gradient. The run used a 190,348,032-parameter model, FP16 autocast, GradScaler scale 65,536, learning rate approximately 5e-5, gradient clipping at 1.0, and gradient norms ranging from approximately 1,100 to 2,184 before clipping.

The finite-gradient failure is intentionally preserved as a hard error. No NaN or Inf is allowed to continue into an optimizer update.

## Code-path inspection

The model uses standard pre-norm residual blocks: RMSNorm, causal attention with RoPE, residual addition, RMSNorm, SwiGLU MLP, and a second residual addition. RMSNorm computes the variance in FP32. Attention uses PyTorch scaled dot-product attention. The language-model head is tied to the token embedding by default. Linear and embedding weights use a 0.02 normal initialization, while attention output and MLP down projections use a depth-scaled standard deviation of `0.02 / sqrt(2 * n_layers)`. The optimizer remains AdamW, warmup-cosine scheduling remains unchanged, and gradient clipping remains 1.0.

The evidence does not justify an architectural redesign. The most direct T4-specific contributors are the aggressive early learning-rate ramp and the large initial FP16 scaler scale. The new experiment therefore keeps the architecture and optimizer family unchanged, starts GradScaler at 1,024 rather than 65,536, lowers the T4-only learning rate to `1e-5` with minimum `1e-6`, and extends warmup to 32 steps.

## Diagnostics added

When enabled by `training.finite_diagnostics`, leaf modules receive forward and full-backward finite hooks. The first non-finite activation reports its component class, and the first non-finite backward tensor reports its component class. Loss, unscaled gradients, and clipped gradient norms remain hard finite checks with the optimizer update blocked on failure.

## Local evidence

| Gate | Result |
|---|---|
| Full local suite after changes | **23/23 passed** |
| 100-step finite-gradient regression | **Passed** on compact CPU model |
| Production architecture changed | **No** |
| Production configuration changed | **No** |
| Gradient clipping removed or raised | **No; remains 1.0** |
| Actual T4 100-step run | **Not yet measured** |

The T4 fix is not declared successful until the new configuration completes at least 100 consecutive T4 optimizer steps with finite loss, validation, gradients, and checkpoint-resume evidence.
