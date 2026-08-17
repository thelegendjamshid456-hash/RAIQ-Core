"""Command-line training entry point for reproducible RAIQ Core experiments."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Iterator

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from raiq.core import RAIQModel, load_config
from raiq.data.manifest import verify_corpus_manifest
from raiq.data.text_dataset import TextBlockDataset
from raiq.tokenizer.byte_tokenizer import ByteTokenizer
from raiq.tokenizer.loader import load_tokenizer
from raiq.training.checkpoints import load_checkpoint, save_checkpoint
from raiq.training.distributed import cleanup_distributed, initialize_distributed
from raiq.training.loaders import build_training_loader
from raiq.training.utils import resolve_device, resolve_dtype, set_seed, warmup_cosine_lr


def build_optimizer(model: RAIQModel, learning_rate: float, weight_decay: float) -> AdamW:
    """Apply weight decay to matrices while leaving norms and scalar parameters unregularized."""

    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if parameter.ndim < 2 or name.endswith("norm.weight") else decay).append(parameter)
    return AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.95),
    )


def _attach_finite_diagnostics(model: torch.nn.Module) -> list[torch.utils.hooks.RemovableHandle]:
    """Attach leaf-module hooks that report the first non-finite tensor and component."""

    handles: list[torch.utils.hooks.RemovableHandle] = []

    def tensors(value: object) -> list[Tensor]:
        if isinstance(value, Tensor):
            return [value]
        if isinstance(value, (tuple, list)):
            return [item for nested in value for item in tensors(nested)]
        return []

    def forward_hook(module: torch.nn.Module, _inputs: tuple[object, ...], output: object) -> None:
        for tensor in tensors(output):
            if not torch.isfinite(tensor.detach()).all():
                raise RuntimeError(
                    f"non-finite forward output in {module.__class__.__name__}"
                )

    def backward_hook(
        module: torch.nn.Module,
        grad_input: tuple[Tensor | None, ...],
        grad_output: tuple[Tensor | None, ...],
    ) -> None:
        for tensor in (*grad_input, *grad_output):
            if tensor is not None and not torch.isfinite(tensor.detach()).all():
                raise RuntimeError(
                    f"non-finite backward gradient in {module.__class__.__name__}"
                )

    for module in model.modules():
        if not any(module.children()):
            handles.append(module.register_forward_hook(forward_hook))
            handles.append(module.register_full_backward_hook(backward_hook))
    return handles


def _make_grad_scaler(device: torch.device, dtype: torch.dtype) -> torch.amp.GradScaler | None:
    """Create a scaler only for CUDA FP16; keep FP32 and BF16 paths unscaled."""

    if device.type != "cuda" or dtype != torch.float16:
        return None
    try:
        return torch.amp.GradScaler(
            "cuda", enabled=True, init_scale=1024.0, growth_interval=2000, backoff_factor=0.5
        )
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(
            enabled=True, init_scale=1024.0, growth_interval=2000, backoff_factor=0.5
        )


def _check_finite_loss(loss: Tensor, step: int) -> None:
    if not torch.isfinite(loss.detach()).all():
        raise RuntimeError(f"non-finite training loss at step {step}: {float(loss.detach())}")


def _check_finite_gradients(model: torch.nn.Module, step: int) -> float:
    total_squared = 0.0
    seen = False
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        seen = True
        gradient = parameter.grad.detach()
        if not torch.isfinite(gradient).all():
            raise RuntimeError(f"non-finite gradient at step {step}")
        total_squared += float(torch.sum(gradient.float() * gradient.float()).cpu())
    if not seen:
        raise RuntimeError(f"no gradients produced at step {step}")
    return math.sqrt(total_squared)


def cycle(loader: DataLoader[tuple[Tensor, Tensor]]) -> Iterator[tuple[Tensor, Tensor]]:
    """Repeat a deterministic data loader indefinitely."""

    while True:
        yield from loader


@torch.inference_mode()
def evaluate(
    model: RAIQModel,
    loader: DataLoader[tuple[Tensor, Tensor]],
    device: torch.device,
    *,
    max_batches: int = 8,
) -> float:
    """Return an average next-token loss across a bounded validation sample."""

    model.eval()
    losses: list[float] = []
    for batch_index, (inputs, labels) in enumerate(loader):
        if batch_index >= max_batches:
            break
        output = model(inputs.to(device), labels=labels.to(device))
        if output.loss is None:
            raise RuntimeError("evaluation forward pass did not return a loss")
        losses.append(float(output.loss.detach().cpu()))
    if not losses:
        raise RuntimeError("validation loader yielded no batches")
    return sum(losses) / len(losses)


def run_training(args: argparse.Namespace) -> Path:
    """Execute a complete, versioned RAIQ Core training run."""

    run_config = load_config(args.config)
    training = run_config.training
    context = initialize_distributed(training.device)
    set_seed(training.seed + context.rank)
    device = context.device if context.enabled else resolve_device(training.device)
    dtype = resolve_dtype(training.dtype, device)
    max_steps = args.max_steps if args.max_steps is not None else training.max_steps
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = (
        load_tokenizer(run_config.data.tokenizer_path)
        if run_config.data.tokenizer_path
        else ByteTokenizer()
    )
    tokenizer_path = tokenizer.save(output_dir / "tokenizer.json")
    if tokenizer.vocab_size > run_config.model.vocab_size:
        raise ValueError("configured model vocabulary is smaller than the selected tokenizer")
    if not run_config.data.train_path or not run_config.data.validation_path:
        raise ValueError("training requires data.train_path and data.validation_path")
    corpus_manifest = None
    if run_config.data.corpus_manifest_path:
        corpus_manifest = verify_corpus_manifest(run_config.data.corpus_manifest_path)

    train_text = Path(run_config.data.train_path).read_text(encoding="utf-8")
    train_token_count = len(tokenizer.encode(train_text, add_bos=True, add_eos=True))
    tokenizer_metrics = {
        "train_bytes": len(train_text.encode("utf-8")),
        "train_tokens": train_token_count,
        "train_tokens_per_byte": train_token_count / max(1, len(train_text.encode("utf-8"))),
    }
    train_dataset = TextBlockDataset(run_config.data.train_path, tokenizer, run_config.model.max_seq_len)
    validation_dataset = TextBlockDataset(run_config.data.validation_path, tokenizer, run_config.model.max_seq_len)
    train_loader, train_sampler = build_training_loader(
        train_dataset,
        batch_size=training.batch_size,
        context=context,
        seed=training.seed,
    )
    validation_loader = DataLoader(validation_dataset, batch_size=training.batch_size, shuffle=False)

    # Keep trainable master parameters in FP32. CUDA autocast controls forward precision.
    model = RAIQModel(run_config.model).to(device=device, dtype=torch.float32)
    diagnostic_handles = (
        _attach_finite_diagnostics(model) if training.finite_diagnostics else []
    )
    if context.enabled:
        model = DistributedDataParallel(
            model,
            device_ids=[context.local_rank] if device.type == "cuda" else None,
            bucket_cap_mb=64,
        )
    base_model = model.module if isinstance(model, DistributedDataParallel) else model
    optimizer = build_optimizer(model, training.learning_rate, training.weight_decay)
    scaler = _make_grad_scaler(device, dtype)
    start_step = 0
    if args.resume is not None:
        payload = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            map_location=device,
            restore_rng=True,
            scaler=scaler,
        )
        start_step = int(payload["step"])

    history: list[dict[str, float | int]] = []
    metrics_path = output_dir / "metrics.json"
    if start_step > 0:
        if not metrics_path.is_file():
            raise FileNotFoundError(
                f"cannot resume at step {start_step} without existing metrics: {metrics_path}"
            )
        loaded_history = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not isinstance(loaded_history, list) or not loaded_history:
            raise ValueError(f"resume metrics must be a non-empty list: {metrics_path}")
        final_record = loaded_history[-1]
        if int(final_record.get("step", -1)) != start_step:
            raise ValueError(
                f"resume metrics end at step {final_record.get('step')}, "
                f"but checkpoint is step {start_step}"
            )
        history = loaded_history

    metadata = {
        "model": base_model.metadata(),
        "dataset": run_config.data.to_dict(),
        "tokenizer": tokenizer.to_dict(),
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_metrics": tokenizer_metrics,
        "corpus_manifest": {
            "path": run_config.data.corpus_manifest_path,
            "corpus_id": None if corpus_manifest is None else corpus_manifest["corpus_id"],
        },
        "seed": training.seed,
        "device": str(device),
        "dtype": training.dtype,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "started_at_unix": time.time(),
        "resume_checkpoint": None if args.resume is None else str(Path(args.resume).resolve()),
    }
    if context.is_primary:
        (output_dir / "run_config.json").write_text(
            json.dumps(run_config.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    train_iterator = cycle(train_loader)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    tokens_processed = 0
    started = time.perf_counter()
    for step in range(start_step, max_steps):
        if train_sampler is not None:
            train_sampler.set_epoch(step)
        lr = warmup_cosine_lr(
            step,
            max_steps=max_steps,
            warmup_steps=training.warmup_steps,
            max_lr=training.learning_rate,
            min_lr=training.min_learning_rate,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        accumulated_loss = 0.0
        for _ in range(training.grad_accumulation_steps):
            inputs, labels = next(train_iterator)
            inputs, labels = inputs.to(device), labels.to(device)
            tokens_processed += inputs.numel()
            autocast_context = (
                torch.autocast(device_type=device.type, dtype=dtype)
                if device.type == "cuda" and dtype != torch.float32
                else nullcontext()
            )
            with autocast_context:
                output = model(inputs, labels=labels)
                if output.loss is None:
                    raise RuntimeError("training forward pass did not return a loss")
                loss = output.loss
                _check_finite_loss(loss, step + 1)
                scaled_loss = loss / training.grad_accumulation_steps
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            accumulated_loss += float(loss.detach().cpu())

        if scaler is not None:
            scaler.unscale_(optimizer)
        gradient_norm = _check_finite_gradients(model, step + 1)
        pre_clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training.grad_clip_norm)
        if not torch.isfinite(torch.as_tensor(pre_clip_norm)).all():
            raise RuntimeError(f"non-finite pre-clip gradient norm at step {step + 1}")
        clipped_gradient_norm = _check_finite_gradients(model, step + 1)
        clip_tolerance = max(1e-6, training.grad_clip_norm * 1e-5)
        if clipped_gradient_norm > training.grad_clip_norm + clip_tolerance:
            raise RuntimeError(
                f"gradient clipping exceeded its threshold at step {step + 1}: "
                f"{clipped_gradient_norm} > {training.grad_clip_norm}"
            )
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        completed_step = step + 1

        record: dict[str, float | int] = {
            "step": completed_step,
            "train_loss": accumulated_loss / training.grad_accumulation_steps,
            "learning_rate": lr,
            "gradient_norm": gradient_norm,
            "clipped_gradient_norm": clipped_gradient_norm,
        }
        if scaler is not None:
            record["grad_scaler_scale"] = float(scaler.get_scale())
        if completed_step % training.eval_interval == 0 or completed_step == max_steps:
            validation_loss = evaluate(model, validation_loader, device)
            record["validation_loss"] = validation_loss
            record["validation_perplexity"] = math.exp(min(validation_loss, 20.0))
            model.train()
        history.append(record)
        if context.is_primary and (completed_step % training.log_interval == 0 or completed_step == 1):
            print(json.dumps(record, sort_keys=True), flush=True)
        if context.is_primary and completed_step % training.save_interval == 0:
            save_checkpoint(
                output_dir / f"checkpoint_step_{completed_step}.pt",
                model=model,
                optimizer=optimizer,
                step=completed_step,
                run_config=run_config.to_dict(),
                metadata=metadata,
                scaler=scaler,
            )

    elapsed_seconds = time.perf_counter() - started
    metadata["completed_at_unix"] = time.time()
    metadata["elapsed_seconds"] = elapsed_seconds
    metadata["max_steps"] = max_steps
    metadata["start_step"] = start_step
    metadata["optimizer_steps_completed"] = max_steps - start_step
    metadata["world_size"] = context.world_size
    metadata["rank"] = context.rank
    metadata["tokens_processed_per_rank"] = tokens_processed
    metadata["tokens_per_second_per_rank"] = tokens_processed / max(elapsed_seconds, 1e-12)
    metadata["global_tokens_processed"] = tokens_processed * context.world_size
    metadata["estimated_global_tokens_per_second"] = (
        tokens_processed * context.world_size / max(elapsed_seconds, 1e-12)
    )
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        metadata["cuda_device_name"] = torch.cuda.get_device_name(device)
        metadata["cuda_total_memory_bytes"] = properties.total_memory
        metadata["cuda_peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        metadata["cuda_peak_memory_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    if context.is_primary:
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (output_dir / "metrics.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
        result = save_checkpoint(
            output_dir / "checkpoint_last.pt",
            model=base_model,
            optimizer=optimizer,
            step=max_steps,
            run_config=run_config.to_dict(),
            metadata=metadata,
            scaler=scaler,
        )
    else:
        result = output_dir / "checkpoint_last.pt"
    for handle in diagnostic_handles:
        handle.remove()
    cleanup_distributed(context)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a RAIQ Core decoder-only language model")
    parser.add_argument("--config", required=True, help="Path to a RAIQ YAML configuration")
    parser.add_argument("--run-name", required=True, help="Name for this experiment under the output directory")
    parser.add_argument("--output-dir", default="artifacts", help="Root directory for generated experiment artifacts")
    parser.add_argument("--max-steps", type=int, default=None, help="Override configured maximum optimizer steps")
    parser.add_argument("--resume", default=None, help="Optional checkpoint from which to continue")
    return parser


def main() -> None:
    checkpoint = run_training(build_parser().parse_args())
    print(f"Saved checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
