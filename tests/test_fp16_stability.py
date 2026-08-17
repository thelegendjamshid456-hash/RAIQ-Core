from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import torch

from raiq.core import ModelConfig, RAIQModel
from raiq.training.checkpoints import load_checkpoint, save_checkpoint
from raiq.training.train import (
    _check_finite_gradients,
    _check_finite_loss,
    _make_grad_scaler,
    run_training,
)


ROOT = Path(__file__).resolve().parents[1]


class FakeScaler:
    def __init__(self, value: float = 128.0) -> None:
        self.value = value
        self.loaded = None

    def state_dict(self):
        return {"scale": self.value}

    def load_state_dict(self, state):
        self.loaded = dict(state)
        self.value = state["scale"]


def tiny_config() -> ModelConfig:
    return ModelConfig(
        name="RAIQ-FP16-Test",
        vocab_size=128,
        max_seq_len=16,
        n_layers=1,
        d_model=16,
        n_heads=2,
        n_kv_heads=2,
        d_ff=32,
    )


def test_scaler_is_cuda_fp16_only() -> None:
    assert _make_grad_scaler(torch.device("cpu"), torch.float16) is None
    assert _make_grad_scaler(torch.device("cpu"), torch.float32) is None


def test_finite_checks_accept_valid_loss_and_gradients() -> None:
    model = RAIQModel(tiny_config())
    output = model(torch.randint(0, 128, (1, 8)), labels=torch.randint(0, 128, (1, 8)))
    assert output.loss is not None
    _check_finite_loss(output.loss, 1)
    output.loss.backward()
    assert _check_finite_gradients(model, 1) > 0.0


def test_finite_checks_raise_with_step_context() -> None:
    with pytest.raises(RuntimeError, match="step 7"):
        _check_finite_loss(torch.tensor(float("nan")), 7)


def test_scaler_state_round_trip_is_backward_compatible(tmp_path) -> None:
    config = tiny_config()
    model = RAIQModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = FakeScaler()
    checkpoint = save_checkpoint(
        tmp_path / "scaled.pt",
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        step=4,
        run_config={"model": config.to_dict()},
        metadata={"test": True},
    )
    restored = RAIQModel(config)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    restored_scaler = FakeScaler(1.0)
    payload = load_checkpoint(
        checkpoint,
        model=restored,
        optimizer=restored_optimizer,
        scaler=restored_scaler,
    )
    assert payload["scaler_state"] == {"scale": 128.0}
    assert restored_scaler.loaded == {"scale": 128.0}


def test_one_hundred_optimizer_steps_keep_loss_and_gradients_finite() -> None:
    torch.manual_seed(23)
    model = RAIQModel(tiny_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    for step in range(1, 101):
        optimizer.zero_grad(set_to_none=True)
        tokens = torch.randint(0, 128, (1, 8))
        output = model(tokens, labels=tokens.roll(-1, dims=1))
        assert output.loss is not None and torch.isfinite(output.loss)
        output.loss.backward()
        gradient_norm = _check_finite_gradients(model, step)
        assert torch.isfinite(torch.tensor(gradient_norm))
        pre_clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        clipped_gradient_norm = _check_finite_gradients(model, step)
        assert torch.isfinite(torch.as_tensor(pre_clip_norm))
        assert clipped_gradient_norm <= 1.0 + 1e-5
        optimizer.step()


def test_cpu_run_persists_token_throughput_metadata(tmp_path) -> None:
    config_path = tmp_path / "tiny_cpu.yaml"
    config_text = (ROOT / "configs/tiny.yaml").read_text(encoding="utf-8")
    config_text = config_text.replace(
        "train_path: datasets/technical_toy_train.txt",
        f"train_path: {ROOT / 'datasets/technical_toy_train.txt'}",
    )
    config_text = config_text.replace(
        "validation_path: datasets/technical_toy_validation.txt",
        f"validation_path: {ROOT / 'datasets/technical_toy_validation.txt'}",
    )
    config_path.write_text(config_text, encoding="utf-8")
    checkpoint = run_training(
        argparse.Namespace(
            config=str(config_path),
            run_name="cpu-telemetry",
            output_dir=str(tmp_path / "artifacts"),
            max_steps=1,
            resume=None,
        )
    )
    metadata = json.loads((checkpoint.parent / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["tokens_processed_per_rank"] == 512
    assert metadata["global_tokens_processed"] == 512
    assert metadata["tokens_per_second_per_rank"] > 0.0
    assert metadata["estimated_global_tokens_per_second"] > 0.0
    assert "cuda_peak_memory_allocated_bytes" not in metadata

    resumed_checkpoint = run_training(
        argparse.Namespace(
            config=str(config_path),
            run_name="cpu-telemetry",
            output_dir=str(tmp_path / "artifacts"),
            max_steps=2,
            resume=str(checkpoint),
        )
    )
    history = json.loads((resumed_checkpoint.parent / "metrics.json").read_text(encoding="utf-8"))
    resumed_metadata = json.loads(
        (resumed_checkpoint.parent / "metadata.json").read_text(encoding="utf-8")
    )
    assert [record["step"] for record in history] == [1, 2]
    assert resumed_metadata["start_step"] == 1
    assert resumed_metadata["optimizer_steps_completed"] == 1
    assert resumed_metadata["resume_checkpoint"] == str(checkpoint.resolve())
    assert resumed_metadata["metrics_history_status"] == "restored"

    (resumed_checkpoint.parent / "metrics.json").unlink()
    recovered_checkpoint = run_training(
        argparse.Namespace(
            config=str(config_path),
            run_name="cpu-telemetry",
            output_dir=str(tmp_path / "artifacts"),
            max_steps=3,
            resume=str(resumed_checkpoint),
        )
    )
    recovered_history = json.loads(
        (recovered_checkpoint.parent / "metrics.json").read_text(encoding="utf-8")
    )
    recovered_metadata = json.loads(
        (recovered_checkpoint.parent / "metadata.json").read_text(encoding="utf-8")
    )
    assert [record["step"] for record in recovered_history] == [3]
    assert recovered_metadata["start_step"] == 2
    assert recovered_metadata["metrics_history_status"] == "missing_before_resume"
