"""Produce a pass/fail evidence record for a trained RAIQ checkpoint.

The command intentionally distinguishes an engineering-functional model (E1/E2) from
unproven domain capability (E3+). It exits non-zero when any mandatory engineering gate fails.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from raiq.core.config import ModelConfig
from raiq.core.model import RAIQModel
from raiq.data.manifest import verify_corpus_manifest
from raiq.data.text_dataset import TextBlockDataset
from raiq.tokenizer.loader import load_tokenizer
from raiq.training.checkpoints import load_checkpoint


def _is_finite_tensor(tensor: torch.Tensor | None) -> bool:
    return tensor is not None and bool(torch.isfinite(tensor).all())


def _run_tests(repo_root: Path) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo_root)
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=repo_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "output": (completed.stdout + completed.stderr).strip(),
    }


def _evaluate_loss(model: RAIQModel, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    with torch.inference_mode():
        for inputs, labels in loader:
            output = model(inputs.to(device), labels=labels.to(device))
            if output.loss is None or not _is_finite_tensor(output.loss):
                raise RuntimeError("validation produced a non-finite loss")
            losses.append(float(output.loss.cpu()))
    if not losses:
        raise RuntimeError("validation loader yielded no batches")
    return sum(losses) / len(losses)


def validate(checkpoint_path: Path, repo_root: Path) -> dict[str, Any]:
    """Run all required engineering gates and return a JSON-safe evidence record."""

    raw_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    run_config = raw_checkpoint["run_config"]
    model_config = ModelConfig(**run_config["model"])
    data_config = run_config["data"]
    tokenizer = load_tokenizer(checkpoint_path.parent / "tokenizer.json")
    model = RAIQModel(model_config)
    payload = load_checkpoint(checkpoint_path, model=model, map_location="cpu")

    gates: dict[str, dict[str, Any]] = {}
    gates["unit_suite"] = _run_tests(repo_root)

    try:
        manifest = verify_corpus_manifest(repo_root / data_config["corpus_manifest_path"])
        gates["data_integrity"] = {"passed": True, "corpus_id": manifest["corpus_id"]}
    except Exception as error:  # Evidence must record a failed gate rather than hiding it.
        gates["data_integrity"] = {"passed": False, "error": str(error)}

    technical_text = "H2O at 3.5 kg/s; ΔH = 12 kJ/mol; def f(x): return x**2"
    encoded = tokenizer.encode(technical_text, add_bos=True, add_eos=True)
    round_trip = tokenizer.decode(encoded) == technical_text
    gates["tokenizer_integrity"] = {
        "passed": round_trip and tokenizer.vocab_size <= model_config.vocab_size,
        "round_trip": round_trip,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "model_vocab_size": model_config.vocab_size,
    }

    validation_path = repo_root / data_config["validation_path"]
    dataset = TextBlockDataset(validation_path, tokenizer, model_config.max_seq_len)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    inputs, labels = next(iter(loader))
    model.train()
    model.zero_grad(set_to_none=True)
    training_output = model(inputs, labels=labels)
    finite_loss = _is_finite_tensor(training_output.loss)
    if training_output.loss is not None and finite_loss:
        training_output.loss.backward()
    gradients_finite = all(
        parameter.grad is None or _is_finite_tensor(parameter.grad) for parameter in model.parameters()
    )
    gates["model_numerics"] = {"passed": finite_loss and gradients_finite, "loss": None if training_output.loss is None else float(training_output.loss.detach())}

    clone = RAIQModel(model_config)
    load_checkpoint(checkpoint_path, model=clone, map_location="cpu")
    model.eval()
    clone.eval()
    with torch.inference_mode():
        original_logits = model(inputs).logits
        restored_logits = clone(inputs).logits
    checkpoint_equivalent = bool(torch.allclose(original_logits, restored_logits, rtol=1e-5, atol=1e-6))
    gates["checkpoint_continuity"] = {"passed": checkpoint_equivalent, "checkpoint_step": payload["step"]}

    prompt_ids = torch.tensor([tokenizer.encode("RAIQ", add_bos=True)], dtype=torch.long)
    try:
        generated = clone.generate(prompt_ids, max_new_tokens=4, temperature=1.0, top_k=16)
        inference_ok = generated.size(1) > prompt_ids.size(1)
        gates["inference_continuity"] = {"passed": inference_ok, "generated_token_count": int(generated.size(1))}
    except Exception as error:
        gates["inference_continuity"] = {"passed": False, "error": str(error)}

    metrics_path = checkpoint_path.parent / "metrics.json"
    metadata_path = checkpoint_path.parent / "metadata.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    recorded_final = metrics[-1]
    evaluated_validation_loss = _evaluate_loss(clone, loader, torch.device("cpu"))
    uniform_baseline = math.log(model_config.vocab_size)
    finite_metrics = math.isfinite(evaluated_validation_loss) and math.isfinite(float(recorded_final["validation_loss"]))
    learning_passed = finite_metrics and evaluated_validation_loss < uniform_baseline
    gates["learning_signal"] = {
        "passed": learning_passed,
        "evaluated_validation_loss": evaluated_validation_loss,
        "recorded_validation_loss": float(recorded_final["validation_loss"]),
        "recorded_validation_perplexity": float(recorded_final["validation_perplexity"]),
        "uniform_token_baseline_loss": uniform_baseline,
    }

    required_metadata = {"model", "dataset", "tokenizer", "seed", "device", "dtype", "torch", "platform", "tokenizer_metrics"}
    missing_metadata = sorted(required_metadata - set(metadata))
    gates["reproducibility_record"] = {
        "passed": not missing_metadata,
        "missing_fields": missing_metadata,
        "seed": metadata.get("seed"),
    }

    engineering_passed = all(gate["passed"] for gate in gates.values())
    claim_level = "E2 — Learns on a defined corpus" if engineering_passed else "E0 — Source only"
    return {
        "schema_version": 1,
        "checkpoint": str(checkpoint_path),
        "passed": engineering_passed,
        "claim_level": claim_level,
        "capability_status": "not_established",
        "capability_reason": "No held-out Code, Neural, Chem, or reasoning benchmark has passed a predefined threshold.",
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate evidence gates for a trained RAIQ checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to a RAIQ checkpoint_last.pt")
    parser.add_argument("--output", default=None, help="Evidence JSON output path; defaults beside the checkpoint")
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    repo_root = Path(__file__).resolve().parents[1]
    evidence = validate(checkpoint_path, repo_root)
    output_path = Path(args.output) if args.output else checkpoint_path.parent / "evidence.json"
    output_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    if not evidence["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
