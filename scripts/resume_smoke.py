"""Verify an actual RAIQ training checkpoint can be restored and continued deterministically."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from raiq.core import RAIQModel, load_config
from raiq.data.text_dataset import TextBlockDataset
from raiq.tokenizer.loader import load_tokenizer
from raiq.training.checkpoints import load_checkpoint, save_checkpoint
from raiq.training.utils import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a RAIQ checkpoint-resume smoke test")
    parser.add_argument("--config", default="configs/tiny_bpe.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_config = load_config(args.config)
    set_seed(run_config.training.seed)
    tokenizer = load_tokenizer(run_config.data.tokenizer_path)
    dataset = TextBlockDataset(run_config.data.train_path, tokenizer, run_config.model.max_seq_len)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    first_inputs, first_labels = next(iter(loader))

    model = RAIQModel(run_config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    output = model(first_inputs, labels=first_labels)
    if output.loss is None:
        raise RuntimeError("resume smoke training step returned no loss")
    output.loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    checkpoint = Path(args.output).with_suffix(".checkpoint.pt")
    save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        step=1,
        run_config=run_config.to_dict(),
        metadata={"purpose": "resume smoke"},
    )
    restored = RAIQModel(run_config.model)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    payload = load_checkpoint(
        checkpoint,
        model=restored,
        optimizer=restored_optimizer,
        restore_rng=True,
    )
    weights_equivalent = all(
        torch.allclose(original, recovered, rtol=1e-6, atol=1e-7)
        for original, recovered in zip(model.parameters(), restored.parameters(), strict=True)
    )
    second_inputs, second_labels = list(loader)[1]
    resumed_output = restored(second_inputs, labels=second_labels)
    finite_continuation = resumed_output.loss is not None and bool(torch.isfinite(resumed_output.loss))
    passed = weights_equivalent and finite_continuation and payload["step"] == 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "passed": passed,
                "checkpoint_step": payload["step"],
                "weights_equivalent": weights_equivalent,
                "finite_continuation_loss": finite_continuation,
                "continuation_loss": None if resumed_output.loss is None else float(resumed_output.loss.detach()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
