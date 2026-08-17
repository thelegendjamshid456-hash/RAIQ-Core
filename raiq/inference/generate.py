"""Command-line text generation for trained RAIQ Core checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from raiq.core.config import ModelConfig
from raiq.core.model import RAIQModel
from raiq.tokenizer.loader import load_tokenizer
from raiq.training.checkpoints import load_checkpoint


def generate_text(args: argparse.Namespace) -> str:
    checkpoint_path = Path(args.checkpoint)
    payload = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    model_config = ModelConfig(**payload["run_config"]["model"])
    model = RAIQModel(model_config).to(args.device)
    load_checkpoint(checkpoint_path, model=model, map_location=args.device)
    tokenizer_path = checkpoint_path.parent / "tokenizer.json"
    tokenizer = load_tokenizer(tokenizer_path)
    encoded = tokenizer.encode(args.prompt, add_bos=True)
    if len(encoded) >= model_config.max_seq_len:
        encoded = encoded[-(model_config.max_seq_len - 1) :]
    input_ids = torch.tensor([encoded], dtype=torch.long, device=args.device)
    generated = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(generated[0].tolist())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate text from a RAIQ Core checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint_last.pt")
    parser.add_argument("--prompt", required=True, help="Initial prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    print(generate_text(build_parser().parse_args()))


if __name__ == "__main__":
    main()
