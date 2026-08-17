"""Train a deterministic RAIQ byte-pair tokenizer from local corpus files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from raiq.tokenizer.bpe_tokenizer import BPETokenizerMetadata, BytePairTokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a RAIQ byte-pair tokenizer")
    parser.add_argument("--input", action="append", required=True, help="UTF-8 corpus path; may be repeated")
    parser.add_argument("--output", required=True, help="Destination tokenizer JSON path")
    parser.add_argument("--vocab-size", type=int, required=True, help="Requested vocabulary size")
    parser.add_argument("--name", default="raiq-technical-bpe")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--min-pair-frequency", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = [Path(path) for path in args.input]
    texts = [path.read_text(encoding="utf-8") for path in paths]
    tokenizer = BytePairTokenizer.train(
        texts,
        vocab_size=args.vocab_size,
        min_pair_frequency=args.min_pair_frequency,
        metadata=BPETokenizerMetadata(
            name=args.name,
            version=args.version,
            training_corpus_id=args.corpus_id,
        ),
    )
    tokenizer.save(args.output)
    sample = "RAIQ Core: Python, ΔH, H2SO4, kg/s, and heat duty."
    output = {
        "output": args.output,
        "vocab_size": tokenizer.vocab_size,
        "merge_count": len(tokenizer.merges),
        "sample_compression": tokenizer.compression_stats(sample),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
