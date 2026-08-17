from __future__ import annotations

from raiq.tokenizer.bpe_tokenizer import BPETokenizerMetadata, BytePairTokenizer


def test_bpe_training_round_trip_and_serialization(tmp_path) -> None:
    corpus = [
        "Python code uses functions and tests. Python code uses functions.",
        "Heat duty uses energy balance, kJ/mol, H2O, and ΔH.",
    ]
    tokenizer = BytePairTokenizer.train(
        corpus,
        vocab_size=300,
        metadata=BPETokenizerMetadata(training_corpus_id="test-corpus-v1"),
    )
    text = "Python functions calculate ΔH for H2O at kg/s."
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    assert tokenizer.decode(ids) == text
    assert tokenizer.vocab_size <= 300
    assert tokenizer.vocab_size > tokenizer.base_vocab_size
    saved = tokenizer.save(tmp_path / "tokenizer.json")
    restored = BytePairTokenizer.load(saved)
    assert restored.to_dict() == tokenizer.to_dict()
    assert restored.decode(restored.encode(text)) == text


def test_bpe_compression_is_measurable_on_repeated_technical_text() -> None:
    corpus = ["heat exchanger heat exchanger heat exchanger heat exchanger"]
    tokenizer = BytePairTokenizer.train(corpus, vocab_size=300)
    stats = tokenizer.compression_stats(corpus[0])
    assert stats["tokens"] < stats["bytes"]
    assert 0.0 < stats["tokens_per_byte"] < 1.0


def test_manifest_verification_accepts_hashed_local_splits(tmp_path) -> None:
    import hashlib
    import json

    from raiq.data.manifest import verify_corpus_manifest

    repository_root = tmp_path
    split_dir = repository_root / "datasets"
    split_dir.mkdir()
    split = split_dir / "train.txt"
    split.write_text("RAIQ technical corpus", encoding="utf-8")
    manifest_dir = repository_root / "data" / "manifests"
    manifest_dir.mkdir(parents=True)
    payload = {
        "corpus_id": "test-v1",
        "splits": [
            {
                "name": "train",
                "path": "datasets/train.txt",
                "bytes": split.stat().st_size,
                "sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
            }
        ],
    }
    manifest = manifest_dir / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    assert verify_corpus_manifest(manifest)["corpus_id"] == "test-v1"


def test_tokenizer_loader_reconstructs_saved_bpe(tmp_path) -> None:
    from raiq.tokenizer.loader import load_tokenizer

    tokenizer = BytePairTokenizer.train(["RAIQ RAIQ Core Core"], vocab_size=290)
    saved = tokenizer.save(tmp_path / "bpe.json")
    loaded = load_tokenizer(saved)
    assert loaded.decode(loaded.encode("RAIQ Core")) == "RAIQ Core"
