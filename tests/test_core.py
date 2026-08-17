from __future__ import annotations

import torch

from raiq.core import ModelConfig, RAIQModel
from raiq.tokenizer.byte_tokenizer import ByteTokenizer
from raiq.training.checkpoints import load_checkpoint, save_checkpoint


def tiny_config() -> ModelConfig:
    return ModelConfig(
        name="RAIQ-Test",
        vocab_size=300,
        max_seq_len=32,
        n_layers=2,
        d_model=32,
        n_heads=4,
        n_kv_heads=4,
        d_ff=64,
        dropout=0.0,
        tie_embeddings=True,
    )


def test_model_forward_loss_and_gradients() -> None:
    torch.manual_seed(7)
    model = RAIQModel(tiny_config())
    tokens = torch.randint(0, 300, (2, 12))
    labels = torch.randint(0, 300, (2, 12))
    output = model(tokens, labels=labels)
    assert output.logits.shape == (2, 12, 300)
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    assert model.token_embeddings.weight.grad is not None
    assert model.parameter_count() > 0


def test_causal_mask_prevents_future_leakage() -> None:
    torch.manual_seed(11)
    model = RAIQModel(tiny_config()).eval()
    first = torch.tensor([[5, 6, 7, 8, 9]])
    second = first.clone()
    second[0, -1] = 99
    logits_first = model(first).logits
    logits_second = model(second).logits
    torch.testing.assert_close(logits_first[:, :-1], logits_second[:, :-1], rtol=0.0, atol=1e-6)


def test_kv_cache_matches_full_forward_for_last_token() -> None:
    torch.manual_seed(13)
    model = RAIQModel(tiny_config()).eval()
    tokens = torch.tensor([[4, 8, 15, 16, 23, 42]])
    full_last_logits = model(tokens).logits[:, -1]
    prefix = model(tokens[:, :-1], use_cache=True)
    cached_last_logits = model(
        tokens[:, -1:], past_key_values=prefix.past_key_values, use_cache=True
    ).logits[:, -1]
    torch.testing.assert_close(full_last_logits, cached_last_logits, rtol=1e-4, atol=1e-5)


def test_checkpoint_round_trip(tmp_path) -> None:
    torch.manual_seed(17)
    config = tiny_config()
    model = RAIQModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint = save_checkpoint(
        tmp_path / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        step=3,
        run_config={"model": config.to_dict()},
        metadata={"test": True},
    )
    restored = RAIQModel(config)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    payload = load_checkpoint(checkpoint, model=restored, optimizer=restored_optimizer)
    assert payload["step"] == 3
    for original, loaded in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(original, loaded)


def test_byte_tokenizer_round_trip_for_technical_text(tmp_path) -> None:
    tokenizer = ByteTokenizer()
    text = "H2O at 3.5 kg/s; ΔH = 12 kJ/mol; def f(x): return x**2"
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    assert tokenizer.decode(ids) == text
    saved = tokenizer.save(tmp_path / "tokenizer.json")
    restored = ByteTokenizer.load(saved)
    assert restored.decode(restored.encode(text)) == text
