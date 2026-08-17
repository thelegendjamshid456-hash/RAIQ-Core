from pathlib import Path

from raiq.core import RAIQModel, load_config
from raiq.data.manifest import verify_corpus_manifest
from raiq.data.text_dataset import TextBlockDataset
from raiq.tokenizer.loader import load_tokenizer
from raiq.training.utils import warmup_cosine_lr


ROOT = Path(__file__).resolve().parents[1]


def test_colab_config_preserves_200m_architecture_and_t4_limits() -> None:
    config = load_config(ROOT / 'configs/200m_t4_smoke.yaml')
    assert config.model.name == 'RAIQ-200M-v1-T4-smoke'
    assert config.model.max_seq_len == 2048
    assert config.training.batch_size == 1
    assert config.training.grad_accumulation_steps == 32
    assert config.model.gradient_checkpointing is True
    assert RAIQModel(config.model).parameter_count() == 190_348_032


def test_smoke_v2_config_uses_long_warmup_and_preserves_safety_invariants() -> None:
    config = load_config(ROOT / 'configs/200m_t4_smoke_v2.yaml')
    training = config.training
    assert config.model.name == 'RAIQ-200M-v1-T4-smoke-v2'
    assert training.max_steps == 5000
    assert training.learning_rate == 3e-4
    assert training.min_learning_rate == 3e-5
    assert training.warmup_steps == 250
    assert training.grad_clip_norm == 1.0
    assert training.finite_diagnostics is True
    assert config.data.train_path == '/content/drive/MyDrive/Raiq/datasets/technical_toy_train.txt'
    assert config.data.validation_path == '/content/drive/MyDrive/Raiq/datasets/technical_toy_validation.txt'
    assert config.data.tokenizer_path == '/content/drive/MyDrive/Raiq/tokenizer/raiq_code_bpe.json'
    assert config.data.corpus_manifest_path == '/content/drive/MyDrive/Raiq/manifests/technical_smoke_v1.json'

    initial_lr = warmup_cosine_lr(
        0,
        max_steps=training.max_steps,
        warmup_steps=training.warmup_steps,
        max_lr=training.learning_rate,
        min_lr=training.min_learning_rate,
    )
    peak_lr = warmup_cosine_lr(
        training.warmup_steps - 1,
        max_steps=training.max_steps,
        warmup_steps=training.warmup_steps,
        max_lr=training.learning_rate,
        min_lr=training.min_learning_rate,
    )
    decayed_lr = warmup_cosine_lr(
        500,
        max_steps=training.max_steps,
        warmup_steps=training.warmup_steps,
        max_lr=training.learning_rate,
        min_lr=training.min_learning_rate,
    )
    assert 0.0 < initial_lr < training.learning_rate
    assert peak_lr == training.learning_rate
    assert training.min_learning_rate < decayed_lr < training.learning_rate


def test_actual_local_dataset_manifest_and_text_dataset() -> None:
    manifest = verify_corpus_manifest(ROOT / 'data/manifests/technical_smoke_v1.json')
    assert manifest['corpus_id'] == 'raiq-technical-smoke-v1'
    tokenizer = load_tokenizer(ROOT / 'artifacts/tiny-bpe-v1/tokenizer.json')
    train = TextBlockDataset(ROOT / 'datasets/technical_toy_train.txt', tokenizer, 64)
    validation = TextBlockDataset(ROOT / 'datasets/technical_toy_validation.txt', tokenizer, 64)
    assert len(train) > 0
    assert len(validation) > 0
