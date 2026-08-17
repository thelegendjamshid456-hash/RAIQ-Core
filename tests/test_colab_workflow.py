from pathlib import Path

from raiq.core import RAIQModel, load_config
from raiq.data.manifest import verify_corpus_manifest
from raiq.data.text_dataset import TextBlockDataset
from raiq.tokenizer.loader import load_tokenizer


ROOT = Path(__file__).resolve().parents[1]


def test_colab_config_preserves_200m_architecture_and_t4_limits() -> None:
    config = load_config(ROOT / 'configs/200m_t4_smoke.yaml')
    assert config.model.name == 'RAIQ-200M-v1-T4-smoke'
    assert config.model.max_seq_len == 2048
    assert config.training.batch_size == 1
    assert config.training.grad_accumulation_steps == 32
    assert config.model.gradient_checkpointing is True
    assert RAIQModel(config.model).parameter_count() == 190_348_032


def test_actual_local_dataset_manifest_and_text_dataset() -> None:
    manifest = verify_corpus_manifest(ROOT / 'data/manifests/technical_smoke_v1.json')
    assert manifest['corpus_id'] == 'raiq-technical-smoke-v1'
    tokenizer = load_tokenizer(ROOT / 'artifacts/tiny-bpe-v1/tokenizer.json')
    train = TextBlockDataset(ROOT / 'datasets/technical_toy_train.txt', tokenizer, 64)
    validation = TextBlockDataset(ROOT / 'datasets/technical_toy_validation.txt', tokenizer, 64)
    assert len(train) > 0
    assert len(validation) > 0
