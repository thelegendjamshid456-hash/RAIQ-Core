# Local Dataset Inspection

## Observed files

The local workspace contains exactly two dataset files under `datasets/`:

| Split | Path | Format | Bytes | Lines |
|---|---|---|---:|---:|
| Train | `datasets/technical_toy_train.txt` | UTF-8 plain text with CRLF line endings | 13,860 | 930 |
| Validation | `datasets/technical_toy_validation.txt` | UTF-8 plain text with CRLF line endings | 14,547 | 719 |

The combined dataset size is approximately 36 KiB on disk. The files are raw technical text, not pre-tokenized tensors, JSONL, Parquet, or source-code files with language extensions. They are already separated into train and validation files. An existing manifest, `data/manifests/technical_smoke_v1.json`, records their SHA-256 hashes and byte counts.

## Compatibility

The files are directly compatible with the existing `TextBlockDataset`, which reads UTF-8 text and tokenizes it through a RAIQ tokenizer before producing fixed next-token windows. The current corpus is far too small for meaningful 200M pretraining and is suitable only for smoke tests. No production corpus was found in the local workspace.

The production configuration remains separate and continues to require its existing multi-GPU, storage, license, quality, and recovery evidence. The Colab workflow must use a separate experiment configuration and must not weaken those production gates.
