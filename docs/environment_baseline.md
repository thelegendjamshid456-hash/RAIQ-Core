# RAIQ Core Environment Baseline

**Recorded:** 17 August 2026

The initial build environment provides six logical CPU cores on an Intel Xeon processor, approximately 3.8 GiB of system memory, 32 GiB of free storage in the working volume, and Python 3.12.3. No NVIDIA GPU, CUDA compiler, or preinstalled PyTorch package was detected.

| Resource | Observed baseline | Development consequence |
|---|---:|---|
| CPU | 6 logical processors | Sufficient for unit tests and very small CPU experiments |
| RAM | ~3.8 GiB total | Limits practical model size and data-loader parallelism |
| Free storage | ~32 GiB | Adequate for source, test artifacts, and small datasets; not a production corpus/checkpoint store |
| NVIDIA GPU/CUDA | Not available | No CUDA or BF16/FP16 training claims can be made locally |
| Python | 3.12.3 | Supported by the project’s declared Python floor |
| PyTorch | Not installed at baseline | Must be installed before model tests/training can run |

> This environment is intentionally used only to prove the end-to-end RAIQ Core engineering path with a tiny CPU model. It is not sufficient evidence that RAIQ-200M v1 has been trained, tuned, or evaluated.

The `configs/200m.yaml` target remains a reproducible architecture and training specification. A future 200M run requires an appropriately sized accelerator environment, a legally documented technical corpus, a measured token budget, and a separate experiment record.
