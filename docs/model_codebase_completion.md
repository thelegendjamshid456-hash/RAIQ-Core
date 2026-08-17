# RAIQ Model Codebase Completion Record

## Evidence status

The RAIQ model codebase has passed its local engineering completion gates and is ready to be transferred to a Google training environment. This record does **not** claim that RAIQ-200M has been pretrained or that domain capability has been demonstrated.

| Gate | Result |
|---|---|
| Full automated test suite | **16/16 passed** |
| Real CPU smoke training through the integrated trainer | **Passed**; checkpoint saved after two optimizer steps |
| Two-process distributed coordination | **Passed**; Gloo all-reduce mean was 1.5 as expected |
| Checkpoint resume | **Passed**; weights restored equivalently and continuation loss was finite |
| Tokenizer and corpus manifest validation | **Passed** |
| Held-out loss evidence | **Passed at E2**; the Tiny-BPE model learns on the declared local smoke corpus |
| Capability evidence | **Not established**; no held-out Code, Neural, Chem, or reasoning benchmark has passed |

## Completed code

The repository now contains the configurable decoder-only Transformer, trained BPE tokenizer path, corpus-manifest and shard validation, rank-aware loading, distributed context and DDP wrapping, checkpoint RNG restoration, inference, evidence validator, benchmark contracts, and a Google managed-training package.

## Deliberate boundary

The codebase is complete for transfer and external training setup. Actual RAIQ-200M pretraining still requires an approved production corpus, a provisioned Google GPU cluster, storage, and a passing environment-specific production preflight. Those requirements are not present in the local sandbox and have not been represented as passed.
