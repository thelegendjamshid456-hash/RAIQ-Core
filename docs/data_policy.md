# RAIQ Core Data Governance Baseline

The current `tiny-smoke` experiment uses only text derived from the user-provided RAIQ project specification. This small local corpus is used to confirm engineering behavior; it is not a production training dataset and must not be represented as one.

A future RAIQ-200M corpus must be assembled under a versioned data-governance process. Each source must have documented provenance, license or permission basis, acquisition date, content category, hash or immutable version identifier, filtering record, and train/validation/benchmark allocation. Datasets and checkpoints remain outside Git by default.

| Data category | Intended use | Required controls before use |
|---|---|---|
| Technical documentation and education | General technical language | Provenance, license review, quality filter, source-level split control |
| Source code and tests | RAIQ Code specialization | License compatibility, repository/file attribution, secret/credential scanning, deduplication |
| Mathematics and reasoning | Numerical and logical tasks | Source rights, answer validation, contamination review |
| ML and neural-network material | RAIQ Neural specialization | Versioning, code/test execution where applicable, benchmark isolation |
| Chemical/process-engineering material | RAIQ Chem specialization | Authoritative sources, unit/assumption checks, conservative safety review |
| Tool-use and verification traces | Agent and Verify layers | Explicit policy review, sandboxed replay, no secret or unsafe-command leakage |

The byte tokenizer is an interim transparent baseline. Its successor must be trained on the approved corpus and evaluated quantitatively on source code, mathematical notation, scientific notation, chemical formulae, engineering units, and ordinary technical prose. The vocabulary, special-token set, training parameters, corpus manifest, and tokenizer evaluation must be versioned together.
