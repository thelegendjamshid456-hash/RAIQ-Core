# RAIQ Coding and Reasoning Capability Target

RAIQ may be described as comparable to an external model only if it is evaluated under a shared, reproducible protocol with the same task splits, sampling budget, tool policy, and scoring rules. A single benchmark or an unverified leaderboard comparison is insufficient.

| Capability | Required evidence | Initial benchmark contract |
|---|---|---|
| Repository-level coding | Patch passes the benchmark harness | SWE-bench Verified, reporting percent resolved and complete inference/tool policy |
| Fresh code reasoning | Held-out programming solutions, repair, execution, and test-output performance | LiveCodeBench problems released after the final RAIQ training-data cutoff |
| General reasoning | Held-out objective reasoning tasks with declared contamination controls | A separate frozen reasoning suite selected before Google pretraining; its items must be excluded from training and tuning |
| Parity assertion | Independent, same-protocol result for RAIQ and reference system | RAIQ score must meet or exceed the reference score within the predeclared statistical tolerance on every required domain |

SWE-bench evaluates real GitHub issues by whether a generated patch resolves the issue; its Verified subset contains 500 human-filtered instances.[1] LiveCodeBench evaluates code generation, self-repair, test-output prediction, and code execution, and supports post-training-cutoff evaluation to reduce contamination risk.[2]

The present RAIQ Tiny model is not eligible for these benchmarks as a parity candidate. Its evidence level is **E2**, meaning that it learns on its declared local smoke corpus. No held-out coding or reasoning score has been established.

## References

[1]: https://www.swebench.com/ "SWE-bench official leaderboards"
[2]: https://livecodebench.github.io/ "LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code"
