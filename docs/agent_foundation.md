# RAIQ Agent Foundation

## Purpose

This phase adds the first executable **RAIQ Agent** foundation around RAIQ Core. It accepts a technical task, routes it to RAIQ Code, RAIQ Neural, RAIQ Chem, or general RAIQ Technical; selects a bounded reasoning depth; produces a structured plan; proposes verification-oriented tool intents; and records an append-only audit trace.

> The Agent foundation is intentionally a **planner, not an executor**. It cannot open files, run Python, execute shell commands, retrieve web content, access a network, or invoke a language model. Every proposed capability is retained as an auditable intent for a later policy-controlled tool layer.

## Routing model

The initial router is deterministic and transparent. It scores retained technical keywords and returns both the selected specialization and all per-domain scores. This is scaffolding for later evaluation against a trained RAIQ Core router; it is not presented as learned intelligence.

| Specialization | Example evidence signals | Plan emphasis |
|---|---|---|
| RAIQ Code | Python, API, debug, compile, test, stack trace | Control flow, minimal patch, regression tests, execution verification |
| RAIQ Neural | PyTorch, neural, gradient, optimizer, validation, transformer | Objective/data definition, diagnostic experiment, metrics, training checks |
| RAIQ Chem | Heat duty, energy balance, distillation, reactor, thermodynamics | System boundary, units, governing equations, conservation and plausibility |
| RAIQ Technical | No decisive specialist evidence | Objective, evidence, method selection, assumptions, general verification |

Reasoning depth is classified as **simple**, **moderate**, or **complex**. Complex plans include an explicit comparison of approaches, trade-offs, dependencies, and failure modes before committing to a path.

## Tool-intent boundary

The Agent may propose `files`, `python`, `shell`, `search`, `retrieval`, and `verify` intents. These are data values, not invoked tools. Each planning trace records a mandatory `execution_blocked` event stating that no execution backend exists in this phase.

| Trace event | Meaning |
|---|---|
| `task_received` | A task was accepted for planning only |
| `route_selected` | Specialization, scores, and keyword evidence were retained |
| `reasoning_depth_selected` | The bounded plan depth was selected |
| `tool_intents_proposed` | Tool capabilities were proposed without authorization or invocation |
| `execution_blocked` | The non-executing safety boundary was enforced |

## Usage

Run an inspectable plan through the source tree:

```bash
PYTHONPATH=. python3 -m raiq.agent.cli \
  --task 'Calculate heat duty for a heat exchanger and verify the energy balance.' \
  --file streams.csv
```

When installed as a package, the same interface is available as `raiq-plan`. It prints JSON containing the request, routing result, reasoning depth, plan steps, proposed tool intents, limitations, and trace events.

## Validation

The phase adds five Agent-specific tests and retains all five Core tests. The complete suite contains **10 passing tests**. The tests cover Code, Neural, Chem, and fallback technical routes; moderate and complex planning; proposal serialization; and the mandatory non-execution trace. A CLI smoke case routed a heat-exchanger energy-balance task to RAIQ Chem at 0.850 deterministic routing confidence and emitted `execution_blocked` after planning.

## Next boundary

The next safe implementation phase is not unrestricted tool execution. It should first add typed tool contracts, allow-lists, timeouts, resource limits, policy checks, result schemas, and verification hooks. Only then can individual low-risk capabilities be connected and tested behind explicit authorization.
