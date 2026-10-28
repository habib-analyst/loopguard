[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)

# loopguard

**Your ReAct agent is thinking the same thought in different words — and you're paying per token for it.** loopguard catches semantic loops (same intent, reworded — the kind action-hashing misses), forces reflections to propose genuinely different strategies, and stops iterations whose expected gain doesn't justify their cost.

## Quickstart

```bash
pip install loopguard
```

```python
from loopguard import ThoughtEmbedder, SemanticRevisitDetector, CostJustifiedBudget, ReActLoop

embedder = ThoughtEmbedder().fit(your_agent_thought_corpus)
detector = SemanticRevisitDetector(embedder=embedder, threshold=0.45)
budget = CostJustifiedBudget(cost_per_step=1.0, max_steps=25, patience=3)
loop = ReActLoop(detector=detector, budget=budget, embedder=embedder)

outcome = loop.run(think=agent.think, act=agent.act, observe=env.step,
                   score=score_fn, reflect=agent.reflect)
print(outcome.success, outcome.steps, outcome.termination)
```

Real output on a stuck agent:

```
success:     False
steps used:  4
revisits:    2
reflections: 1
termination: terminated: reflection proposed a strategy too similar (max sim 1.00) to a failed one; refusing to re-loop...
```

## Measured results

40 scripted trajectories (20 semantic loops, 20 clean), threshold calibrated on train split, evaluated on held-out test split:

| method | precision | recall | F1 |
|---|---|---|---|
| loopguard (semantic) | 1.000 | 1.000 | **1.000** |
| action-hash baseline | 0.692 | 0.900 | 0.783 |

ΔF1 **+0.217** (kill-gate bar +0.10); robust across 5 seeds (+0.20–+0.25). End-to-end: **56% of budget saved** on a doomed task (11 steps vs 25), zero false revisits on the clean task.

## v2: the recovery ladder

v1 detects a loop and tries reflections — but a rejected reflection is fatal, so recoverable agents die on the floor. v2 (`RecoveringLoop`) climbs a ladder of concrete recovery actions at every detection, escalating whenever a rung fails:

1. **reflect** — loop-trace reflection (as v1), except a copycat proposal is no longer fatal: the trace is annotated ("previous proposal too similar") and `reflect()` is asked again (bounded retries).
2. **strategy_switch** — a dedicated `switch_strategy()` callable that must return a strategy from a different family (still dissimilarity-checked).
3. **tool_restriction** — the loop restricts the tools used inside the detected loop; actions calling them are intercepted (the environment never runs) and the agent is told to pick a different tool.
4. **handoff** — a human-handoff stub: the run ends with a structured packet (strategies tried, tools restricted, recovery events, history tail). A handoff is a graceful outcome, not a completion.

```python
from loopguard import RecoveringLoop
loop = RecoveringLoop(detector=detector, budget=budget, embedder=embedder)
outcome = loop.run(think=agent.think, act=agent.act, observe=env.step,
                   score=score_fn, reflect=agent.reflect,
                   switch_strategy=agent.switch_strategy)
print(outcome.success, outcome.max_rung, outcome.handed_off)
```

**Measured on a 28-scenario multi-task suite** (4 task types — web search, code debug, data query, file retrieval — × 5 looping variants + 2 clean variants; scripted deterministic agents; one calibrated threshold, identical budgets for all arms; bars pre-registered in `PREREG.md`):

| arm | looping completion (20) | clean completion (8) | clean false positives |
|---|---|---|---|
| detect-only (terminate at first revisit) | 0/20 | 8/8 | 0 |
| v1 `ReActLoop` (as shipped) | 7/20 (35%) | 8/8 | 0 |
| **v2 `RecoveringLoop`** | **16/20 (80%)** | 8/8 | 0 |

Δ completion v2 − v1 = **+45.0pp** (pre-registered bar +30pp). Per looping variant, v2 vs v1: reflect 4/4 vs 4/4; reflect-retry 4/4 vs 3/4; switch 4/4 vs 0/4; restrict 4/4 vs 0/4; unrecoverable 0/4 vs 0/4 — with all 4 unrecoverable runs ending in a structured handoff instead of a bare termination. The ladder costs steps on doomed runs (mean 9.9 vs v1's 7.1 steps on looping scenarios): it spends more to rescue the rescuable and to prove the rest are not.

**Honest limits:** scripted agents and synthetic environments — the strategies an agent can switch to are enumerable, and a real LLM agent's reflection may be copycat in ways TF-IDF similarity misses (or genuinely novel in ways it over-rejects). Detection quality is unchanged from v1 (same detector, same F1); v2 is purely about what happens after detection. The handoff rung is a stub — it emits the packet; wiring it to a real human queue is downstream work.

## How it works

1. **Thought embedding** — TF-IDF (word + char n-grams) + SVD. Deterministic, zero downloads, CPU-only; swappable.
2. **Semantic revisit detector** — flags when a thought's cosine similarity to the recent window exceeds threshold *while the observation hasn't substantively changed*. The observation gate is outcome-class based (empty/error/results/done), not string-based — two wordings of "nothing found" are the same outcome.
3. **Loop reflection** — the loop trace is injected into `reflect()`; the new strategy must be embedding-dissimilar to failed ones, or the loop terminates instead of re-entering. (v1 semantics — v2's `RecoveringLoop` escalates to the next ladder rung instead of terminating.)
4. **Cost-justified budget** — stop when expected marginal gain < cost for `patience` steps, or at the hard cap; degrade gracefully with best-so-far and an honest reason.

## Citation

```bibtex
@software{rehman2026loopguard,
  author = {Rehman, Habib Ur},
  title = {loopguard: Semantic Loop Detection for ReAct Agents},
  year = {2026},
  url = {https://github.com/habib-analyst/loopguard}
}
```

---
Built by [Habib Ur Rehman](https://github.com/habib-analyst) · ML/AI Engineer & healthcare-AI researcher ·
[LinkedIn](https://www.linkedin.com/in/hur-dev) · [habib.top](https://habib.top)
