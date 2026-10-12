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

## How it works

1. **Thought embedding** — TF-IDF (word + char n-grams) + SVD. Deterministic, zero downloads, CPU-only; swappable.
2. **Semantic revisit detector** — flags when a thought's cosine similarity to the recent window exceeds threshold *while the observation hasn't substantively changed*. The observation gate is outcome-class based (empty/error/results/done), not string-based — two wordings of "nothing found" are the same outcome.
3. **Loop reflection** — the loop trace is injected into `reflect()`; the new strategy must be embedding-dissimilar to failed ones, or the loop terminates instead of re-entering.
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
