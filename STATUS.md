# loopguard — STATUS

## 2026-10-12 ~22:45 PKT — START
PIVOT invention: semantic loop detection via thought-embedding revisit + cost-justified iteration budgets.
(Action-hash crowded per gap research — dropped as primary mechanism, kept as baseline for the kill-gate comparison.)

Plan:
1. `src/loopguard/embeddings.py` — ThoughtEmbedder (TF-IDF+SVD, deterministic, no downloads; pluggable)
2. `src/loopguard/detector.py` — SemanticRevisitDetector (cosine sim + observation-change gate)
3. `src/loopguard/budget.py` — CostJustifiedBudget (gain estimator, hard cap, graceful degradation)
4. `src/loopguard/loop.py` — ReActLoop wrapper with reflection injection
5. `tests/` — full suite
6. `experiments/` — 40 scripted trajectories, precision/recall vs action-hash, budget success rates
7. Kill-gate: semantic F1 ≥ action-hash F1 + 10pp, else kill
8. paper/PAPER.md, README.md, LinkedIn draft, push payload
