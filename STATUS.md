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

## 2026-10-28 — v2 (0.2.0): recovery ladder

- `src/loopguard/recovery.py` — `RecoveringLoop`: at each detection, climb
  reflect (bounded annotated retries on copycat proposals) → strategy_switch
  → tool_restriction (actions intercepted pre-environment) → handoff stub
  (structured packet, not a completion). v1 modules byte-identical.
- `experiments/scenarios.py` + `run_recovery.py` — 28-scenario multi-task
  suite (4 task types × 5 looping + 2 clean variants), 3 arms (v1 /
  detect-only / v2), threshold calibrated once (0.30, F1 1.000) and frozen.
- Pre-registered bars (PREREG.md, locked pre-run): B1 Δ completion ≥ +30pp —
  measured **+45.0pp** (v2 16/20 vs v1 7/20; detect-only 0/20). B2 clean
  false positives = 0 — measured 0 both arms. B3 no regression — v1 tests
  12/12 unmodified, v1 experiments reproduce exactly, v2 clean 8/8.
  **All bars PASS → ship as update to loopguard.**
- Tests: 19/19 (12 v1 + 7 new ladder tests). Handoffs on unrecoverable: 4/4.
