# loopguard: Semantic Loop Detection for ReAct Agents

**Catch the agent thinking the same thought in different words — and stop paying for doomed iterations.**

## Abstract

ReAct agents fail expensively: they get stuck reasoning in circles, burning tokens on iterations that were never going to work. Existing loop guards hash *actions* — they catch exact repeats but miss the common case where the agent rephrases the same intent with different wording or tool arguments. loopguard detects **semantic revisits**: it embeds each thought, flags when a new thought is semantically close to a recent one *while the observation hasn't substantively changed*, injects the loop trace into a reflection step that must propose a genuinely different strategy, and enforces **cost-justified iteration budgets** — continuation requires expected marginal gain to exceed cost, with graceful degradation (best-so-far + honest "gave up because") instead of silent spinning.

## 1. Gap analysis

Action-hash loop detection is standard practitioner knowledge: hashing recent actions and flagging repeats (dev.to ReAct guides, production "trajectory fingerprinting" writeups, governed-agent designs with "Action Hash Filter (Cycle Detection)" plus budget governors). All of these detect only *exact* repeats. Nothing found in the literature or open source for: (a) hashing (observation, thought-embedding) pairs — semantic revisit detection that catches loops where actions differ textually but the reasoning is circular; (b) reflection triggered *with the full loop trace* as input and a dissimilarity requirement on the new strategy; (c) an iteration budget the agent must cost-justify per step. The plain-hash guard was deliberately dropped as the primary mechanism (crowded) and kept only as the evaluation baseline.

## 2. Method

**Thought embedding.** Word-level TF-IDF + character n-gram TF-IDF fused via SVD (deterministic, zero downloads, CPU-only). A sentence-transformer backend was evaluated and rejected: ~90MB download for no measured advantage at trajectory scale (tens of thoughts). Any `embed(list[str]) -> ndarray` callable can be injected.

**Semantic revisit detector.** Rolling window (default 8) of thought embeddings. A revisit fires when max cosine similarity to the window exceeds a calibrated threshold AND the observation has not substantively changed. The observation gate is outcome-class based, not string-based: observations are classified into outcome classes (empty / error / results / done); the gate passes unless the outcome class changed. This is the key design decision — two wordings of "nothing found" are the same outcome, and a token-overlap gate cannot see that (verified empirically during development).

**Loop reflection.** On detection, the loop trace (similar thoughts + history) is injected into `reflect()`. The returned strategy's embedding must be dissimilar (default < 0.55 cosine) to all failed strategies, else the reflection is rejected and the loop terminates rather than re-entering the same reasoning.

**Cost-justified budget.** Each iteration has a cost; a gain EMA tracks expected marginal gain. Stop when gain < cost × floor for `patience` consecutive steps, or at the hard cap. On exhaustion, return best-so-far with a machine-readable reason.

## 3. Experiments

**Detection (40 scripted trajectories, 20 semantic loops / 20 clean).** Loop trajectories use paraphrased thoughts and actions (same intent, different text — what action-hashing misses); clean trajectories include exact-repeat retries *with new observations* (which naive hashing false-positives on). Threshold calibrated on a 20-trajectory train split, evaluated on a held-out 20-trajectory test split. Robustness checked across 5 RNG seeds.

| method | precision | recall | F1 (test) |
|---|---|---|---|
| loopguard (semantic) | 1.000 | 1.000 | **1.000** |
| action-hash baseline | 0.692 | 0.900 | 0.783 |

ΔF1 = **+0.217** (kill-gate bar: +0.10). Across 5 seeds: semantic F1 0.947–1.000, ΔF1 +0.20–+0.25 every seed.

**End-to-end budget (scripted ReActLoop).** Clean task: both guarded and naive succeed in 4 steps, 0 false revisits. Looping task: naive burns all 25 steps; guarded detects 3 revisits, reflects 3× with genuinely different strategies, then exhausts the cost-justified budget at step 11 with an honest reason — **56% of the budget saved** on the doomed task.

## 4. Limitations (honest)

- Scripted trajectories, not live LLM agents: template-generated paraphrases are cleaner than real model outputs. The numbers measure the mechanism, not production performance.
- The outcome-class keyword map is English- and tool-output-shaped; exotic observation formats need the pluggable gate.
- TF-IDF+SVD embeddings are weaker than transformer embeddings on deep paraphrase; the tradeoff (deterministic, zero-download) is documented and the embedder is swappable.
- The gain estimator in the budget is heuristic (user-supplied per-step gain); miscalibrated gains miscalibrate stopping.

## 5. Kill-gate verdict

**PASS.** Semantic detection beats action-hash F1 by +21.7pp on held-out trajectories (bar: +10pp), robust across seeds. The observation-gate design (outcome classes, not token overlap) was validated by ablation during development: token-Jaccard gating blocked nearly all true loops.

---

## v2 addendum: the recovery ladder (2026-10-28)

**Gap.** v1 answers "is the agent looping?" but its post-detection policy is brittle: reflections are the only recovery mechanism, a rejected (copycat) reflection terminates the run, and at most 3 reflections are attempted. Detection without recovery leaves rescuable agents dead.

**Method.** `RecoveringLoop` keeps v1's detector and budget untouched and replaces the post-detection policy with a ladder climbed at every detection: (1) reflect, with bounded annotated retries when a proposal is rejected as too similar to a failed strategy; (2) strategy_switch, a dedicated callable whose proposal must come from a different strategy family and pass the same dissimilarity check; (3) tool_restriction — tools appearing in the detected loop window are restricted for the rest of the run, their actions intercepted before the environment is called; (4) handoff, a stub that terminates with a structured trace packet (not counted as completion). Escalation across detections is carried by persistent state: the failed-strategy set and the restricted-tool set only grow. One v1 behaviour matters for interpretation: the first reflection is never dissimilarity-checked (the failed set is empty), in both arms.

**Experiment (pre-registered, PREREG.md).** 28 deterministic scripted scenarios: 4 task types (web search, code debug, data query, file retrieval) × 5 looping variants (reflect / reflect-retry / switch / restrict / unrecoverable) + 2 clean variants (progress / progress-with-legit-retry). Each looping variant encodes which rung a recovery *should* need, via the proposals the agent's reflect/switch callables return and which strategies actually work in its environment. Arms: v1 ReActLoop (unmodified), detect-only (terminate at first revisit), v2 RecoveringLoop. One detector threshold calibrated on uncontrolled trajectories (0.30, calibration F1 1.000) and frozen; identical budgets (max 40 steps, patience 99).

**Results.**

| arm | looping completion | clean completion | clean FP flags | mean steps (looping) |
|---|---|---|---|---|
| detect-only | 0/20 (0%) | 8/8 | 0 | 2.2 |
| v1 | 7/20 (35%) | 8/8 | 0 | 7.1 |
| v2 | 16/20 (80%) | 8/8 | 0 | 9.9 |

Δ v2 − v1 = **+45.0pp** (pre-registered bar +30pp: PASS). Clean false positives 0 in both guarded arms (bar 0: PASS); v2 clean completion 8/8 (bar: PASS). Per variant (v2 vs v1): reflect 4/4 vs 4/4; reflect-retry 4/4 vs 3/4 — v1 recovered three of these because the second paraphrase happened to fall below the 0.55 dissimilarity bar, letting v1 reach the working proposal within its 3-reflection cap; the fourth (data_query) rejected and terminated exactly as designed; switch 4/4 vs 0/4; restrict 4/4 vs 0/4; unrecoverable 0/4 vs 0/4, with v2 handing off gracefully in 4/4 (structured packet) where v1 terminated bare. v2 rung usage across the suite: 60 reflect proposals, 25 reflect retries, 24 switch proposals, 8 restrictions, 4 handoffs.

**Cost honesty.** The ladder is not free: on looping scenarios v2 spends a mean 9.9 steps vs v1's 7.1. It buys +45pp completion and graceful handoffs with those steps; on clean scenarios it spends exactly what v1 spends (3–4 steps, no detections, no recovery actions).

**Limitations.** Scripted agents with enumerable strategy sets; a live LLM's reflect() may produce copycats that TF-IDF similarity scores as novel (the ladder then wastes rungs) or novel strategies it over-rejects. Environments are synthetic; "works" is binary per strategy. The handoff rung emits a packet only. Detection itself is unchanged from v1, including its limitations.
