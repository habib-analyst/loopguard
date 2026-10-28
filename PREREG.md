# loopguard v2 — PREREG (pre-committed bars, written before any v2 code or runs)

Date: 2026-10-28. Status: LOCKED before implementation. Bars below do not move;
scenario wording/mechanics may be tuned during development, bars may not.

## What v2 claims

v1 detects semantic loops and, on detection, tries ONE mechanism (a reflection
that must be embedding-dissimilar to failed strategies, else it terminates).
v2 adds a **recovery ladder**: on each detection, escalate through concrete
recovery actions instead of giving up at the first failed reflection:

- Rung 1 `reflect` — loop-trace reflection (as v1), PLUS bounded retries when a
  proposal is rejected as too similar (v1 treats one copycat proposal as fatal;
  v2 annotates the trace "previous proposal too similar" and asks again, up to
  `max_reflect_retries` extra proposals per detection).
- Rung 2 `strategy_switch` — an explicit strategy-switch callable that must
  return a strategy from a *different family* than the failed ones; also
  subject to the dissimilarity check.
- Rung 3 `tool_restriction` — the loop itself restricts the tools that appear
  in the detected loop window: subsequent actions using a restricted tool are
  intercepted (the environment is never called) and the agent receives a
  restriction notice, forcing a different tool.
- Rung 4 `handoff` — human-handoff stub: terminate with a structured handoff
  packet (history tail, strategies tried, restricted tools, reason). A handoff
  is a graceful outcome, NOT a task completion.

## Evaluation design (locked)

- Multi-task scenario suite: **4 task types** (web_search, code_debug,
  data_query, file_retrieval). Each type has 5 looping variants —
  `reflect` (first reflection proposal is the working strategy),
  `reflect_retry` (first proposal is a same-family paraphrase, second works),
  `switch` (all reflection proposals are paraphrases; only the family switch
  works), `restrict` (reflection + switch keep the looping tool; only tool
  restriction forces the working fallback), `unrecoverable` (no working
  strategy exists) — and 2 clean variants (progressing; progressing with a
  legitimate retry after a transient error). 28 scenarios total, 20 looping.
- Arms: (a) **v1** = unmodified `ReActLoop`; (b) **detect-only** = detector +
  budget, terminate at first revisit; (c) **v2** = `RecoveringLoop` ladder.
  Same scenario agents/environments, same budgets (max_steps 40, patience 99),
  same calibrated detector threshold for all arms.
- Detector threshold calibrated once, before arm runs, via
  `SemanticRevisitDetector.calibrate` on calibration trajectories (uncontrolled
  looping prefixes labelled loop; natural clean sequences labelled clean).
- All agents scripted and deterministic; no RNG in scenarios; no LLMs.

## Bars (locked)

- **B1 (primary):** v2 task-completion on the 20 looping scenarios minus v1
  task-completion on the same scenarios **≥ +30 percentage points**.
- **B2 (no false positives):** false-positive revisit detections on the 8 clean
  runs = **0** (counted per arm; must hold for v2 and v1 arms).
- **B3 (no regression):** (i) the v1 test suite passes unmodified in
  behaviour; (ii) v2 completion on clean scenarios = 100% (8/8); (iii) v2
  completion on looping scenarios ≥ v1 completion (implied by B1, restated
  for the miss rule below).
- Reported but NOT gated: detect-only completion; mean steps per arm; rung
  usage counts; handoff rate on the 4 unrecoverable scenarios (design target
  4/4, reported honestly either way); v1 per-variant completion.

## Ship / kill rule (locked)

- B1 + B2 + B3 all pass → ship v2 as an update to loopguard.
- B1 missed but B2 + B3 hold (nothing regresses; clean runs untouched; v1
  behaviour intact) → ship as a *documented miss*: README and PAPER state the
  measured delta and that the +30pp bar was not met.
- B2 or B3 fails (false positives, clean-run breakage, v1 regression) →
  KILLED.md, no ship.

## Amendment (2026-10-28, still before any v2 code run or experiment)

Written after close-reading the v1 source, before implementation ran:
v1's reflection check only compares a proposal against *previously accepted*
reflections — the FIRST reflection is accepted unconditionally. The scenario
mechanics above are therefore made precise (bars unchanged):

- `reflect`: proposals = [working]. Both arms recover.
- `reflect_retry`: proposals = [paraphrase A, paraphrase A2, working W].
  Det1: A accepted by both arms (unchecked). Det2: v1 checks A2 against A,
  rejects, and terminates (a rejected reflection is fatal in v1). v2 retries
  with the annotated trace, W is dissimilar to A, accepted → recovery is
  credited to rung 1 (reflect_retry).
- `switch`: proposals = [A, A2] only; switch callable returns W. v1 dies at
  Det2 as above; v2 exhausts reflect + retry, escalates to rung 2.
- `restrict`: proposals = [A, B] (paraphrases); switch returns a different-
  family strategy that still uses the looping tool and does not work. v1 dies
  at Det2; v2 accepts the switch at Det2, loops again, and at Det3 escalates
  past exhausted reflect/switch to rung 3 (tool restriction), which forces
  the working fallback strategy.
- `unrecoverable`: no working strategy; v1 dies at Det2; v2 escalates to
  rung 4 (handoff).

Ladder semantics (clarified): every detection climbs reflect → switch →
restriction → handoff *within that detection*, escalating whenever a rung
fails; escalation across detections is carried by the persistent
failed-strategy set and the persistent restricted-tool set.
