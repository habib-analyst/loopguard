"""Experiment: does the v2 recovery ladder lift task completion on looping
agents vs v1 (detect + single-reflection, as shipped) and vs detect-only?

Suite: experiments/scenarios.py — 4 task types x (5 looping + 2 clean)
variants = 28 deterministic scripted scenarios (20 looping, 8 clean).

Protocol (pre-registered in PREREG.md):
- One embedder fit on the suite corpus; detector threshold calibrated ONCE
  via SemanticRevisitDetector.calibrate on uncontrolled trajectories
  (looping prefixes = True, clean runs = False), then frozen for all arms.
- Arms share scenarios, budgets (max_steps=40, patience=99) and threshold:
  v1 = unmodified ReActLoop; detect_only = terminate at first revisit;
  v2 = RecoveringLoop (reflect+retry -> switch -> restriction -> handoff).
- Bars: B1 v2-v1 completion on looping >= +30pp; B2 false-positive
  detections on clean runs = 0 (per arm); B3 v2 clean completion = 8/8.
"""
import json
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "experiments")
import numpy as np

from loopguard import (CostJustifiedBudget, ReActLoop, RecoveringLoop,
                       SemanticRevisitDetector, ThoughtEmbedder)
from scenarios import build_scenarios, corpus_texts, make_agent, uncontrolled_trajectory

MAX_STEPS = 40


def fresh_budget():
    return CostJustifiedBudget(cost_per_step=1.0, max_steps=MAX_STEPS, patience=99)


def run_v1(scn, emb, thr):
    det = SemanticRevisitDetector(embedder=emb, threshold=thr)
    loop = ReActLoop(detector=det, budget=fresh_budget(), embedder=emb)
    a = make_agent(scn)
    out = loop.run(a.think, a.act, a.observe, a.score, a.reflect)
    return {"success": out.success, "steps": out.steps,
            "flags": len(out.revisits), "handed_off": False,
            "max_rung": 0, "termination": out.termination}


def run_v2(scn, emb, thr):
    det = SemanticRevisitDetector(embedder=emb, threshold=thr)
    loop = RecoveringLoop(detector=det, budget=fresh_budget(), embedder=emb)
    a = make_agent(scn)
    out = loop.run(a.think, a.act, a.observe, a.score, a.reflect,
                   a.switch_strategy)
    return {"success": out.success, "steps": out.steps,
            "flags": len(out.revisits), "handed_off": out.handed_off,
            "max_rung": out.max_rung, "termination": out.termination,
            "rungs": [e.action for e in out.recovery_events]}


def run_detect_only(scn, emb, thr):
    det = SemanticRevisitDetector(embedder=emb, threshold=thr)
    budget = fresh_budget()
    a = make_agent(scn)
    history, flags = [], 0
    while True:
        th = a.think(history)
        ob = a.observe(a.act(th))
        gain, q = a.score(ob)
        budget.observe(gain, result=ob, score=q)
        r = det.step(th, ob)
        history.append(f"thought: {th} | action: - | obs: {ob}")
        if q >= 1.0:
            return {"success": True, "steps": budget.steps_used,
                    "flags": flags, "handed_off": False, "max_rung": 0,
                    "termination": "task solved"}
        if r is not None:
            flags += 1
            return {"success": False, "steps": budget.steps_used,
                    "flags": flags, "handed_off": False, "max_rung": 0,
                    "termination": "detect-only: terminated at first revisit"}
        stop, reason = budget.should_stop()
        if stop:
            return {"success": False, "steps": budget.steps_used,
                    "flags": flags, "handed_off": False, "max_rung": 0,
                    "termination": f"budget exhausted: {reason}"}


ARMS = {"v1": run_v1, "detect_only": run_detect_only, "v2": run_v2}


def main():
    scenarios = build_scenarios()
    looping = [s for s in scenarios if s.kind == "loop"]
    clean = [s for s in scenarios if s.kind == "clean"]
    print(f"suite: {len(scenarios)} scenarios ({len(looping)} looping, "
          f"{len(clean)} clean), task types: "
          f"{sorted({s.task_type for s in scenarios})}")

    emb = ThoughtEmbedder().fit(corpus_texts(scenarios))

    # Calibrate the detector threshold once, on uncontrolled trajectories.
    cal_trajs, cal_labels = [], []
    for s in looping:
        cal_trajs.append(uncontrolled_trajectory(s))
        cal_labels.append(True)
    for s in clean:
        cal_trajs.append(uncontrolled_trajectory(s))
        cal_labels.append(False)
    thr, cal_f1 = SemanticRevisitDetector.calibrate(
        emb, cal_trajs, cal_labels, thresholds=np.arange(0.30, 0.96, 0.02))
    print(f"calibrated threshold: {thr:.2f} (calibration F1 {cal_f1:.3f})")

    rows = []
    for scn in scenarios:
        row = {"scenario": scn.id, "task_type": scn.task_type,
               "kind": scn.kind, "variant": scn.variant}
        for arm, fn in ARMS.items():
            row[arm] = fn(scn, emb, thr)
        rows.append(row)
        print(f"{scn.id:32s} "
              + "  ".join(f"{arm}: {'OK ' if row[arm]['success'] else 'FAIL'}"
                          f"({row[arm]['steps']} steps)" for arm in ARMS))

    def completion(rs, arm):
        return sum(r[arm]["success"] for r in rs) / len(rs) if rs else 0.0

    summary = {}
    for arm in ARMS:
        summary[arm] = {
            "loop_completion": round(completion(looping_rows(rows), arm), 3),
            "loop_successes": sum(r[arm]["success"] for r in rows
                                  if r["kind"] == "loop"),
            "clean_completion": round(completion(clean_rows(rows), arm), 3),
            "clean_false_positive_flags": sum(r[arm]["flags"] for r in rows
                                              if r["kind"] == "clean"),
            "mean_steps_loop": round(float(np.mean(
                [r[arm]["steps"] for r in rows if r["kind"] == "loop"])), 1),
        }
    delta_pp = round(100 * (summary["v2"]["loop_completion"]
                            - summary["v1"]["loop_completion"]), 1)
    unrec = [r for r in rows if r["variant"] == "unrecoverable"]
    handoffs = sum(r["v2"]["handed_off"] for r in unrec)
    rung_use = {}
    for r in rows:
        for a in r["v2"].get("rungs", []):
            rung_use[a] = rung_use.get(a, 0) + 1

    bars = {
        "B1_delta_pp_v2_minus_v1_loop": delta_pp,
        "B1_pass": bool(delta_pp >= 30.0),
        "B2_clean_fp_zero_v1": bool(summary["v1"]["clean_false_positive_flags"] == 0),
        "B2_clean_fp_zero_v2": bool(summary["v2"]["clean_false_positive_flags"] == 0),
        "B3_v2_clean_complete": bool(summary["v2"]["clean_completion"] == 1.0),
    }
    bars["all_pass"] = bool(bars["B1_pass"] and bars["B2_clean_fp_zero_v1"]
                            and bars["B2_clean_fp_zero_v2"]
                            and bars["B3_v2_clean_complete"])

    results = {
        "suite": {"n_scenarios": len(scenarios), "n_looping": len(looping),
                  "n_clean": len(clean),
                  "task_types": sorted({s.task_type for s in scenarios})},
        "calibrated_threshold": round(thr, 2),
        "calibration_f1": round(cal_f1, 3),
        "summary": summary,
        "delta_pp_v2_minus_v1_loop": delta_pp,
        "unrecoverable_handoffs_v2": f"{handoffs}/{len(unrec)}",
        "v2_rung_usage": rung_use,
        "bars": bars,
        "rows": [{k: v for k, v in r.items()} for r in rows],
        "note": ("threshold calibrated once on uncontrolled trajectories and "
                 "frozen for all arms; budgets identical across arms "
                 "(max_steps=40, patience=99); agents scripted/deterministic"),
    }
    with open("results/recovery.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: results[k] for k in
                      ("summary", "delta_pp_v2_minus_v1_loop",
                       "unrecoverable_handoffs_v2", "v2_rung_usage", "bars")},
                     indent=2))
    print("BARS:", "ALL PASS" if bars["all_pass"] else "SEE ABOVE")
    return results


def looping_rows(rows):
    return [r for r in rows if r["kind"] == "loop"]


def clean_rows(rows):
    return [r for r in rows if r["kind"] == "clean"]


if __name__ == "__main__":
    main()
