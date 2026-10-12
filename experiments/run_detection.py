"""Experiment: semantic revisit detection vs action-hash baseline.

40 scripted trajectories (20 semantic loops, 20 clean). Loop trajectories
use DIFFERENT wording for the same intent, so exact action hashing misses
them. Clean trajectories include exact-repeat retries with NEW observations
(which a naive hash flags as loops — the observation gate should save us).
"""
import json
import random
import sys

sys.path.insert(0, "src")
import numpy as np

from loopguard import SemanticRevisitDetector, ThoughtEmbedder

rng = random.Random(7)

TOPICS = ["the quarterly revenue figure", "the API rate limit", "the database password",
          "the deployment config", "the customer churn rate", "the model checkpoint"]

# --- loop trajectory templates: same intent, different words, stale obs ---
LOOP_THOUGHTS = [
    "I should search for {t}",
    "Let me try searching {t} again with different terms",
    "Maybe a reworded query will find {t}",
    "I'll run another search for {t}, tweaking the keywords",
    "Searching {t} once more — the last query must have been off",
    "Let me look up {t} in the docs instead",
    "Perhaps the docs have {t} under a different name",
    "I will check the documentation for {t} again",
]
LOOP_OBS = ["search: 0 results for '{q}'", "docs: no page matching '{q}'",
            "search: empty result set for '{q}'", "kb: 0 hits for '{q}'"]
LOOP_QUERIES = ["{t}", "{t} q3", "{t} latest", "{t} docs", "{t} 2026"]
LOOP_ACTIONS = ["search[{t} :: q{i}]", "docs_lookup[{t} :: q{i}]",
                "search[{t} :: retry{i}]", "kb_query[{t} :: attempt{i}]"]

# --- clean trajectories: progressing thoughts, changing observations ---
CLEAN_STEPS = [
    ("I need to find {t}; first I'll check the index", "search[{t}]", "index lists 3 candidates"),
    ("The index has candidates; I'll inspect the first one", "inspect[candidate_1]", "candidate_1 looks relevant"),
    ("Candidate 1 is close but version is old; checking candidate 2", "inspect[candidate_2]", "candidate_2 matches the version"),
    ("Candidate 2 matches; verifying its checksum", "verify[candidate_2]", "checksum valid — done"),
    ("Checksum valid, task complete", "finish[]", "task complete"),
]
# clean-with-retry: exact action repeats but observation CHANGES (not a loop)
CLEAN_RETRY = [
    ("I should search for {t}", "search[{t}]", "search: timeout after 30s"),
    ("I should search for {t}", "search[{t}]", "search: partial results, 2 candidates found"),
    ("Two candidates this time; inspecting the first", "inspect[candidate_1]", "inspect: candidate_1 matches {t}"),
    ("Candidate 1 matches; verifying freshness", "verify[{t}]", "verify: fresh — done"),
]


def make_loop_trajectory():
    t = rng.choice(TOPICS)
    n = rng.randint(6, 9)
    traj, acts = [], []
    for i in range(n):
        th = rng.choice(LOOP_THOUGHTS).format(t=t)
        # Paraphrased actions: same intent, different text — what action-hash misses.
        # In ~30% of steps, an exact repeat sneaks in (realistic agent behavior).
        if acts and rng.random() < 0.30:
            acts.append(rng.choice(acts))
        else:
            acts.append(rng.choice(LOOP_ACTIONS).format(t=t, i=i))
        traj.append((th, rng.choice(LOOP_OBS).format(t=t, q=rng.choice(LOOP_QUERIES).format(t=t))))
    return traj, acts


def make_clean_trajectory():
    t = rng.choice(TOPICS)
    if rng.random() < 0.5:
        steps = CLEAN_STEPS
    else:
        steps = CLEAN_RETRY
    traj = [(th.format(t=t), ob.format(t=t)) for th, _, ob in steps]
    acts = [a.format(t=t) for _, a, _ in steps]
    return traj, acts


def action_hash_predict(actions, window=8):
    seen = set()
    for a in actions:
        if a in seen:
            return True
        seen.add(a)
        if len(seen) > window:
            seen.pop()
    return False


def prf(preds, labels):
    tp = sum(p and l for p, l in zip(preds, labels))
    fp = sum(p and not l for p, l in zip(preds, labels))
    fn = sum((not p) and l for p, l in zip(preds, labels))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def main():
    trajs, acts_list, labels = [], [], []
    for _ in range(20):
        tr, ac = make_loop_trajectory()
        trajs.append(tr); acts_list.append(ac); labels.append(True)
    for _ in range(20):
        tr, ac = make_clean_trajectory()
        trajs.append(tr); acts_list.append(ac); labels.append(False)

    all_thoughts = [th for tr in trajs for th, _ in tr]
    embedder = ThoughtEmbedder().fit(all_thoughts)

    # Honest split: calibrate on half, evaluate on the other half (stratified).
    tr_idx = [i for i, l in enumerate(labels) if l]
    cl_idx = [i for i, l in enumerate(labels) if not l]
    train_idx = tr_idx[:10] + cl_idx[:10]
    test_idx = tr_idx[10:] + cl_idx[10:]
    train_trajs = [trajs[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    test_trajs = [trajs[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    test_acts = [acts_list[i] for i in test_idx]

    # calibrate threshold on TRAIN only
    best_t, best_f1 = SemanticRevisitDetector.calibrate(
        embedder, train_trajs, train_labels,
        thresholds=np.arange(0.30, 0.96, 0.02))
    print(f"calibrated threshold: {best_t:.2f} (train F1 {best_f1:.3f})")

    sem_preds = []
    for tr in test_trajs:
        det = SemanticRevisitDetector(embedder=embedder, threshold=best_t)
        sem_preds.append(any(det.step(th, ob) is not None for th, ob in tr))
    hash_preds = [action_hash_predict(ac) for ac in test_acts]

    sp, sr, sf1 = prf(sem_preds, test_labels)
    hp, hr, hf1 = prf(hash_preds, test_labels)
    print(f"semantic : P={sp:.3f} R={sr:.3f} F1={sf1:.3f}")
    print(f"act-hash : P={hp:.3f} R={hr:.3f} F1={hf1:.3f}")
    print(f"delta F1 : {sf1 - hf1:+.3f} (kill-gate needs >= +0.10)")

    results = {
        "n_trajectories": 40, "n_train": 20, "n_test": 20,
        "calibrated_threshold": round(best_t, 2),
        "train_f1": round(best_f1, 3),
        "semantic_test": {"precision": round(sp, 3), "recall": round(sr, 3), "f1": round(sf1, 3)},
        "action_hash_test": {"precision": round(hp, 3), "recall": round(hr, 3), "f1": round(hf1, 3)},
        "delta_f1": round(sf1 - hf1, 3),
        "kill_gate_pass": bool(sf1 - hf1 >= 0.10),
        "note": "threshold calibrated on train split only; metrics on held-out test split",
    }
    with open("results/detection.json", "w") as f:
        json.dump(results, f, indent=2)
    print("KILL-GATE:", "PASS" if results["kill_gate_pass"] else "FAIL")
    return results


if __name__ == "__main__":
    main()
