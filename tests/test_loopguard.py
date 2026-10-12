import sys

sys.path.insert(0, "src")
import numpy as np
import pytest

from loopguard import (BudgetExhausted, CostJustifiedBudget, ReActLoop,
                       SemanticRevisitDetector, ThoughtEmbedder)
from loopguard.detector import _outcome_class

CORPUS = [
    "I should search for the quarterly revenue figure",
    "Let me try searching the quarterly revenue figure again",
    "Maybe a reworded query will find the quarterly revenue figure",
    "The index has candidates; I'll inspect the first one",
    "Candidate matches; verifying the checksum now",
    "search: 0 results for 'revenue'",
    "docs: no page matching 'revenue'",
    "search: 3 candidates",
]


@pytest.fixture()
def emb():
    return ThoughtEmbedder().fit(CORPUS)


def test_embedder_deterministic(emb):
    a = emb.embed("search for the revenue figure")
    b = emb.embed("search for the revenue figure")
    assert np.allclose(a, b)


def test_embedder_paraphrase_gt_unrelated(emb):
    p1 = emb.embed("I should search for the quarterly revenue figure")
    p2 = emb.embed("Let me try searching the quarterly revenue figure again")
    u = emb.embed("Candidate matches; verifying the checksum now")
    assert ThoughtEmbedder.cosine(p1, p2) > ThoughtEmbedder.cosine(p1, u)


def test_outcome_classes():
    assert _outcome_class("search: 0 results for 'x'") == "empty"
    assert _outcome_class("docs: no page matching 'x'") == "empty"
    assert _outcome_class("search: timeout after 30s") == "error"
    assert _outcome_class("search: 3 candidates") == "results"
    assert _outcome_class("verify: fresh — done") == "done"


def test_detector_flags_semantic_revisit(emb):
    det = SemanticRevisitDetector(embedder=emb, threshold=0.40)
    assert det.step("I should search for the quarterly revenue figure",
                    "search: 0 results for 'revenue'") is None
    r = det.step("Let me try searching the quarterly revenue figure again",
                 "docs: no page matching 'revenue'")
    assert r is not None
    assert r.similarity >= 0.40
    assert r.observation_changed is False


def test_detector_no_flag_when_observation_changes(emb):
    det = SemanticRevisitDetector(embedder=emb, threshold=0.40)
    det.step("I should search for the quarterly revenue figure",
             "search: timeout after 30s")
    # Same thought, but the world gave new information -> not a loop.
    r = det.step("I should search for the quarterly revenue figure",
                 "search: 3 candidates")
    assert r is None


def test_detector_no_flag_for_progressing_thoughts(emb):
    det = SemanticRevisitDetector(embedder=emb, threshold=0.40)
    steps = [
        ("I should search for the quarterly revenue figure", "search: 0 results for 'revenue'"),
        ("The index has candidates; I'll inspect the first one", "search: 3 candidates"),
        ("Candidate matches; verifying the checksum now", "verify: fresh — done"),
    ]
    assert all(det.step(t, o) is None for t, o in steps)


def test_budget_hard_cap():
    b = CostJustifiedBudget(cost_per_step=1.0, max_steps=5, patience=99)
    for _ in range(5):
        b.observe(gain=1.0, note="great progress")
    stop, reason = b.should_stop()
    assert stop and "hard cap" in reason


def test_budget_cost_justification():
    b = CostJustifiedBudget(cost_per_step=1.0, max_steps=100, patience=2,
                            gain_decay=0.0)
    b.observe(gain=0.0, note="nothing")
    b.observe(gain=0.0, note="nothing")
    stop, reason = b.should_stop()
    assert stop and "cost" in reason


def test_budget_keeps_going_when_gainful():
    b = CostJustifiedBudget(cost_per_step=1.0, max_steps=100, patience=3)
    for _ in range(10):
        b.observe(gain=0.9, note="progress")
    stop, _ = b.should_stop()
    assert not stop


def test_budget_degrade_returns_best_so_far():
    b = CostJustifiedBudget(cost_per_step=1.0, max_steps=2, patience=99)
    b.observe(gain=0.5, result="partial answer", score=0.6, note="half there")
    b.observe(gain=0.5, result="partial answer", score=0.6, note="half there")
    d = b.degrade()
    assert isinstance(d, BudgetExhausted)
    assert d.best_result == "partial answer"
    assert "hard cap" in d.reason


def test_react_loop_solves_clean_task(emb):
    det = SemanticRevisitDetector(embedder=emb, threshold=0.45)
    budget = CostJustifiedBudget(max_steps=25)
    loop = ReActLoop(detector=det, budget=budget, embedder=emb)
    obs_seq = ["search: 3 candidates", "inspect: matches", "verify: fresh — done"]
    state = {"i": 0}

    def think(h): return ["find it", "inspect it", "verify it"][min(state["i"], 2)]
    def act(t): return "tool"
    def observe(a):
        o = obs_seq[min(state["i"], 2)]; state["i"] += 1; return o
    def score(o): return (0.5, 1.0 if "done" in o else 0.3)
    def reflect(t): return "different strategy entirely"

    out = loop.run(think, act, observe, score, reflect)
    assert out.success and out.steps == 3 and not out.revisits


def test_react_loop_rejects_copycat_reflection(emb):
    det = SemanticRevisitDetector(embedder=emb, threshold=0.30)
    budget = CostJustifiedBudget(max_steps=50, patience=99)
    loop = ReActLoop(detector=det, budget=budget, embedder=emb,
                     reflection_dissimilarity=0.90)
    thoughts = ["I should search for the quarterly revenue figure",
                "Let me try searching the quarterly revenue figure again"]

    def think(h): return thoughts[len(h) % 2]
    def act(t): return "tool"
    def observe(a): return "search: 0 results for 'revenue'"
    def score(o): return (0.0, 0.1)
    def reflect(t): return "search for the quarterly revenue figure again"

    out = loop.run(think, act, observe, score, reflect)
    assert not out.success
    assert "too similar" in out.termination
    assert len(out.revisits) >= 1
