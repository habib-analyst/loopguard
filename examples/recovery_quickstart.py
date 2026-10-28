"""loopguard v2 quickstart: a stuck agent whose reflections keep coming
back as rephrased searches. v1 gives up after a handful of reflections;
the v2 ladder keeps climbing (reflect, with annotated retries on copycat
proposals, then switch, restriction, handoff) until a genuinely different
strategy lands."""
import sys

sys.path.insert(0, "src")

from loopguard import (CostJustifiedBudget, RecoveringLoop,
                       SemanticRevisitDetector, ThoughtEmbedder)

corpus = [
    "I should search for the quarterly revenue figure",
    "Let me try searching the quarterly revenue figure again",
    "Maybe a reworded query will find the quarterly revenue figure",
    "Search again with broader keywords and synonyms for the figure",
    "Search the archive once more with wider keyword variants",
    "Query the finance data API directly for the structured record",
    "The API returned candidate records; checking the Q3 entry",
    "Verifying the API figure against the filing",
]
embedder = ThoughtEmbedder().fit(corpus)
detector = SemanticRevisitDetector(embedder=embedder, threshold=0.30)
budget = CostJustifiedBudget(cost_per_step=1.0, max_steps=40, patience=99)
loop = RecoveringLoop(detector=detector, budget=budget, embedder=embedder)

LOOP_THOUGHTS = corpus[0:3]
WORK_THOUGHTS = corpus[5:8]
PROPOSALS = [
    "Search again with broader keywords and synonyms for the figure",
    "Search the archive once more with wider keyword variants",
    "Query the finance data API directly for the structured record",
]
PROGRESS_OBS = ["api: 3 candidates for the revenue record",
                "inspect: candidate matches the requested figure",
                "verify: figure confirmed — done"]
state = {"ti": 0, "ri": 0, "progress": 0, "mode": "loop"}


def think(history):
    # The agent adopts a strategy when the loop's directive line names it.
    state["mode"] = "loop"
    for line in reversed(history):
        if line.startswith("RECOVERY"):
            if PROPOSALS[2] in line:
                state["mode"] = "work"
            break
    pool = WORK_THOUGHTS if state["mode"] == "work" else LOOP_THOUGHTS
    th = pool[state["ti"] % len(pool)]
    state["ti"] += 1
    return th


def act(thought):
    return "finance_api[revenue]" if state["mode"] == "work" else "search[revenue]"


def observe(action):
    if state["mode"] == "work":
        state["progress"] += 1
        return PROGRESS_OBS[state["progress"] - 1]
    return "search: 0 results for 'revenue'"


def score(obs):
    if "done" in obs:
        return (0.5, 1.0)
    if "candidates" in obs:
        return (0.4, 0.3)
    if "matches" in obs:
        return (0.4, 0.6)
    return (0.0, 0.1)


def reflect(trace):
    p = PROPOSALS[min(state["ri"], len(PROPOSALS) - 1)]
    state["ri"] += 1
    return p


outcome = loop.run(think=think, act=act, observe=observe, score=score,
                   reflect=reflect)

print(f"success:     {outcome.success}")
print(f"steps used:  {outcome.steps}")
print(f"revisits:    {len(outcome.revisits)}")
print(f"max rung:    {outcome.max_rung}")
for e in outcome.recovery_events:
    print(f"  rung {e.rung} {e.action}: {e.detail[:80]}")
print(f"termination: {outcome.termination}")
