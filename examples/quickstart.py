"""loopguard quickstart: wrap a ReAct agent, catch semantic loops, stop
paying for doomed iterations."""
import sys

sys.path.insert(0, "src")

from loopguard import (CostJustifiedBudget, ReActLoop, SemanticRevisitDetector,
                       ThoughtEmbedder)

# 1. Fit the embedder on representative thoughts (or your agent's history).
corpus = [
    "I should search for the quarterly revenue figure",
    "Let me try searching the quarterly revenue figure again",
    "The index has candidates; I'll inspect the first one",
    "Candidate matches; verifying the checksum now",
]
embedder = ThoughtEmbedder().fit(corpus)

# 2. Build the guard: semantic detector + cost-justified budget.
detector = SemanticRevisitDetector(embedder=embedder, threshold=0.45)
budget = CostJustifiedBudget(cost_per_step=1.0, max_steps=25, patience=3)
loop = ReActLoop(detector=detector, budget=budget, embedder=embedder)

# 3. Plug in your agent's callables (here: a scripted agent stuck in a loop).
thoughts = ["I should search for the quarterly revenue figure",
            "Let me try searching the quarterly revenue figure again",
            "Maybe a reworded query will find it"]
state = {"i": 0}


def think(history):
    return thoughts[state["i"] % len(thoughts)]


def observe(action):
    state["i"] += 1
    return f"search: 0 results (attempt {state['i']})"


outcome = loop.run(
    think=think,
    act=lambda thought: "search_tool",
    observe=observe,
    score=lambda obs: (0.0, 0.1),  # (marginal gain, task quality)
    reflect=lambda trace: "Try the finance API directly instead of web search",
)

print(f"success:     {outcome.success}")
print(f"steps used:  {outcome.steps}")
print(f"revisits:    {len(outcome.revisits)}")
print(f"reflections: {len(outcome.reflections)}")
print(f"termination: {outcome.termination[:90]}...")
