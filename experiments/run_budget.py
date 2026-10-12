"""Experiment: ReActLoop end-to-end — task success under tight budgets.

Scripted agents (deterministic, no LLM needed):
- clean agent: progresses each step, solves at step 4.
- looping agent: repeats paraphrased thoughts, never progresses.

Compare loopguard (detect + reflect + budget) vs naive (fixed max steps,
no detection). Metrics: success rate, mean steps used, budget saved on
doomed tasks, honest termination rate.
"""
import json
import sys

sys.path.insert(0, "src")
from loopguard import (CostJustifiedBudget, ReActLoop, SemanticRevisitDetector,
                       ThoughtEmbedder)

THOUGHTS_LOOP = [
    "I should search for the quarterly revenue figure",
    "Let me try searching the quarterly revenue figure again with different terms",
    "Maybe a reworded query will find the quarterly revenue figure",
    "I'll run another search for the quarterly revenue figure, tweaking keywords",
    "Searching the quarterly revenue figure once more",
    "Let me look up the quarterly revenue figure in the docs instead",
]
THOUGHTS_CLEAN = [
    "I need the revenue figure; checking the index first",
    "Index has candidates; inspecting the first",
    "First candidate matches; verifying the number",
    "Number verified against two sources",
]
STRATEGIES = [
    "Try the finance API directly instead of web search",
    "Ask the data warehouse with a SQL query",
    "Check the quarterly report PDF",
]


def make_agent(kind):
    state = {"i": 0}

    def think(history):
        if kind == "loop":
            return THOUGHTS_LOOP[state["i"] % len(THOUGHTS_LOOP)]
        return THOUGHTS_CLEAN[min(state["i"], len(THOUGHTS_CLEAN) - 1)]

    def act(thought):
        return f"tool_call[{state['i']}]"

    def observe(action):
        state["i"] += 1
        if kind == "loop":
            return f"search: 0 results (attempt {state['i']})"
        progress = ["search: 3 candidates", "inspect: candidate matches",
                    "verify: confirmed by 2 sources", "done"][min(state["i"] - 1, 3)]
        return progress

    def score(obs):
        if kind == "loop":
            return 0.0, 0.1
        q = min(0.3 * state["i"], 1.0)
        return 0.4, q

    def reflect(trace):
        # Cycles through genuinely different strategies on repeated calls.
        s = STRATEGIES[state.get("r", 0) % len(STRATEGIES)]
        state["r"] = state.get("r", 0) + 1
        return s

    return think, act, observe, score, reflect


def run_guarded(kind):
    corpus = THOUGHTS_LOOP + THOUGHTS_CLEAN + STRATEGIES
    emb = ThoughtEmbedder().fit(corpus)
    det = SemanticRevisitDetector(embedder=emb, threshold=0.45)
    budget = CostJustifiedBudget(cost_per_step=1.0, max_steps=25, patience=3)
    loop = ReActLoop(detector=det, budget=budget, embedder=emb)
    think, act, observe, score, reflect = make_agent(kind)
    return loop.run(think, act, observe, score, reflect)


def run_naive(kind, max_steps=25):
    think, act, observe, score, reflect = make_agent(kind)
    history = []
    for _ in range(max_steps):
        th = think(history)
        ob = observe(act(th))
        _, q = score(ob)
        history.append(th)
        if q >= 1.0:
            return True, len(history)
    return False, len(history)


def main():
    rows = {}
    for kind in ["clean", "loop"]:
        g = run_guarded(kind)
        n_ok, n_steps = run_naive(kind)
        rows[kind] = {
            "guarded": {"success": g.success, "steps": g.steps,
                        "revisits": len(g.revisits), "reflections": len(g.reflections),
                        "termination": g.termination},
            "naive": {"success": n_ok, "steps": n_steps},
        }
        print(f"{kind}: guarded success={g.success} steps={g.steps} "
              f"revisits={len(g.revisits)} reflections={len(g.reflections)}")
        print(f"       naive   success={n_ok} steps={n_steps}")
        if kind == "loop":
            print(f"       termination: {g.termination[:100]}")
    # budget saved on the doomed task
    saved = rows["loop"]["naive"]["steps"] - rows["loop"]["guarded"]["steps"]
    rows["budget_saved_on_doomed_task"] = saved
    rows["budget_saved_pct"] = round(100 * saved / rows["loop"]["naive"]["steps"], 1)
    print(f"budget saved on doomed task: {saved} steps ({rows['budget_saved_pct']}%)")
    with open("results/budget.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
