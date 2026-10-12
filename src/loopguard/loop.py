"""ReAct loop with semantic revisit detection, loop reflection, and
cost-justified budgets.

The agent is driven by user-supplied callables (think/act/observe/score),
so loopguard wraps ANY ReAct-style agent without framework lock-in:

    loop = ReActLoop(detector=..., budget=...)
    outcome = loop.run(
        think=agent.think,      # (history) -> thought string
        act=agent.act,          # (thought) -> action string
        observe=env.step,       # (action) -> observation string
        score=lambda obs: ...,  # (observation) -> (gain 0..1, quality score)
        reflect=agent.reflect,  # (loop_trace) -> new strategy string
    )

On a semantic revisit, the loop trace (similar thoughts + what was tried)
is injected into reflect(), which must return a strategy whose embedding
is DISSIMILAR to the failed one — otherwise the reflection is rejected
and the loop terminates instead of re-entering the same reasoning.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .budget import BudgetExhausted, CostJustifiedBudget
from .detector import Revisit, SemanticRevisitDetector
from .embeddings import ThoughtEmbedder


@dataclass
class LoopOutcome:
    success: bool
    result: object
    steps: int
    revisits: list[Revisit] = field(default_factory=list)
    reflections: list[str] = field(default_factory=list)
    termination: str = ""


class ReActLoop:
    def __init__(
        self,
        detector: SemanticRevisitDetector,
        budget: CostJustifiedBudget,
        embedder: ThoughtEmbedder,
        reflection_dissimilarity: float = 0.55,
        max_reflections: int = 3,
    ):
        self.detector = detector
        self.budget = budget
        self.embedder = embedder
        self.reflection_dissimilarity = reflection_dissimilarity
        self.max_reflections = max_reflections

    def run(self, think, act, observe, score, reflect) -> LoopOutcome:
        self.detector.reset()
        self.budget.reset()
        history: list[str] = []
        revisits: list[Revisit] = []
        reflections: list[str] = []
        failed_strategies: list = []
        n_reflections = 0

        while True:
            thought = think(history)
            action = act(thought)
            observation = observe(action)
            gain, quality = score(observation)
            self.budget.observe(gain, result=observation, score=quality,
                                note=f"thought={thought[:60]!r} gain={gain:.2f}")

            revisit = self.detector.step(thought, observation)
            history.append(f"thought: {thought} | action: {action} | obs: {observation}")

            if revisit is not None:
                revisits.append(revisit)
                if n_reflections >= self.max_reflections:
                    term = (f"terminated: semantic loop re-detected after "
                            f"{n_reflections} reflections; giving up honestly")
                    return LoopOutcome(False, self.budget._best_result,
                                       self.budget.steps_used, revisits, reflections, term)
                trace = (
                    f"LOOP DETECTED at step {revisit.step} (similarity "
                    f"{revisit.similarity:.2f} to step {revisit.matched_step}).\n"
                    f"Earlier thought: {revisit.matched_thought}\n"
                    f"Repeated thought: {revisit.thought}\n"
                    f"History so far:\n" + "\n".join(history[-6:])
                )
                new_strategy = reflect(trace)
                # Reflection must be genuinely different, not a rephrase.
                if failed_strategies:
                    sims = [ThoughtEmbedder.cosine(self.embedder.embed(new_strategy), s)
                            for s in failed_strategies]
                    if max(sims) >= self.reflection_dissimilarity:
                        term = ("terminated: reflection proposed a strategy too similar "
                                f"(max sim {max(sims):.2f}) to a failed one; refusing to re-loop")
                        return LoopOutcome(False, self.budget._best_result,
                                           self.budget.steps_used, revisits, reflections, term)
                failed_strategies.append(self.embedder.embed(new_strategy))
                reflections.append(new_strategy)
                history.append(f"REFLECTION (new strategy): {new_strategy}")
                n_reflections += 1
                # Fresh detector window so the new strategy gets a fair run.
                self.detector.reset()

            stop, reason = self.budget.should_stop()
            if stop:
                exhausted: BudgetExhausted = self.budget.degrade()
                return LoopOutcome(
                    success=False,
                    result=exhausted.best_result,
                    steps=self.budget.steps_used,
                    revisits=revisits,
                    reflections=reflections,
                    termination=f"budget exhausted: {exhausted.reason}",
                )
            if quality is not None and quality >= 1.0:
                return LoopOutcome(True, observation, self.budget.steps_used,
                                    revisits, reflections, "task solved")
