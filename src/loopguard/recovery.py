"""Recovery ladder: what happens AFTER a loop is detected (loopguard v2).

v1 detects a semantic loop and tries a single reflection; if a later
reflection is a rephrase of a failed strategy, v1 terminates on the spot.
That is honest but leaves recoverable agents on the floor. v2 instead
climbs a ladder of concrete recovery actions at every detection, escalating
whenever a rung fails:

    rung 1  reflect          loop-trace reflection (as v1). A copycat
                             proposal is not fatal: the trace is annotated
                             ("previous proposal too similar") and reflect()
                             is asked again, up to max_reflect_retries extra
                             proposals, before escalating.
    rung 2  strategy_switch  a dedicated switch callable that must return a
                             strategy from a different family; still subject
                             to the dissimilarity check against failed
                             strategies.
    rung 3  tool_restriction the loop restricts the tools that appear in the
                             detected loop window. Actions using a restricted
                             tool are intercepted — the environment is never
                             called — and the agent gets a restriction
                             notice, forcing a different tool.
    rung 4  handoff          human-handoff stub: terminate with a structured
                             packet (history tail, strategies tried, tools
                             restricted, recovery events). A handoff is a
                             graceful outcome, NOT a task completion.

Escalation across detections is carried by persistent state: the
failed-strategy set only grows, and restricted tools stay restricted for
the rest of the run, so a later detection starts with the earlier rungs
already spent. The cost-justified budget remains the backstop throughout.

    loop = RecoveringLoop(detector=..., budget=..., embedder=...)
    outcome = loop.run(think=..., act=..., observe=..., score=...,
                       reflect=agent.reflect,
                       switch_strategy=agent.switch_strategy)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .budget import CostJustifiedBudget
from .detector import Revisit, SemanticRevisitDetector
from .embeddings import ThoughtEmbedder

_TOOL_SPLIT = re.compile(r"[\[\(:\s]")


def tool_of(action: str) -> str:
    """Tool name of an action string: the token before '[', '(', ':' or
    whitespace. 'search[revenue]' -> 'search'; 'run_tests' -> 'run_tests'."""
    return _TOOL_SPLIT.split(action.strip(), 1)[0]


@dataclass
class RecoveryEvent:
    step: int
    rung: int
    action: str  # reflect | reflect_retry | strategy_switch | tool_restriction | handoff
    detail: str
    accepted: bool = True


@dataclass
class RecoveryOutcome:
    success: bool
    result: object
    steps: int
    revisits: list[Revisit] = field(default_factory=list)
    reflections: list[str] = field(default_factory=list)
    termination: str = ""
    recovery_events: list[RecoveryEvent] = field(default_factory=list)
    handed_off: bool = False
    handoff_packet: dict | None = None
    restricted_tools: list[str] = field(default_factory=list)
    max_rung: int = 0


class RecoveringLoop:
    """ReAct loop with semantic detection + the recovery ladder."""

    def __init__(
        self,
        detector: SemanticRevisitDetector,
        budget: CostJustifiedBudget,
        embedder: ThoughtEmbedder,
        reflection_dissimilarity: float = 0.55,
        max_reflect_retries: int = 1,
        max_recovery_events: int = 12,
    ):
        self.detector = detector
        self.budget = budget
        self.embedder = embedder
        self.reflection_dissimilarity = reflection_dissimilarity
        self.max_reflect_retries = max_reflect_retries
        self.max_recovery_events = max_recovery_events

    # -- helpers ---------------------------------------------------------

    def _too_similar(self, proposal: str, failed_embs: list) -> float:
        """Max cosine similarity of a proposal to any failed strategy
        (0.0 when nothing has failed yet or the proposal is empty)."""
        if not failed_embs or not proposal.strip():
            return 0.0
        emb = self.embedder.embed(proposal)
        return max(ThoughtEmbedder.cosine(emb, e) for e in failed_embs)

    # -- main loop ---------------------------------------------------------

    def run(self, think, act, observe, score, reflect=None,
            switch_strategy=None) -> RecoveryOutcome:
        self.detector.reset()
        self.budget.reset()
        history: list[str] = []
        revisits: list[Revisit] = []
        reflections: list[str] = []
        events: list[RecoveryEvent] = []
        failed_embs: list = []
        restricted: list[str] = []
        restricted_set: set[str] = set()
        window_actions: list[str] = []

        def outcome(success, result, termination, handed_off=False, packet=None):
            return RecoveryOutcome(
                success=success, result=result, steps=self.budget.steps_used,
                revisits=revisits, reflections=reflections,
                termination=termination, recovery_events=events,
                handed_off=handed_off, handoff_packet=packet,
                restricted_tools=list(restricted),
                max_rung=max((e.rung for e in events), default=0),
            )

        while True:
            thought = think(history)
            action = act(thought)
            tool = tool_of(action)
            if tool in restricted_set:
                # Rung-3 interception: the environment is never called.
                observation = (
                    f"system: tool '{tool}' is restricted by loopguard after "
                    "repeated loops; it cannot run — choose a different tool"
                )
            else:
                observation = observe(action)
            gain, quality = score(observation)
            self.budget.observe(gain, result=observation, score=quality,
                                note=f"thought={thought[:60]!r} gain={gain:.2f}")
            revisit = self.detector.step(thought, observation)
            history.append(
                f"thought: {thought} | action: {action} | obs: {observation}")
            window_actions.append(action)

            if quality is not None and quality >= 1.0:
                return outcome(True, observation, "task solved")

            if revisit is not None:
                revisits.append(revisit)
                result = self._climb_ladder(
                    revisit=revisit, n_revisits=len(revisits), history=history,
                    reflect=reflect, switch_strategy=switch_strategy,
                    failed_embs=failed_embs, reflections=reflections,
                    events=events, restricted=restricted,
                    restricted_set=restricted_set,
                    window_actions=window_actions, outcome=outcome)
                if result is not None:
                    return result
                window_actions = []

            stop, _reason = self.budget.should_stop()
            if stop:
                exhausted = self.budget.degrade()
                return outcome(False, exhausted.best_result,
                               f"budget exhausted: {exhausted.reason}")

    # -- the ladder ---------------------------------------------------------

    def _climb_ladder(self, *, revisit, n_revisits, history, reflect,
                      switch_strategy, failed_embs, reflections, events,
                      restricted, restricted_set, window_actions, outcome):
        """Handle one detection, climbing rungs until one is accepted.
        Returns a RecoveryOutcome to terminate the run, or None to continue."""
        step = self.budget.steps_used
        trace = (
            f"LOOP DETECTED at step {revisit.step} (similarity "
            f"{revisit.similarity:.2f} to step {revisit.matched_step}).\n"
            f"Earlier thought: {revisit.matched_thought}\n"
            f"Repeated thought: {revisit.thought}\n"
            f"History so far:\n" + "\n".join(history[-6:])
        )

        def spent():
            return len(events) >= self.max_recovery_events

        # Rung 1 — reflect, with bounded annotated retries on rejection.
        if reflect is not None:
            for i in range(1 + self.max_reflect_retries):
                if spent():
                    return outcome(
                        False, self.budget._best_result,
                        f"terminated: recovery ladder stopped after "
                        f"{len(events)} recovery actions (max_recovery_events)")
                t = trace
                if i > 0:
                    t += ("\nYour previous proposal was rejected: too similar "
                          "to a strategy that already failed. Propose a "
                          "genuinely different strategy.")
                proposal = reflect(t)
                sim = self._too_similar(proposal, failed_embs)
                ok = bool(proposal.strip()) and sim < self.reflection_dissimilarity
                events.append(RecoveryEvent(
                    step=step, rung=1,
                    action="reflect" if i == 0 else "reflect_retry",
                    detail=(f"proposal {proposal[:80]!r} max-sim {sim:.2f} "
                            f"-> {'accepted' if ok else 'rejected'}"),
                    accepted=ok))
                if ok:
                    failed_embs.append(self.embedder.embed(proposal))
                    reflections.append(proposal)
                    history.append(
                        f"RECOVERY [rung 1 reflect] new strategy: {proposal}")
                    self.detector.reset()
                    return None

        # Rung 2 — explicit strategy switch (different family).
        if switch_strategy is not None:
            if spent():
                return outcome(
                    False, self.budget._best_result,
                    f"terminated: recovery ladder stopped after "
                    f"{len(events)} recovery actions (max_recovery_events)")
            proposal = switch_strategy(trace)
            sim = self._too_similar(proposal, failed_embs)
            ok = bool(proposal.strip()) and sim < self.reflection_dissimilarity
            events.append(RecoveryEvent(
                step=step, rung=2, action="strategy_switch",
                detail=(f"proposal {proposal[:80]!r} max-sim {sim:.2f} "
                        f"-> {'accepted' if ok else 'rejected'}"),
                accepted=ok))
            if ok:
                failed_embs.append(self.embedder.embed(proposal))
                reflections.append(proposal)
                history.append(
                    f"RECOVERY [rung 2 strategy_switch] new strategy: {proposal}")
                self.detector.reset()
                return None

        # Rung 3 — restrict the tools that appear in the loop window.
        tools = []
        for a in window_actions:
            t = tool_of(a)
            if t not in tools:
                tools.append(t)
        new_tools = [t for t in tools if t not in restricted_set]
        if new_tools:
            if spent():
                return outcome(
                    False, self.budget._best_result,
                    f"terminated: recovery ladder stopped after "
                    f"{len(events)} recovery actions (max_recovery_events)")
            restricted.extend(new_tools)
            restricted_set.update(new_tools)
            events.append(RecoveryEvent(
                step=step, rung=3, action="tool_restriction",
                detail=f"restricted tools: {', '.join(new_tools)}"))
            history.append(
                "RECOVERY [rung 3 tool_restriction] restricted tools: "
                + ", ".join(new_tools)
                + " — these tools cannot run; use a different tool")
            self.detector.reset()
            return None

        # Rung 4 — human-handoff stub with a structured trace packet.
        events.append(RecoveryEvent(
            step=step, rung=4, action="handoff",
            detail="human-handoff stub: trace packet emitted"))
        packet = {
            "reason": ("semantic loop persisted through reflect, "
                       "strategy_switch and tool_restriction rungs"),
            "steps": self.budget.steps_used,
            "revisits": n_revisits,
            "strategies_tried": list(reflections),
            "restricted_tools": list(restricted),
            "recovery_events": [vars(e) for e in events],
            "history_tail": history[-6:],
        }
        return outcome(
            False, self.budget._best_result,
            "handoff: recovery ladder exhausted (reflect, strategy_switch, "
            "tool_restriction all failed to break the loop); handing off to "
            "a human with a trace packet",
            handed_off=True, packet=packet)
