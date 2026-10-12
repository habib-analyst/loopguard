"""Cost-justified iteration budgets.

Every iteration costs `cost_per_step`. The budget keeps a running estimate
of expected marginal gain (heuristic: normalized information novelty of
recent observations + tool-output usefulness) and stops when expected gain
falls below cost for `patience` consecutive steps, or at the hard cap.

On exhaustion the agent degrades gracefully: it returns the best-so-far
result together with an honest machine-readable "gave up because" summary
instead of silently returning a partial answer or spinning forever.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class BudgetExhausted:
    reason: str
    steps_used: int
    best_result: object
    history: list[str]


@dataclass
class CostJustifiedBudget:
    cost_per_step: float = 1.0
    max_steps: int = 25
    gain_floor: float = 0.05  # stop if expected gain < cost * gain_floor
    patience: int = 3
    gain_decay: float = 0.7  # how fast past gains fade in the estimator
    _steps: int = field(default=0, repr=False)
    _gain_ema: float = field(default=1.0, repr=False)
    _flat_streak: int = field(default=0, repr=False)
    _best_result: object = field(default=None, repr=False)
    _best_score: float = field(default=-math.inf, repr=False)
    _history: list[str] = field(default_factory=list, repr=False)

    def reset(self) -> None:
        self._steps = 0
        self._gain_ema = 1.0
        self._flat_streak = 0
        self._best_result = None
        self._best_score = -math.inf
        self._history.clear()

    def observe(self, gain: float, result: object = None, score: float | None = None,
                note: str = "") -> None:
        """Record the marginal gain of the last step (0..1 scale)."""
        self._steps += 1
        self._gain_ema = self.gain_decay * self._gain_ema + (1 - self.gain_decay) * gain
        if self._gain_ema < self.cost_per_step * self.gain_floor:
            self._flat_streak += 1
        else:
            self._flat_streak = 0
        if score is not None and score > self._best_score:
            self._best_score = score
            self._best_result = result
        if note:
            self._history.append(f"step {self._steps}: {note}")

    def should_stop(self) -> tuple[bool, str]:
        if self._steps >= self.max_steps:
            return True, (
                f"hard cap reached ({self._steps}/{self._max_steps_repr()} steps); "
                "returning best-so-far"
            )
        if self._flat_streak >= self.patience:
            return True, (
                f"expected marginal gain {self._gain_ema:.3f} below cost "
                f"{self.cost_per_step} for {self._flat_streak} consecutive steps; "
                "continuation not cost-justified"
            )
        return False, ""

    def _max_steps_repr(self) -> str:
        return str(self.max_steps)

    def degrade(self) -> BudgetExhausted:
        stop, reason = self.should_stop()
        return BudgetExhausted(
            reason=reason or "budget not exhausted",
            steps_used=self._steps,
            best_result=self._best_result,
            history=list(self._history),
        )

    @property
    def steps_used(self) -> int:
        return self._steps
