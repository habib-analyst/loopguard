"""Semantic revisit detection: catch the agent thinking the same thought
in different words.

A revisit is flagged when a new thought is semantically close (cosine
similarity above a calibrated threshold) to a recent thought AND the
observation has not substantively changed since then. The observation gate
is what separates a genuine loop from legitimate persistence (e.g. retrying
a flaky tool call after new evidence arrives).

Threshold calibration: SemanticRevisitDetector.calibrate() sweeps the
threshold over labeled trajectories and picks the F1-maximizing value.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .embeddings import ThoughtEmbedder

_WORD = re.compile(r"[a-z0-9]+")

# Outcome classes: what the world told the agent. The gate asks "did the
# agent learn something new?" — not "are the strings different?". Two
# different wordings of "nothing found" are the same outcome.
_OUTCOME_KEYWORDS = {
    "empty": {"0 results", "no results", "empty", "no page", "no matches",
              "0 hits", "nothing", "not found"},
    "error": {"timeout", "error", "failed", "exception", "denied"},
    "results": {"candidates", "matches found", "results for", "hits for"},
    "done": {"done", "complete", "valid", "success", "fresh"},
}


def _outcome_class(obs: str) -> str:
    low = obs.lower()
    for cls, kws in _OUTCOME_KEYWORDS.items():
        if any(k in low for k in kws):
            return cls
    return "other"


def _stem(w: str) -> str:
    # Lightweight suffix stemmer (Porter-lite): enough to conflate
    # results/result, searched/searching, returned/return.
    for suf in ("ization", "fulness", "ingly", "ation", "ness", "ment", "ing", "ed", "es", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def _tokens(text: str) -> set[str]:
    return {_stem(w) for w in _WORD.findall(text.lower())}


@dataclass
class Revisit:
    step: int
    thought: str
    matched_step: int
    matched_thought: str
    similarity: float
    observation_changed: bool


@dataclass
class SemanticRevisitDetector:
    """Rolling-window semantic loop detector."""

    embedder: ThoughtEmbedder
    window: int = 8
    threshold: float = 0.82
    obs_change_jaccard: float = 0.35  # below this => observation "unchanged"
    _thoughts: list[str] = field(default_factory=list, repr=False)
    _embeddings: list[np.ndarray] = field(default_factory=list, repr=False)
    _observations: list[str] = field(default_factory=list, repr=False)

    def reset(self) -> None:
        self._thoughts.clear()
        self._embeddings.clear()
        self._observations.clear()

    def _observation_changed(self, obs: str) -> bool:
        if not self._observations:
            return True
        # Primary signal: did the outcome class change?
        prev_cls = _outcome_class(self._observations[-1])
        cur_cls = _outcome_class(obs)
        if cur_cls != prev_cls:
            return True
        if cur_cls != "other":
            # Same known outcome (e.g. two wordings of "nothing found"):
            # the world told the agent nothing new.
            return False
        # Unknown outcome class: fall back to token novelty.
        prev = _tokens(self._observations[-1])
        cur = _tokens(obs)
        new = cur - prev
        return len(new) >= 6

    def step(self, thought: str, observation: str) -> Revisit | None:
        """Record one thought+observation; return a Revisit if a loop is found."""
        emb = self.embedder.embed(thought)
        obs_changed = self._observation_changed(observation)
        revisit = None
        if len(self._embeddings) > 0 and not obs_changed:
            sims = [ThoughtEmbedder.cosine(emb, e) for e in self._embeddings[-self.window:]]
            best = int(np.argmax(sims))
            if sims[best] >= self.threshold:
                matched_idx = len(self._thoughts) - len(sims) + best
                revisit = Revisit(
                    step=len(self._thoughts),
                    thought=thought,
                    matched_step=matched_idx,
                    matched_thought=self._thoughts[matched_idx],
                    similarity=float(sims[best]),
                    observation_changed=False,
                )
        self._thoughts.append(thought)
        self._embeddings.append(emb)
        self._observations.append(observation)
        return revisit

    @classmethod
    def calibrate(
        cls,
        embedder: ThoughtEmbedder,
        trajectories: list[list[tuple[str, str]]],
        labels: list[bool],
        thresholds: np.ndarray | None = None,
        window: int = 8,
    ) -> tuple[float, float]:
        """Sweep threshold; return (best_threshold, best_F1). A trajectory is
        predicted-loop if ANY step flags a revisit."""
        if thresholds is None:
            thresholds = np.arange(0.50, 0.96, 0.02)
        best_t, best_f1 = 0.80, -1.0
        for t in thresholds:
            preds = []
            for traj in trajectories:
                det = cls(embedder=embedder, window=window, threshold=float(t))
                flagged = any(det.step(th, ob) is not None for th, ob in traj)
                preds.append(flagged)
            tp = sum(p and l for p, l in zip(preds, labels))
            fp = sum(p and not l for p, l in zip(preds, labels))
            fn = sum((not p) and l for p, l in zip(preds, labels))
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        return best_t, best_f1
