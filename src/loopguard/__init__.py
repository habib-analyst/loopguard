"""loopguard — semantic loop detection for ReAct agents.

Catch the agent thinking the same thought in different words, and stop
paying for iterations that aren't worth it.
"""
from .budget import BudgetExhausted, CostJustifiedBudget
from .detector import Revisit, SemanticRevisitDetector
from .embeddings import ThoughtEmbedder
from .loop import LoopOutcome, ReActLoop

__all__ = [
    "ThoughtEmbedder",
    "SemanticRevisitDetector",
    "Revisit",
    "CostJustifiedBudget",
    "BudgetExhausted",
    "ReActLoop",
    "LoopOutcome",
]

__version__ = "0.1.0"
