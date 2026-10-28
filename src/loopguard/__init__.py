"""loopguard — semantic loop detection for ReAct agents.

Catch the agent thinking the same thought in different words, stop paying
for iterations that aren't worth it — and, since v2, climb a recovery
ladder (reflect → strategy switch → tool restriction → human handoff)
instead of giving up at the first failed reflection.
"""
from .budget import BudgetExhausted, CostJustifiedBudget
from .detector import Revisit, SemanticRevisitDetector
from .embeddings import ThoughtEmbedder
from .loop import LoopOutcome, ReActLoop
from .recovery import RecoveryEvent, RecoveryOutcome, RecoveringLoop

__all__ = [
    "ThoughtEmbedder",
    "SemanticRevisitDetector",
    "Revisit",
    "CostJustifiedBudget",
    "BudgetExhausted",
    "ReActLoop",
    "LoopOutcome",
    "RecoveringLoop",
    "RecoveryOutcome",
    "RecoveryEvent",
]

__version__ = "0.2.0"
