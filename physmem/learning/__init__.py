"""PhysMem Learning - Scientific learning loop, consolidation, and verification."""

from physmem.learning.consolidation import ConsolidationEngine, ConsolidationConfig
from physmem.learning.verification import VerificationPlanner, VerificationConfig, ExperienceFolding
from physmem.learning.loop import ScientificLearningLoop, ScientificLearningConfig

__all__ = [
    "ConsolidationEngine",
    "ConsolidationConfig",
    "VerificationPlanner",
    "VerificationConfig",
    "ExperienceFolding",
    "ScientificLearningLoop",
    "ScientificLearningConfig",
]
