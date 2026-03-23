"""
PhysMem - Physical Memory System for Experience-to-Principle Learning.

A test-time training memory system that:
1. Collects raw experiences during task execution
2. Discovers patterns through background consolidation
3. Generates hypotheses about causal relationships
4. Verifies hypotheses through directed experiments
5. Promotes verified knowledge into transferable principles
6. Compresses memory by folding old experiences into principles

Quick Start::

    from physmem import PhysMem
    from physmem.llm import create_llm

    # Create with any LLM backend
    llm = create_llm("openai", model="gpt-4o")
    mem = PhysMem(llm=llm)

    # Record experiences during task execution
    mem.record_experience(
        action="grasp_object",
        success=True,
        symbolic_state={"holding": True, "target": "block_A"},
    )

    # End episode
    mem.end_episode(success=True)

    # Get learned knowledge
    principles = mem.get_principles()
    hypotheses = mem.get_hypotheses()

    # Get principles formatted for LLM prompt injection
    prompt_text = mem.get_principles_prompt()
"""

__version__ = "0.1.0"

# Main entry point (alias for ScientificLearningLoop)
from physmem.learning.loop import ScientificLearningLoop as PhysMem
from physmem.learning.loop import ScientificLearningConfig

# Core data structures
from physmem.core.experience import Experience, MemoryBank
from physmem.core.hypothesis import (
    Hypothesis,
    HypothesisStore,
    HypothesisStatus,
    ExperienceCluster,
)
from physmem.core.principle import Principle, PrincipleStore, PrincipleType

# LLM interface
from physmem.llm.base import BaseLLM

__all__ = [
    # Main API
    "PhysMem",
    "ScientificLearningConfig",
    # Core
    "Experience",
    "MemoryBank",
    "Hypothesis",
    "HypothesisStore",
    "HypothesisStatus",
    "ExperienceCluster",
    "Principle",
    "PrincipleStore",
    "PrincipleType",
    # LLM
    "BaseLLM",
]
