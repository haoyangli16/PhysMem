"""
Scientific Learning Loop Coordinator.

This is the main orchestrator that brings together:
- Experience collection and storage
- Background consolidation and hypothesis generation
- Verification planning and execution
- Principle management
- Experience folding for memory efficiency

The system operates as an async learning loop:
    Raw Experience -> [Consolidation] -> Hypotheses -> [Verification] -> Principles
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from physmem.core.experience import Experience, MemoryBank
from physmem.core.principle import Principle, PrincipleStore, PrincipleType
from physmem.core.hypothesis import (
    Hypothesis,
    HypothesisStore,
    HypothesisStatus,
)
from physmem.learning.consolidation import ConsolidationEngine, ConsolidationConfig
from physmem.learning.verification import (
    VerificationPlanner,
    VerificationConfig,
    ExperienceFolding,
)
from physmem.llm.base import BaseLLM


@dataclass
class ScientificLearningConfig:
    """Configuration for the Scientific Learning Loop."""

    # Memory
    memory_name: str = "physmem"
    max_memory_size: int = 3000

    # Consolidation
    consolidation_interval: int = 50  # Episodes between consolidation
    min_experiences_for_consolidation: int = 3
    run_consolidation_async: bool = True

    # Hypothesis
    min_cluster_size: int = 2
    min_experiences_for_hypothesis: int = 2
    max_hypotheses_per_cluster: int = 3

    # Verification
    verification_probability: float = 0.3
    promotion_confidence: float = 0.8
    refutation_confidence: float = 0.3

    # Principles
    max_principles_in_prompt: int = 5
    principle_format: str = "structured"

    # Persistence
    save_path: Optional[str] = None
    auto_save_interval: int = 100


PRINCIPLE_PROMPT_HEADER = """## Learned Principles from Experience

The following principles have been learned from past experience.
Apply them when making decisions.

"""


class ScientificLearningLoop:
    """
    Coordinator for the complete scientific learning system.

    Usage::

        from physmem import PhysMem
        from physmem.llm import create_llm

        llm = create_llm("openai", model="gpt-4o")
        loop = PhysMem(llm=llm)  # or ScientificLearningLoop(llm=llm)

        # During task execution
        eid, is_surprising = loop.record_experience(
            action="grasp_object_A",
            success=True,
            symbolic_state={"holding": True, "progress": 0.5},
        )

        # End of episode
        loop.end_episode(success=True)

        # Get learned knowledge
        principles = loop.get_principles()
        hypotheses = loop.get_hypotheses()
    """

    def __init__(
        self,
        config: Optional[ScientificLearningConfig] = None,
        memory: Optional[MemoryBank] = None,
        llm: Optional[BaseLLM] = None,
    ):
        self.config = config or ScientificLearningConfig()

        # Core stores
        self.memory = memory or MemoryBank(name=self.config.memory_name)
        self.principle_store = PrincipleStore(name=f"{self.config.memory_name}_principles")
        self.hypothesis_store = HypothesisStore(name=f"{self.config.memory_name}_hypotheses")

        # Consolidation engine
        consolidation_config = ConsolidationConfig(
            consolidation_interval=self.config.consolidation_interval,
            min_experiences_for_consolidation=self.config.min_experiences_for_consolidation,
            min_cluster_size=self.config.min_cluster_size,
            min_experiences_for_hypothesis=self.config.min_experiences_for_hypothesis,
            max_hypotheses_per_cluster=self.config.max_hypotheses_per_cluster,
            run_async=self.config.run_consolidation_async,
        )
        self.consolidation_engine = ConsolidationEngine(
            memory=self.memory,
            hypothesis_store=self.hypothesis_store,
            config=consolidation_config,
            llm=llm,
            principle_store=self.principle_store,
        )

        # Verification planner
        verification_config = VerificationConfig(
            verification_probability=self.config.verification_probability,
            promotion_confidence=self.config.promotion_confidence,
            refutation_confidence=self.config.refutation_confidence,
        )
        self.verification_planner = VerificationPlanner(
            hypothesis_store=self.hypothesis_store,
            principle_store=self.principle_store,
            config=verification_config,
        )

        # Experience folding
        self.experience_folding = ExperienceFolding(
            memory=self.memory,
            principle_store=self.principle_store,
        )

        # State
        self._episode_count = 0
        self._current_episode_experiences: List[str] = []
        self._last_save = 0

        # Start background processing
        if self.config.run_consolidation_async:
            self.consolidation_engine.start_background()

    # =========================================================================
    # Experience Recording
    # =========================================================================

    def record_experience(
        self,
        action: str,
        success: bool,
        fail: bool = False,
        fail_tag: Optional[str] = None,
        symbolic_state: Optional[Dict[str, Any]] = None,
        state_vec: Optional[np.ndarray] = None,
        oracle_action: Optional[str] = None,
        extra_metrics: Optional[Dict[str, Any]] = None,
        active_principles: Optional[List[Principle]] = None,
    ) -> Tuple[str, bool]:
        """
        Record an experience with surprise-driven filtering.

        Args:
            action: Action that was executed
            success: Whether the action succeeded
            fail: Whether the action failed
            fail_tag: Failure tag if failed
            symbolic_state: Discrete task state (user-defined dict)
            state_vec: State vector for similarity retrieval
            oracle_action: Correct action (if known)
            extra_metrics: Additional metadata
            active_principles: Principles active during this action

        Returns:
            (experience_id, is_surprising)
        """
        metrics = extra_metrics or {}
        metrics["action"] = action
        if oracle_action:
            metrics["oracle_action"] = oracle_action
        if fail_tag:
            metrics["fail_tag"] = fail_tag

        # Resonance check
        is_surprising = False
        resonance_score = 0.0
        active_principle_ids = []

        if active_principles:
            active_principle_ids = [p.pid for p in active_principles]
            resonance_result = self._check_resonance(
                active_principles, action, success, fail
            )
            is_surprising = resonance_result["is_surprising"]
            resonance_score = resonance_result["resonance_score"]

            for principle in active_principles:
                if resonance_result.get(principle.pid) == "reinforced":
                    principle.reinforce(self._episode_count)
                elif resonance_result.get(principle.pid) == "prediction_error":
                    principle.record_prediction_error(
                        self._episode_count,
                        context=f"Action: {action}, success={success}, fail={fail}",
                    )

        exp = Experience(
            task=self.config.memory_name,
            subtask="action",
            state_vec=state_vec,
            symbolic_state=symbolic_state,
            success=success,
            fail=fail,
            fail_tag=fail_tag,
            extra_metrics=metrics,
            is_surprising=is_surprising,
            resonance_score=resonance_score,
            active_principle_ids=active_principle_ids,
            creation_episode=self._episode_count,
            last_accessed_episode=self._episode_count,
        )

        eid = self.memory.add(exp)
        self._current_episode_experiences.append(eid)

        # Only send surprising experiences to consolidation
        if is_surprising or not active_principles:
            self.consolidation_engine.add_experience(exp)

        return eid, is_surprising

    def _check_resonance(
        self,
        active_principles: List[Principle],
        action: str,
        success: bool,
        fail: bool,
    ) -> Dict[str, Any]:
        """Check if experience matches active principles' predictions."""
        result = {"is_surprising": False, "resonance_score": 0.0}

        if not active_principles:
            result["is_surprising"] = True
            return result

        matches = 0
        mismatches = 0

        for principle in active_principles:
            p_type = principle.principle_type
            if p_type == PrincipleType.PREFER:
                if success:
                    result[principle.pid] = "reinforced"
                    matches += 1
                else:
                    result[principle.pid] = "prediction_error"
                    mismatches += 1
            elif p_type == PrincipleType.AVOID:
                action_type = action.split()[0].lower() if action else ""
                if action_type in principle.action_types:
                    if fail:
                        result[principle.pid] = "reinforced"
                        matches += 1
                    else:
                        result[principle.pid] = "prediction_error"
                        mismatches += 1
                else:
                    result[principle.pid] = "neutral"
            else:
                result[principle.pid] = "neutral"

        total = matches + mismatches
        if total > 0:
            result["resonance_score"] = matches / total
            result["is_surprising"] = mismatches > 0
        else:
            result["is_surprising"] = True

        return result

    # =========================================================================
    # Episode Management
    # =========================================================================

    def end_episode(
        self,
        success: bool,
        episode_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        End the current episode and perform housekeeping.

        This includes:
        - Recording verification results
        - Applying decay to principles
        - Running garbage collection if needed
        - Checking for promotions
        - Auto-saving
        """
        self._episode_count += 1
        eid = episode_id or f"ep_{self._episode_count}"

        # Verification tracking
        if self.verification_planner.is_in_verification_mode():
            self.verification_planner.record_result(success=success, episode_id=eid)

        # Sync consolidation (when not running async)
        if (
            not self.config.run_consolidation_async
            and self._episode_count % self.config.consolidation_interval == 0
        ):
            clusters = self.consolidation_engine.consolidate()
            if clusters:
                self.consolidation_engine.generate_hypotheses(clusters)

        # Passive verification: test PROPOSED hypotheses against recent
        # experiences and resolve PREFER/AVOID contradictions. This is
        # what enables verified hypotheses to actually flow into the
        # principle store via check_promotions() below.
        self.verification_planner.passive_verify_all(
            memory=self.memory,
            recent_window=self.config.max_memory_size,
        )
        self.verification_planner.resolve_contradictions()

        # Decay all principles
        for principle in self.principle_store.principles:
            principle.apply_decay(decay_factor=0.995)

        # Garbage collection
        if len(self.memory.experiences) > self.config.max_memory_size:
            self._run_garbage_collection()
        elif self._episode_count % 50 == 0:
            self._run_garbage_collection()

        # Promotions
        promoted = self.check_promotions()

        # Auto-save
        if (
            self.config.save_path
            and self._episode_count - self._last_save >= self.config.auto_save_interval
        ):
            self.save_state()
            self._last_save = self._episode_count

        episode_exp_count = len(self._current_episode_experiences)
        self._current_episode_experiences = []

        return {
            "episode": self._episode_count,
            "success": success,
            "experiences_recorded": episode_exp_count,
            "total_experiences": len(self.memory),
            "total_principles": len(self.principle_store),
            "total_hypotheses": len(self.hypothesis_store),
            "promoted": len(promoted) if promoted else 0,
        }

    # =========================================================================
    # Promotion & Maintenance
    # =========================================================================

    def check_promotions(self) -> List[Principle]:
        """Check and promote verified hypotheses to principles."""
        return self.verification_planner.promote_verified()

    def _run_garbage_collection(self) -> Dict[str, int]:
        """Run garbage collection on memory and principles."""
        # Prune low-confidence principles
        pruned_principles = self.principle_store.prune()

        # Fold experiences covered by established principles
        folded = self.experience_folding.fold_covered_experiences(max_fold=50)

        # Archive old folded experiences if over limit
        pruned_exp = 0
        if len(self.memory.experiences) > self.config.max_memory_size:
            # Remove oldest folded experiences first
            to_remove = []
            for i, exp in enumerate(self.memory.experiences):
                if exp.memory_status in ("folded", "archived"):
                    to_remove.append(i)
                if len(self.memory.experiences) - len(to_remove) <= self.config.max_memory_size:
                    break

            for i in reversed(to_remove):
                self.memory.experiences.pop(i)
                pruned_exp += 1

            if pruned_exp > 0:
                # Rebuild indices
                self.memory._eid_to_idx = {
                    exp.eid: i for i, exp in enumerate(self.memory.experiences)
                }
                self.memory.rebuild_symbolic_index()

        return {
            "principles_archived": pruned_principles,
            "experiences_pruned": pruned_exp,
            "experiences_folded": folded,
        }

    # =========================================================================
    # Query API
    # =========================================================================

    def get_principles(
        self,
        action_type: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Principle]:
        """Get learned principles, optionally filtered by action type."""
        top_k = top_k or self.config.max_principles_in_prompt
        return self.principle_store.retrieve(action_type=action_type, top_k=top_k)

    def get_hypotheses(
        self,
        status: Optional[str] = None,
    ) -> List[Hypothesis]:
        """Get current hypotheses, optionally filtered by status."""
        if status:
            try:
                return self.hypothesis_store.get_by_status(HypothesisStatus(status))
            except ValueError:
                return []
        return self.hypothesis_store.hypotheses

    def get_principles_prompt(
        self,
        action_type: Optional[str] = None,
        max_principles: Optional[int] = None,
    ) -> str:
        """Get principles formatted for LLM prompt injection."""
        max_p = max_principles or self.config.max_principles_in_prompt
        principles = self.get_principles(action_type=action_type, top_k=max_p)
        if not principles:
            return ""
        return PRINCIPLE_PROMPT_HEADER + self.principle_store.format_for_prompt(
            principles, max_principles=max_p
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "episode_count": self._episode_count,
            "memory": self.memory.get_stats(),
            "principles": self.principle_store.get_stats(),
            "hypotheses": self.hypothesis_store.get_stats(),
            "consolidation": self.consolidation_engine.get_stats(),
            "verification": self.verification_planner.get_stats(),
            "folding": self.experience_folding.get_stats(),
        }

    # =========================================================================
    # Persistence
    # =========================================================================

    def save_state(self, path: Optional[str] = None) -> None:
        """Save full state to disk."""
        save_dir = Path(path or self.config.save_path or "./physmem_state")
        save_dir.mkdir(parents=True, exist_ok=True)

        self.memory.save(save_dir / "memory.json")
        self.hypothesis_store.save(save_dir / "hypotheses.json")
        self.principle_store.save(save_dir / "principles.json")

        # Save metadata
        metadata = {
            "episode_count": self._episode_count,
            "timestamp": datetime.now().isoformat(),
            "stats": self.get_stats(),
        }
        with open(save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load_state(
        cls,
        path: str,
        llm: Optional[BaseLLM] = None,
        config: Optional[ScientificLearningConfig] = None,
    ) -> "ScientificLearningLoop":
        """Load state from disk."""
        load_dir = Path(path)

        memory = MemoryBank.load(load_dir / "memory.json")
        loop = cls(config=config, memory=memory, llm=llm)

        if (load_dir / "hypotheses.json").exists():
            loop.hypothesis_store = HypothesisStore.load(load_dir / "hypotheses.json")
        if (load_dir / "principles.json").exists():
            loop.principle_store = PrincipleStore.load(load_dir / "principles.json")

        if (load_dir / "metadata.json").exists():
            with open(load_dir / "metadata.json") as f:
                metadata = json.load(f)
            loop._episode_count = metadata.get("episode_count", 0)

        return loop

    def shutdown(self):
        """Gracefully stop background processing."""
        self.consolidation_engine.stop_background()
        if self.config.save_path:
            self.save_state()
