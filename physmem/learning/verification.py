"""
Verification Planner for the Scientific Learning Loop.

Responsibilities:
1. Designing experiments to test hypotheses
2. Tracking verification progress
3. Analyzing results and updating hypothesis confidence
4. Promoting verified hypotheses to principles
5. Experience folding for memory compression
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path

import numpy as np

from physmem.core.hypothesis import (
    Hypothesis,
    HypothesisStore,
    VerificationPlan,
    HypothesisStatus,
    PrincipleType,
)
from physmem.core.principle import Principle, PrincipleStore
from physmem.core.experience import MemoryBank


@dataclass
class VerificationConfig:
    """Configuration for the verification planner."""
    verification_probability: float = 0.3
    max_episodes_per_hypothesis: int = 5
    min_episodes_per_hypothesis: int = 2
    promotion_confidence: float = 0.8
    refutation_confidence: float = 0.3
    min_verifications_for_promotion: int = 2
    selection_strategy: str = "ucb"  # "ucb", "oldest", "highest_confidence"
    ucb_exploration: float = 1.0


class VerificationPlanner:
    """
    Planner for designing and executing hypothesis verification.

    Usage::

        planner = VerificationPlanner(hypothesis_store, principle_store)

        # During episode
        plan = planner.get_next_verification()
        if plan:
            constraint = plan.get_current_condition()
            ...
            planner.record_result(success, episode_id)

        # Periodically
        new_principles = planner.promote_verified()
    """

    def __init__(
        self,
        hypothesis_store: HypothesisStore,
        principle_store: PrincipleStore,
        config: Optional[VerificationConfig] = None,
    ):
        self.hypothesis_store = hypothesis_store
        self.principle_store = principle_store
        self.config = config or VerificationConfig()
        self.current_plan: Optional[VerificationPlan] = None
        self._episodes_since_verification = 0
        self._verifications_completed = 0
        self._promotions_count = 0
        self._refutations_count = 0

    def should_run_verification(self) -> bool:
        proposed = self.hypothesis_store.get_proposed()
        if not proposed:
            return False
        return random.random() < self.config.verification_probability

    def get_next_verification(self) -> Optional[VerificationPlan]:
        """Get the next verification plan to execute."""
        if self.current_plan and not self.current_plan.is_complete():
            return self.current_plan

        plan = self.hypothesis_store.get_next_verification_plan()
        if plan:
            self.current_plan = plan
            return plan

        if not self.should_run_verification():
            self._episodes_since_verification += 1
            return None

        hypothesis = self._select_hypothesis()
        if not hypothesis:
            return None

        plan = self._design_experiment(hypothesis)
        if not plan:
            return None

        self.hypothesis_store.queue_for_verification(plan)
        self.current_plan = plan
        return plan

    def _select_hypothesis(self) -> Optional[Hypothesis]:
        candidates = self.hypothesis_store.get_proposed()
        if not candidates:
            return None

        if self.config.selection_strategy == "oldest":
            return min(candidates, key=lambda h: h.created_at)
        elif self.config.selection_strategy == "highest_confidence":
            return max(candidates, key=lambda h: h.confidence)
        else:  # UCB
            scores = []
            for h in candidates:
                n_tests = len(h.verification_history) + 1
                exploration = self.config.ucb_exploration * np.sqrt(
                    np.log(self._verifications_completed + 1) / n_tests
                )
                scores.append((h, h.confidence + exploration))
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[0][0]

    def _design_experiment(self, hypothesis: Hypothesis) -> Optional[VerificationPlan]:
        conditions = []

        if hypothesis.predictions:
            for pred in hypothesis.predictions:
                conditions.append({
                    "name": f"test_{hypothesis.hypothesis_type}",
                    "description": pred.get("condition", hypothesis.statement),
                    "expected_outcome": "success"
                    if hypothesis.hypothesis_type in [PrincipleType.PREFER, PrincipleType.SEQUENCE]
                    else "fail",
                    "action_constraint": None,
                    "completed": False,
                    "actual_outcome": None,
                })
        else:
            expected = "fail" if hypothesis.hypothesis_type == PrincipleType.AVOID else "success"
            conditions.append({
                "name": f"test_{hypothesis.hypothesis_type}",
                "description": hypothesis.statement,
                "expected_outcome": expected,
                "action_constraint": None,
                "completed": False,
            })

        if not conditions:
            return None

        return VerificationPlan(
            hypothesis_id=hypothesis.hid,
            conditions=conditions,
            episodes_remaining=min(len(conditions), self.config.max_episodes_per_hypothesis),
        )

    def record_result(self, success: bool, episode_id: str, notes: str = "") -> None:
        """Record the result of a verification episode."""
        if not self.current_plan:
            return
        self.current_plan.record_result("success" if success else "fail", episode_id, notes)
        if self.current_plan.is_complete():
            self._finalize_verification()

    def _finalize_verification(self) -> None:
        if not self.current_plan:
            return
        accuracy = self.current_plan.calculate_accuracy()
        hypothesis = self.hypothesis_store.get(self.current_plan.hypothesis_id)
        if hypothesis:
            hypothesis.add_verification(
                accuracy=accuracy,
                conditions=self.current_plan.conditions,
                episode_ids=[
                    c.get("episode_id", "") for c in self.current_plan.conditions if c.get("completed")
                ],
            )
            if accuracy >= self.config.promotion_confidence:
                hypothesis.status = HypothesisStatus.VERIFIED
            elif accuracy < self.config.refutation_confidence:
                hypothesis.status = HypothesisStatus.REFUTED
                self._refutations_count += 1
            else:
                hypothesis.status = HypothesisStatus.PROPOSED
            self.hypothesis_store.update(hypothesis)

        self.current_plan.completed_at = datetime.now().isoformat()
        self._verifications_completed += 1
        self._episodes_since_verification = 0
        self.current_plan = None

    def promote_verified(self) -> List[Principle]:
        """Promote verified hypotheses to principles."""
        promoted = []
        for hypothesis in self.hypothesis_store.get_ready_for_promotion():
            principle = self._hypothesis_to_principle(hypothesis)
            if principle:
                pid = self.principle_store.add(principle)
                self.hypothesis_store.mark_promoted(hypothesis.hid, pid)
                promoted.append(principle)
                self._promotions_count += 1
        return promoted

    def _hypothesis_to_principle(self, hypothesis: Hypothesis) -> Optional[Principle]:
        if not hypothesis.statement:
            return None
        content = hypothesis.statement
        if hypothesis.hypothesis_type == PrincipleType.AVOID:
            if not content.lower().startswith(("don't", "never", "avoid")):
                content = f"Avoid: {content}"
        elif hypothesis.hypothesis_type == PrincipleType.PREFER:
            if not content.lower().startswith(("prefer", "when", "for")):
                content = f"Prefer: {content}"
        elif hypothesis.hypothesis_type == PrincipleType.SEQUENCE:
            if not content.lower().startswith(("before", "after", "first")):
                content = f"Sequence: {content}"

        return Principle(
            content=content,
            formal_rule=hypothesis.formal_rule,
            evidence_for=hypothesis.source_experience_ids,
            importance_score=2.0 + hypothesis.confidence * 5.0,
            action_types=hypothesis.action_types,
            trigger_conditions=hypothesis.trigger_conditions,
            extraction_method="hypothesis_verification",
            source_task=f"hypothesis_{hypothesis.hid}",
        )

    def get_verification_context(self) -> Optional[Dict[str, Any]]:
        if not self.current_plan:
            return None
        condition = self.current_plan.get_current_condition()
        if not condition:
            return None
        hypothesis = self.hypothesis_store.get(self.current_plan.hypothesis_id)
        return {
            "action_constraint": condition.get("action_constraint"),
            "hypothesis_statement": hypothesis.statement if hypothesis else "",
            "hypothesis_type": str(hypothesis.hypothesis_type) if hypothesis else "",
            "expected_outcome": condition.get("expected_outcome"),
        }

    def is_in_verification_mode(self) -> bool:
        return self.current_plan is not None and not self.current_plan.is_complete()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "verifications_completed": self._verifications_completed,
            "promotions": self._promotions_count,
            "refutations": self._refutations_count,
        }


class ExperienceFolding:
    """
    Compress raw experiences once they are covered by established principles.

    Folded experiences are marked as "folded" and can be archived to save memory.
    """

    def __init__(self, memory: MemoryBank, principle_store: PrincipleStore):
        self.memory = memory
        self.principle_store = principle_store
        self._total_folded = 0

    def fold_covered_experiences(self, max_fold: int = 50) -> int:
        """Fold experiences that are now covered by established principles."""
        folded_count = 0
        for exp in self.memory.experiences:
            if exp.memory_status != "active":
                continue
            if folded_count >= max_fold:
                break

            # Check if any established principle covers this experience
            for principle in self.principle_store.principles:
                if not principle.is_established:
                    continue
                if exp.eid in principle.evidence_for:
                    exp.memory_status = "folded"
                    exp.folded_into_principle_id = principle.pid
                    folded_count += 1
                    break

        self._total_folded += folded_count
        return folded_count

    def get_stats(self) -> Dict[str, Any]:
        active = sum(1 for e in self.memory.experiences if e.memory_status == "active")
        folded = sum(1 for e in self.memory.experiences if e.memory_status == "folded")
        return {
            "total_folded": folded,
            "active_experiences": active,
            "folding_ratio": folded / len(self.memory) if len(self.memory) > 0 else 0,
        }
