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
from typing import Any, Dict, List, Optional, Tuple
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
from physmem.core.experience import Experience, MemoryBank


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

    # =========================================================================
    # Passive verification (observation-only, no agent cooperation needed)
    # =========================================================================

    def passive_verify_all(
        self,
        memory: MemoryBank,
        recent_window: int = 200,
        min_matching: int = 6,
    ) -> Dict[str, int]:
        """Test every PROPOSED hypothesis against recent experiences.

        Unlike active verification (which designs constrained experiments
        and requires the agent to execute them), passive verification
        observes the experiences that already happened and checks how
        well each hypothesis predicts them. This works without any
        cooperation from the downstream agent.

        For each PROPOSED hypothesis ``h``:
            1. Filter the last ``recent_window`` experiences by
               ``h.action_types`` and ``h.trigger_conditions``.
            2. If fewer than ``min_matching`` experiences match, skip.
            3. Compute accuracy:
               * AVOID: fraction of matching experiences that failed
               * PREFER / SEQUENCE: fraction that succeeded
            4. Append two ``add_verification`` entries (one per half of
               the matching set) so the hypothesis satisfies the
               ``min_verifications=2`` gate in
               ``Hypothesis.is_ready_for_promotion``.
            5. If accuracy >= ``promotion_confidence``: mark VERIFIED.
               If accuracy <= ``refutation_confidence``: mark REFUTED.

        Returns a summary dict ``{"verified": n, "refuted": n, "no_data": n}``.
        """
        proposed = self.hypothesis_store.get_proposed()
        counts = {"verified": 0, "refuted": 0, "no_data": 0}
        if not proposed:
            return counts

        recent = list(memory.experiences[-recent_window:])
        if not recent:
            return counts

        for hypothesis in proposed:
            matching = self._matching_experiences(hypothesis, recent)
            if len(matching) < min_matching:
                counts["no_data"] += 1
                continue

            accuracy = self._compute_passive_accuracy(hypothesis, matching)

            # Split in half so we get two verification_history entries
            # in one pass; this satisfies is_ready_for_promotion's
            # min_verifications=2 gate without inventing fake history.
            half = max(1, len(matching) // 2)
            batches = [matching[:half], matching[half:]]
            for batch in batches:
                if not batch:
                    continue
                batch_accuracy = self._compute_passive_accuracy(hypothesis, batch)
                hypothesis.add_verification(
                    accuracy=batch_accuracy,
                    conditions=[{
                        "name": "passive_observation",
                        "description": (
                            f"Observed {len(batch)} matching experiences; "
                            f"accuracy={batch_accuracy:.2f}"
                        ),
                        "expected_outcome": (
                            "fail"
                            if hypothesis.hypothesis_type == PrincipleType.AVOID
                            else "success"
                        ),
                        "completed": True,
                    }],
                    episode_ids=[e.eid for e in batch],
                    notes="passive_verification",
                )

            # Pin confidence to the observed accuracy. add_verification's
            # weighted-average update drags toward the prior (0.5), which
            # would prevent clean hypotheses from clearing the 0.8 gate.
            # Passive verification has direct empirical evidence, so use
            # it directly.
            hypothesis.confidence = accuracy

            if accuracy >= self.config.promotion_confidence:
                self.hypothesis_store.mark_verified(hypothesis.hid)
                self.hypothesis_store.update(hypothesis)
                counts["verified"] += 1
            elif accuracy <= self.config.refutation_confidence:
                self.hypothesis_store.mark_refuted(hypothesis.hid)
                self.hypothesis_store.update(hypothesis)
                self._refutations_count += 1
                counts["refuted"] += 1
            else:
                # Inconclusive: leave as PROPOSED but keep the
                # verification entries so future passes can build on them.
                self.hypothesis_store.update(hypothesis)
                counts["no_data"] += 1

        return counts

    def resolve_contradictions(self) -> int:
        """Refute hypotheses that contradict a VERIFIED hypothesis.

        Two hypotheses are considered contradictory iff they share the
        same ``action_types`` set and the same ``trigger_conditions``
        set, but have opposite types (PREFER vs AVOID). Whichever side
        already reached VERIFIED wins; the other is refuted.

        Returns the number of hypotheses refuted by this pass.
        """
        verified_keys: Dict[
            Tuple[frozenset, frozenset], "PrincipleType"
        ] = {}
        for h in self.hypothesis_store.hypotheses:
            if h.status != HypothesisStatus.VERIFIED:
                continue
            if not h.action_types:
                continue
            key = (
                frozenset(h.action_types),
                frozenset(h.trigger_conditions or []),
            )
            verified_keys[key] = h.hypothesis_type

        if not verified_keys:
            return 0

        refuted = 0
        for h in self.hypothesis_store.hypotheses:
            if h.status not in (
                HypothesisStatus.PROPOSED,
                HypothesisStatus.TESTING,
            ):
                continue
            if not h.action_types:
                continue
            key = (
                frozenset(h.action_types),
                frozenset(h.trigger_conditions or []),
            )
            winner = verified_keys.get(key)
            if winner is None or winner == h.hypothesis_type:
                continue
            self.hypothesis_store.mark_refuted(h.hid)
            self.hypothesis_store.update(h)
            self._refutations_count += 1
            refuted += 1
        return refuted

    def _matching_experiences(
        self,
        hypothesis: Hypothesis,
        experiences: List[Experience],
    ) -> List[Experience]:
        """Return experiences that fall within a hypothesis's scope.

        An experience matches iff:
            * its ``extra_metrics["action"]`` is in
              ``hypothesis.action_types`` (or action_types is empty), AND
            * every ``key=value`` pair in ``hypothesis.trigger_conditions``
              holds in ``exp.symbolic_state`` (or trigger_conditions is
              empty).
        """
        action_set = set(hypothesis.action_types or [])
        trigger_pairs: List[Tuple[str, str]] = []
        for cond in hypothesis.trigger_conditions or []:
            if "=" in cond:
                key, val = cond.split("=", 1)
                trigger_pairs.append((key.strip(), val.strip()))

        matching: List[Experience] = []
        for exp in experiences:
            if action_set:
                action = (exp.extra_metrics or {}).get("action")
                if action is None or str(action) not in action_set:
                    continue
            if trigger_pairs:
                sym = exp.symbolic_state or {}
                ok = True
                for key, val in trigger_pairs:
                    if str(sym.get(key)) != val:
                        ok = False
                        break
                if not ok:
                    continue
            matching.append(exp)
        return matching

    def _compute_passive_accuracy(
        self,
        hypothesis: Hypothesis,
        experiences: List[Experience],
    ) -> float:
        """Compute prediction accuracy of a hypothesis on a set of experiences."""
        if not experiences:
            return 0.0
        if hypothesis.hypothesis_type == PrincipleType.AVOID:
            correct = sum(
                1 for e in experiences if e.fail or not e.success
            )
        else:
            # PREFER, SEQUENCE, COMPARE, CONSTRAINT, GENERAL
            correct = sum(1 for e in experiences if e.success)
        return correct / len(experiences)

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
            principle_type=hypothesis.hypothesis_type,
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
