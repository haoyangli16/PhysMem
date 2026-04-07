"""
Hypothesis System for the Scientific Learning Loop.

Hypotheses are proposed but unverified conjectures derived from experience patterns.
They go through: PROPOSED -> TESTING -> VERIFIED/REFUTED -> PROMOTED (to principle).
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from physmem.core.principle import PrincipleType


class HypothesisStatus(str, Enum):
    """Status of a hypothesis in the verification pipeline."""
    PROPOSED = "proposed"
    TESTING = "testing"
    VERIFIED = "verified"
    REFUTED = "refuted"
    PROMOTED = "promoted"

    def __str__(self) -> str:
        return self.value


class ExperienceStatus(str, Enum):
    """Status of an experience in the memory system."""
    ACTIVE = "active"
    CONSOLIDATED = "consolidated"
    FOLDED = "folded"
    ARCHIVED = "archived"

    def __str__(self) -> str:
        return self.value


@dataclass
class Hypothesis:
    """
    A proposed but unverified conjecture derived from experience patterns.

    Lifecycle: PROPOSED -> TESTING -> VERIFIED/REFUTED -> PROMOTED (if verified)
    """

    hid: str = field(default_factory=lambda: f"h_{str(uuid.uuid4())[:8]}")
    statement: str = ""
    hypothesis_type: PrincipleType = PrincipleType.GENERAL

    # Source
    source_experience_ids: List[str] = field(default_factory=list)
    source_cluster_id: Optional[str] = None

    # Testable predictions
    predictions: List[Dict[str, Any]] = field(default_factory=list)
    formal_rule: Optional[str] = None

    # Applicability
    action_types: List[str] = field(default_factory=list)
    trigger_conditions: List[str] = field(default_factory=list)
    shape_patterns: List[str] = field(default_factory=list)

    # Confidence and status
    confidence: float = 0.3
    status: HypothesisStatus = HypothesisStatus.PROPOSED

    # Attribution tracking
    supporting_episodes: List[str] = field(default_factory=list)
    contradicting_episodes: List[str] = field(default_factory=list)

    # Verification tracking
    verification_history: List[Dict[str, Any]] = field(default_factory=list)
    verification_episodes_planned: int = 3
    verification_episodes_completed: int = 0

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_modified: str = field(default_factory=lambda: datetime.now().isoformat())

    # Promotion link
    promoted_to_principle_id: Optional[str] = None

    def update_confidence(self, accuracy: float, weight: float = 0.7) -> None:
        """Update confidence score based on verification accuracy."""
        self.confidence = (1 - weight) * self.confidence + weight * accuracy
        self.last_modified = datetime.now().isoformat()

    def add_verification(
        self,
        accuracy: float,
        conditions: List[Dict[str, Any]],
        episode_ids: List[str],
        notes: str = "",
    ) -> None:
        """Record a verification attempt."""
        self.verification_history.append({
            "timestamp": datetime.now().isoformat(),
            "conditions_tested": conditions,
            "accuracy": accuracy,
            "episode_ids": episode_ids,
            "notes": notes,
        })
        self.verification_episodes_completed += len(episode_ids)
        self.update_confidence(accuracy)

    def is_ready_for_promotion(
        self, min_confidence: float = 0.8, min_verifications: int = 2
    ) -> bool:
        """Check if this hypothesis should be promoted to a principle."""
        return (
            self.confidence >= min_confidence
            and len(self.verification_history) >= min_verifications
            and self.status == HypothesisStatus.VERIFIED
        )

    def should_be_refuted(self, max_failures: int = 2, min_confidence: float = 0.3) -> bool:
        """Check if this hypothesis should be marked as refuted."""
        failed = sum(1 for v in self.verification_history if v.get("accuracy", 0) < 0.5)
        return failed >= max_failures or self.confidence < min_confidence

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["hypothesis_type"] = str(self.hypothesis_type)
        d["status"] = str(self.status)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Hypothesis":
        if "hypothesis_type" in d and isinstance(d["hypothesis_type"], str):
            try:
                d["hypothesis_type"] = PrincipleType(d["hypothesis_type"])
            except ValueError:
                d["hypothesis_type"] = PrincipleType.GENERAL
        if "status" in d and isinstance(d["status"], str):
            try:
                d["status"] = HypothesisStatus(d["status"])
            except ValueError:
                d["status"] = HypothesisStatus.PROPOSED
        return cls(**d)


@dataclass
class ExperienceCluster:
    """A group of similar raw experiences identified through consolidation."""

    cid: str = field(default_factory=lambda: f"c_{str(uuid.uuid4())[:8]}")
    experience_ids: List[str] = field(default_factory=list)
    common_pattern: str = ""
    distinguishing_features: List[str] = field(default_factory=list)
    outcome_distribution: Dict[str, int] = field(default_factory=lambda: {"success": 0, "fail": 0})
    shape_patterns: List[str] = field(default_factory=list)
    action_patterns: List[str] = field(default_factory=list)
    dependency_patterns: List[str] = field(default_factory=list)
    generated_hypotheses: List[str] = field(default_factory=list)
    cohesion: float = 0.0
    size: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        self.size = len(self.experience_ids)

    @property
    def success_rate(self) -> float:
        total = self.outcome_distribution.get("success", 0) + self.outcome_distribution.get("fail", 0)
        return self.outcome_distribution.get("success", 0) / total if total > 0 else 0.0

    @property
    def is_mostly_failures(self) -> bool:
        return self.outcome_distribution.get("fail", 0) > self.outcome_distribution.get("success", 0)

    @property
    def is_mostly_successes(self) -> bool:
        return self.outcome_distribution.get("success", 0) > self.outcome_distribution.get("fail", 0)

    @property
    def is_mixed(self) -> bool:
        return (
            self.outcome_distribution.get("success", 0) >= 2
            and self.outcome_distribution.get("fail", 0) >= 2
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperienceCluster":
        return cls(**d)


@dataclass
class VerificationCondition:
    """A single test condition in a verification plan."""
    name: str
    description: str
    expected_outcome: str
    action_constraint: Optional[str] = None
    completed: bool = False
    actual_outcome: Optional[str] = None
    episode_id: Optional[str] = None
    notes: str = ""

    def is_correct(self) -> bool:
        if not self.completed or self.actual_outcome is None:
            return False
        return self.actual_outcome == self.expected_outcome


@dataclass
class VerificationPlan:
    """Plan for testing a hypothesis through directed experiments."""

    plan_id: str = field(default_factory=lambda: f"vp_{str(uuid.uuid4())[:8]}")
    hypothesis_id: str = ""
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    episodes_remaining: int = 3
    current_condition_idx: int = 0
    accuracy: Optional[float] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None

    def get_current_condition(self) -> Optional[Dict[str, Any]]:
        if self.current_condition_idx < len(self.conditions):
            return self.conditions[self.current_condition_idx]
        return None

    def record_result(self, actual_outcome: str, episode_id: str, notes: str = "") -> None:
        if self.current_condition_idx < len(self.conditions):
            condition = self.conditions[self.current_condition_idx]
            condition["completed"] = True
            condition["actual_outcome"] = actual_outcome
            condition["episode_id"] = episode_id
            condition["notes"] = notes
            self.current_condition_idx += 1
            self.episodes_remaining -= 1

    def is_complete(self) -> bool:
        return self.episodes_remaining <= 0 or self.current_condition_idx >= len(self.conditions)

    def calculate_accuracy(self) -> float:
        if not self.conditions:
            return 0.0
        correct = 0
        total = 0
        for cond in self.conditions:
            if cond.get("completed", False):
                total += 1
                if cond.get("actual_outcome") == cond.get("expected_outcome"):
                    correct += 1
        self.accuracy = correct / total if total > 0 else 0.0
        return self.accuracy

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VerificationPlan":
        return cls(**d)


class HypothesisStore:
    """
    Storage and management for hypotheses.

    Supports adding, status transitions, retrieval, semantic deduplication, and persistence.
    """

    def __init__(self, name: str = "default", semantic_model=None):
        self.name = name
        self.hypotheses: List[Hypothesis] = []
        self._hid_to_idx: Dict[str, int] = {}
        self.verification_queue: List[VerificationPlan] = []
        self.active_plan: Optional[VerificationPlan] = None
        self.semantic_model = semantic_model
        self._hypothesis_embeddings: Dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.hypotheses)

    def _is_too_specific(self, statement: str) -> bool:
        """Reject hypotheses that mention exact counts (limits generalization)."""
        s = statement.lower()
        if re.search(r"\bexactly\s+\d+", s):
            return True
        if re.search(r"\b\d+\s+\w+\s+remain", s):
            return True
        if re.search(r"remain[s]?\s*[=:]\s*\d+", s):
            return True
        return False

    def add(self, hypothesis: Hypothesis) -> str:
        """Add a new hypothesis. Returns the hypothesis ID (empty if rejected)."""
        if self._is_too_specific(hypothesis.statement):
            return ""

        similar = self.find_similar(hypothesis.statement, threshold=0.72)
        if similar:
            existing = similar[0][0]
            existing.source_experience_ids.extend(hypothesis.source_experience_ids)
            existing.last_modified = datetime.now().isoformat()
            return existing.hid

        idx = len(self.hypotheses)
        self.hypotheses.append(hypothesis)
        self._hid_to_idx[hypothesis.hid] = idx
        return hypothesis.hid

    def get(self, hid: str) -> Optional[Hypothesis]:
        idx = self._hid_to_idx.get(hid)
        if idx is not None and idx < len(self.hypotheses):
            return self.hypotheses[idx]
        return None

    def update(self, hypothesis: Hypothesis) -> None:
        idx = self._hid_to_idx.get(hypothesis.hid)
        if idx is not None and idx < len(self.hypotheses):
            self.hypotheses[idx] = hypothesis

    def get_by_status(self, status: HypothesisStatus) -> List[Hypothesis]:
        return [h for h in self.hypotheses if h.status == status]

    def get_proposed(self) -> List[Hypothesis]:
        return self.get_by_status(HypothesisStatus.PROPOSED)

    def get_verified(self) -> List[Hypothesis]:
        return self.get_by_status(HypothesisStatus.VERIFIED)

    def get_ready_for_promotion(self) -> List[Hypothesis]:
        return [h for h in self.hypotheses if h.is_ready_for_promotion()]

    def find_similar(
        self, statement: str, threshold: float = 0.7
    ) -> List[Tuple[Hypothesis, float]]:
        """Find hypotheses with similar statements using embeddings or word overlap."""
        results = []

        if self.semantic_model is not None:
            try:
                query_emb = self.semantic_model.encode(statement, show_progress_bar=False)
                for h in self.hypotheses:
                    if h.hid not in self._hypothesis_embeddings:
                        self._hypothesis_embeddings[h.hid] = self.semantic_model.encode(
                            h.statement, show_progress_bar=False
                        )
                    score = self._cosine_similarity(query_emb, self._hypothesis_embeddings[h.hid])
                    if score >= threshold:
                        results.append((h, float(score)))
                results.sort(key=lambda x: x[1], reverse=True)
                return results
            except Exception:
                pass

        # Fallback: word overlap
        statement_words = set(statement.lower().split())
        for h in self.hypotheses:
            h_words = set(h.statement.lower().split())
            if statement_words and h_words:
                overlap = len(statement_words & h_words)
                score = overlap / max(len(statement_words), len(h_words))
                if score >= threshold:
                    results.append((h, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a).flatten()
        b = np.asarray(b).flatten()
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def get_applicable(
        self,
        action_type: Optional[str] = None,
        status: Optional[HypothesisStatus] = None,
    ) -> List[Hypothesis]:
        results = []
        for h in self.hypotheses:
            if status and h.status != status:
                continue
            if action_type and h.action_types and action_type not in h.action_types:
                continue
            results.append(h)
        return results

    def mark_testing(self, hid: str) -> None:
        h = self.get(hid)
        if h:
            h.status = HypothesisStatus.TESTING
            h.last_modified = datetime.now().isoformat()

    def mark_verified(self, hid: str) -> None:
        h = self.get(hid)
        if h:
            h.status = HypothesisStatus.VERIFIED
            h.last_modified = datetime.now().isoformat()

    def mark_refuted(self, hid: str) -> None:
        h = self.get(hid)
        if h:
            h.status = HypothesisStatus.REFUTED
            h.last_modified = datetime.now().isoformat()

    def mark_promoted(self, hid: str, principle_id: str) -> None:
        h = self.get(hid)
        if h:
            h.status = HypothesisStatus.PROMOTED
            h.promoted_to_principle_id = principle_id
            h.last_modified = datetime.now().isoformat()

    def queue_for_verification(self, plan: VerificationPlan) -> None:
        self.verification_queue.append(plan)
        self.mark_testing(plan.hypothesis_id)

    def get_next_verification_plan(self) -> Optional[VerificationPlan]:
        if self.active_plan and not self.active_plan.is_complete():
            return self.active_plan
        if self.verification_queue:
            self.active_plan = self.verification_queue.pop(0)
            return self.active_plan
        return None

    def get_stats(self) -> Dict[str, Any]:
        by_status = {}
        by_type = {}
        for h in self.hypotheses:
            by_status[str(h.status)] = by_status.get(str(h.status), 0) + 1
            by_type[str(h.hypothesis_type)] = by_type.get(str(h.hypothesis_type), 0) + 1
        avg_confidence = (
            sum(h.confidence for h in self.hypotheses) / len(self.hypotheses)
            if self.hypotheses
            else 0.0
        )
        return {
            "total": len(self.hypotheses),
            "by_status": by_status,
            "by_type": by_type,
            "average_confidence": avg_confidence,
            "verification_queue_size": len(self.verification_queue),
            "ready_for_promotion": len(self.get_ready_for_promotion()),
        }

    def format_for_prompt(
        self,
        hypotheses: Optional[List[Hypothesis]] = None,
        max_hypotheses: int = 5,
    ) -> str:
        """Format active hypotheses as text for inclusion in LLM prompts.

        By default returns only PROPOSED and VERIFIED hypotheses (the
        "working memory" tier in the paper). Refuted and promoted ones
        are excluded: refuted hypotheses are wrong, and promoted ones
        already live in the principle store.
        """
        if hypotheses is None:
            hypotheses = [
                h for h in self.hypotheses
                if h.status in (HypothesisStatus.PROPOSED, HypothesisStatus.VERIFIED)
            ]
            hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        hypotheses = hypotheses[:max_hypotheses]
        if not hypotheses:
            return "No active hypotheses."

        lines = ["Active Hypotheses (under verification, treat as tentative):"]
        for i, h in enumerate(hypotheses, 1):
            conf_str = "HIGH" if h.confidence > 0.7 else "MEDIUM" if h.confidence > 0.4 else "LOW"
            lines.append(f"{i}. [{conf_str}] {h.statement}")
            if h.action_types:
                lines.append(f"   Applies to: {', '.join(h.action_types)}")
        return "\n".join(lines)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "name": self.name,
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "verification_queue": [p.to_dict() for p in self.verification_queue],
            "active_plan": self.active_plan.to_dict() if self.active_plan else None,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "HypothesisStore":
        with open(path, "r") as f:
            data = json.load(f)
        store = cls(name=data.get("name", "default"))
        for h_dict in data.get("hypotheses", []):
            h = Hypothesis.from_dict(h_dict)
            store.hypotheses.append(h)
            store._hid_to_idx[h.hid] = len(store.hypotheses) - 1
        for p_dict in data.get("verification_queue", []):
            store.verification_queue.append(VerificationPlan.from_dict(p_dict))
        if data.get("active_plan"):
            store.active_plan = VerificationPlan.from_dict(data["active_plan"])
        return store
