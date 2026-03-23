"""
Principle Memory for Experience-to-Principle Learning.

A Principle captures:
- WHAT to do or avoid (the rule itself)
- WHEN it applies (action types, preconditions)
- WHY we learned it (evidence from experiences)
- HOW confident we are (importance score from voting)
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import json
import uuid
from pathlib import Path

import numpy as np


class PrincipleType(str, Enum):
    """Types of principles learned from experience."""
    AVOID = "avoid"        # "Don't do X when Y" (from failures)
    PREFER = "prefer"      # "Do X when Y" (from successes)
    SEQUENCE = "sequence"  # "Do X before Y" (from temporal patterns)
    COMPARE = "compare"    # "X is better than Y when Z" (from A/B testing)
    CONSTRAINT = "constraint"  # "Never do X" (hard constraints)
    GENERAL = "general"    # Generic principle

    def __str__(self) -> str:
        return self.value


@dataclass
class Principle:
    """An abstracted principle/rule learned from experience."""

    pid: str = field(default_factory=lambda: f"p_{str(uuid.uuid4())[:8]}")
    content: str = ""
    principle_type: PrincipleType = PrincipleType.GENERAL
    formal_rule: Optional[str] = None

    # Evidence tracking
    evidence_for: List[str] = field(default_factory=list)
    evidence_against: List[str] = field(default_factory=list)

    # Importance score (voting mechanism)
    importance_score: float = 2.0

    # Applicability
    action_types: List[str] = field(default_factory=list)
    trigger_conditions: List[str] = field(default_factory=list)
    domain: str = "general"
    addresses_fail_tags: List[str] = field(default_factory=list)

    # Source
    extraction_method: str = "reflection"
    source_task: str = ""

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_validated: str = field(default_factory=lambda: datetime.now().isoformat())
    last_modified: str = field(default_factory=lambda: datetime.now().isoformat())

    # Embedding for semantic retrieval
    embedding: Optional[np.ndarray] = None

    # Resonance & Decay
    reinforcement_count: int = 0
    prediction_errors: int = 0
    creation_episode: int = 0
    last_accessed_episode: int = 0
    last_used_at: str = field(default_factory=lambda: datetime.now().isoformat())
    retention_score: float = 1.0
    status: str = "active"
    refined_from_pid: Optional[str] = None
    refinement_context: Optional[str] = None

    @property
    def confidence(self) -> float:
        return min(1.0, self.importance_score / 10.0)

    @property
    def is_established(self) -> bool:
        return self.importance_score >= 5.0 and len(self.evidence_for) >= 3

    @property
    def needs_refinement(self) -> bool:
        total_uses = self.reinforcement_count + self.prediction_errors
        if total_uses < 5:
            return False
        return self.prediction_errors / total_uses > 0.2

    @property
    def is_stale(self) -> bool:
        return self.retention_score < 0.1 and self.reinforcement_count < 5

    def reinforce(self, episode: int) -> None:
        """Called when experience matches this principle's prediction (Resonance)."""
        self.reinforcement_count += 1
        self.last_accessed_episode = episode
        self.last_used_at = datetime.now().isoformat()
        self.retention_score = min(1.0, self.retention_score + 0.1)
        self.importance_score += 0.1

    def record_prediction_error(self, episode: int, context: Optional[str] = None) -> None:
        """Called when experience violates this principle's prediction (Surprise)."""
        self.prediction_errors += 1
        self.last_accessed_episode = episode
        self.last_modified = datetime.now().isoformat()
        if context:
            self.refinement_context = context

    def apply_decay(self, decay_factor: float = 0.99) -> None:
        """Apply temporal decay to retention score."""
        self.retention_score *= decay_factor

    def upvote(self, experience_id: str) -> None:
        self.importance_score += 1.0
        if experience_id not in self.evidence_for:
            self.evidence_for.append(experience_id)
        self.last_validated = datetime.now().isoformat()

    def downvote(self, experience_id: str, strength: float = 1.0) -> None:
        self.importance_score -= strength
        if experience_id not in self.evidence_against:
            self.evidence_against.append(experience_id)
        self.last_modified = datetime.now().isoformat()

    def edit(self, new_content: str, experience_id: str) -> None:
        self.content = new_content
        self.importance_score += 1.0
        if experience_id not in self.evidence_for:
            self.evidence_for.append(experience_id)
        self.last_modified = datetime.now().isoformat()

    def should_remove(self) -> bool:
        return self.importance_score <= 0

    def matches_context(
        self,
        action_type: Optional[str] = None,
        fail_tag: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> bool:
        if action_type and self.action_types and action_type not in self.action_types:
            return False
        if fail_tag and self.addresses_fail_tags and fail_tag not in self.addresses_fail_tags:
            return False
        if domain and self.domain != "general" and self.domain != domain:
            return False
        return True

    def to_dict(self) -> Dict:
        d = asdict(self)
        if self.embedding is not None:
            d["embedding"] = self.embedding.tolist()
        d["principle_type"] = str(self.principle_type)
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "Principle":
        if d.get("embedding") is not None:
            d["embedding"] = np.array(d["embedding"], dtype=np.float32)
        if "principle_type" in d and isinstance(d["principle_type"], str):
            try:
                d["principle_type"] = PrincipleType(d["principle_type"])
            except (ValueError, KeyError):
                d["principle_type"] = PrincipleType.GENERAL
        return cls(**d)


class PrincipleStore:
    """Storage and retrieval for learned principles."""

    def __init__(self, name: str = "default", embedder=None):
        self.name = name
        self.principles: List[Principle] = []
        self._pid_to_idx: Dict[str, int] = {}
        self.embedder = embedder
        self._action_type_index: Dict[str, List[int]] = {}
        self._fail_tag_index: Dict[str, List[int]] = {}

    def __len__(self) -> int:
        return len(self.principles)

    def add(self, principle: Principle) -> str:
        """Add a new principle or upvote existing if similar. Returns the principle ID."""
        similar = self.find_similar(principle.content, threshold=0.85)
        if similar:
            existing, _ = similar[0]
            existing.upvote(principle.evidence_for[0] if principle.evidence_for else "unknown")
            return existing.pid

        idx = len(self.principles)
        self.principles.append(principle)
        self._pid_to_idx[principle.pid] = idx

        for action_type in principle.action_types:
            if action_type not in self._action_type_index:
                self._action_type_index[action_type] = []
            self._action_type_index[action_type].append(idx)

        for fail_tag in principle.addresses_fail_tags:
            if fail_tag not in self._fail_tag_index:
                self._fail_tag_index[fail_tag] = []
            self._fail_tag_index[fail_tag].append(idx)

        if self.embedder and principle.embedding is None:
            try:
                principle.embedding = self.embedder(principle.content)
            except Exception:
                pass

        return principle.pid

    def get(self, pid: str) -> Optional[Principle]:
        idx = self._pid_to_idx.get(pid)
        if idx is not None:
            return self.principles[idx]
        return None

    def find_similar(
        self, content: str, threshold: float = 0.8, top_k: int = 5
    ) -> List[Tuple[Principle, float]]:
        if not self.principles:
            return []

        if self.embedder:
            query_emb = self.embedder(content)
            similarities = []
            for p in self.principles:
                if p.embedding is not None:
                    sim = self._cosine_similarity(query_emb, p.embedding)
                    if sim >= threshold:
                        similarities.append((p, sim))
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]

        # Fallback: word overlap
        results = []
        content_lower = content.lower()
        for p in self.principles:
            words_q = set(content_lower.split())
            words_p = set(p.content.lower().split())
            if words_q and words_p:
                score = len(words_q & words_p) / max(len(words_q), len(words_p))
                if score >= threshold:
                    results.append((p, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def retrieve(
        self,
        action_type: Optional[str] = None,
        fail_tag: Optional[str] = None,
        domain: Optional[str] = None,
        min_confidence: float = 0.3,
        top_k: int = 10,
    ) -> List[Principle]:
        """Retrieve applicable principles for the current context."""
        candidates = []
        for principle in self.principles:
            if principle.confidence < min_confidence:
                continue
            if not principle.matches_context(action_type, fail_tag, domain):
                continue
            candidates.append(principle)
        candidates.sort(key=lambda p: p.importance_score, reverse=True)
        return candidates[:top_k]

    def update_from_reflection(
        self, reflection_output: Dict[str, Any], experience_id: str
    ) -> str:
        """Update principles based on reflection output."""
        operation = reflection_output.get("operation", "ADD")

        if operation == "ADD":
            principle = Principle(
                content=reflection_output.get("content", ""),
                action_types=reflection_output.get("action_types", []),
                addresses_fail_tags=reflection_output.get("addresses_fail_tags", []),
                trigger_conditions=reflection_output.get("trigger_conditions", []),
                evidence_for=[experience_id],
                extraction_method="reflection",
            )
            return self.add(principle)

        pid = reflection_output.get("principle_id")
        principle = self.get(pid) if pid else None
        if not principle:
            return ""

        if operation == "UPVOTE":
            principle.upvote(experience_id)
        elif operation == "DOWNVOTE":
            principle.downvote(experience_id)
        elif operation == "EDIT":
            principle.edit(reflection_output.get("content", principle.content), experience_id)
        return pid

    def prune(self) -> int:
        """Remove principles with score <= 0. Returns count of removed."""
        to_remove = [i for i, p in enumerate(self.principles) if p.should_remove()]
        for i in reversed(to_remove):
            pid = self.principles[i].pid
            del self.principles[i]
            del self._pid_to_idx[pid]
        self._rebuild_indices()
        return len(to_remove)

    def _rebuild_indices(self) -> None:
        self._pid_to_idx = {}
        self._action_type_index = {}
        self._fail_tag_index = {}
        for i, p in enumerate(self.principles):
            self._pid_to_idx[p.pid] = i
            for at in p.action_types:
                self._action_type_index.setdefault(at, []).append(i)
            for ft in p.addresses_fail_tags:
                self._fail_tag_index.setdefault(ft, []).append(i)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a).flatten()
        b = np.asarray(b).flatten()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def get_stats(self) -> Dict:
        if not self.principles:
            return {"total": 0}
        return {
            "total": len(self.principles),
            "established": sum(1 for p in self.principles if p.is_established),
            "average_confidence": sum(p.confidence for p in self.principles) / len(self.principles),
            "by_action_type": {at: len(idx) for at, idx in self._action_type_index.items()},
        }

    def format_for_prompt(
        self, principles: Optional[List[Principle]] = None, max_principles: int = 5
    ) -> str:
        """Format principles as text for inclusion in LLM prompts."""
        if principles is None:
            principles = sorted(self.principles, key=lambda p: p.importance_score, reverse=True)
        principles = principles[:max_principles]
        if not principles:
            return "No relevant principles available."

        lines = ["Learned Principles (from past experience):"]
        for i, p in enumerate(principles, 1):
            conf_str = "HIGH" if p.confidence > 0.7 else "MEDIUM" if p.confidence > 0.4 else "LOW"
            lines.append(f"{i}. [{conf_str}] {p.content}")
            if p.action_types:
                lines.append(f"   Applies to: {', '.join(p.action_types)}")
        return "\n".join(lines)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"name": self.name, "principles": [p.to_dict() for p in self.principles]}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path, embedder=None) -> "PrincipleStore":
        with open(path, "r") as f:
            data = json.load(f)
        store = cls(name=data.get("name", "default"), embedder=embedder)
        for p_dict in data.get("principles", []):
            store.principles.append(Principle.from_dict(p_dict))
        store._rebuild_indices()
        return store
