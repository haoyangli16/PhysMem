"""
Experience and MemoryBank for PhysMem.

Experience = atomic episodic record (one per action/step).
MemoryBank = collection of experiences with retrieval capability.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import json
import uuid
from pathlib import Path

import numpy as np


@dataclass
class Experience:
    """
    A single experience item (episodic record).

    This is the atomic unit stored in memory. It captures:
    - Context: the situation when a decision was made
    - Decision: what action/strategy was chosen
    - Outcome: what happened (success/fail, metadata)
    - Symbolic state: discrete task state for filtering
    - Surprise tracking: for resonance-based learning
    """

    # Unique identifier
    eid: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Context
    task: str = ""
    subtask: str = ""
    env_id: str = ""

    # Query representation (for retrieval)
    state_vec: Optional[np.ndarray] = None

    # Symbolic state for state-query based retrieval
    # This is a user-defined dict of discrete state features.
    # Examples: {"action_type": "grasp", "holding": True, "progress": 0.5}
    symbolic_state: Optional[Dict[str, Any]] = None

    # Decision
    strategy_id: str = "default"

    # Outcome
    success: bool = False
    fail: bool = False
    steps: int = 0
    reward: float = 0.0

    # Failure tags (for structured memory)
    fail_tag: Optional[str] = None

    # Evidence (optional)
    keyframe_embeddings: Optional[np.ndarray] = None
    extra_metrics: Dict[str, Any] = field(default_factory=dict)

    # Correction (for failures that were later fixed)
    correction: Optional[Dict[str, Any]] = None

    # Surprise-Driven Learning
    is_surprising: bool = False
    resonance_score: float = 0.0
    active_principle_ids: List[str] = field(default_factory=list)

    # Episode tracking for decay
    creation_episode: int = 0
    last_accessed_episode: int = 0

    # Folding status: "active", "folded", "archived"
    memory_status: str = "active"
    folded_into_principle_id: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        if self.state_vec is not None:
            d["state_vec"] = self.state_vec.tolist()
        if self.keyframe_embeddings is not None:
            d["keyframe_embeddings"] = self.keyframe_embeddings.tolist()
        if self.symbolic_state is not None:
            sym = dict(self.symbolic_state)
            for k, v in sym.items():
                if isinstance(v, (set, frozenset)):
                    sym[k] = list(v)
            d["symbolic_state"] = sym
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "Experience":
        """Create from dictionary."""
        if d.get("state_vec") is not None:
            d["state_vec"] = np.array(d["state_vec"], dtype=np.float32)
        if d.get("keyframe_embeddings") is not None:
            d["keyframe_embeddings"] = np.array(d["keyframe_embeddings"], dtype=np.float32)
        return cls(**d)

    def get_symbolic_filter_key(self) -> Optional[tuple]:
        """
        Get a hashable key for coarse symbolic filtering.

        Override this or provide a custom filter_key_fn to MemoryBank
        for domain-specific filtering.

        Default: uses (task, subtask, success) as the filter key.
        """
        if self.symbolic_state is None:
            return None

        # Default: use task-level grouping
        # Users should override this for domain-specific filtering
        parts = []
        for key in sorted(self.symbolic_state.keys()):
            val = self.symbolic_state[key]
            if isinstance(val, (str, int, float, bool)):
                parts.append((key, val))
        return tuple(parts) if parts else None

    def get_query_vector(self) -> np.ndarray:
        """Get the query vector for retrieval."""
        if self.state_vec is None:
            raise ValueError("No state_vec available for query")
        return self.state_vec.astype(np.float32)


class MemoryBank:
    """
    A collection of experiences with retrieval and update capability.

    Supports symbolic state indexing for filtered retrieval.
    """

    def __init__(self, name: str = "default"):
        self.name = name
        self.experiences: List[Experience] = []
        self._eid_to_idx: Dict[str, int] = {}
        self._symbolic_index: Dict[tuple, List[int]] = {}

    def __len__(self) -> int:
        return len(self.experiences)

    def add(self, exp: Experience) -> str:
        """Add an experience to memory. Returns the experience ID."""
        idx = len(self.experiences)
        self.experiences.append(exp)
        self._eid_to_idx[exp.eid] = idx

        filter_key = exp.get_symbolic_filter_key()
        if filter_key is not None:
            if filter_key not in self._symbolic_index:
                self._symbolic_index[filter_key] = []
            self._symbolic_index[filter_key].append(idx)

        return exp.eid

    def add_batch(self, exps: List[Experience]) -> List[str]:
        """Add multiple experiences."""
        return [self.add(exp) for exp in exps]

    def get(self, eid: str) -> Optional[Experience]:
        """Get experience by ID."""
        idx = self._eid_to_idx.get(eid)
        if idx is not None:
            return self.experiences[idx]
        return None

    def get_by_indices(self, indices: List[int]) -> List[Experience]:
        """Get experiences by indices."""
        return [self.experiences[i] for i in indices if 0 <= i < len(self.experiences)]

    def get_all_query_vectors(self) -> np.ndarray:
        """Get all query vectors as a matrix for indexing."""
        vectors = []
        for exp in self.experiences:
            if exp.state_vec is not None:
                vectors.append(exp.get_query_vector())
        if not vectors:
            return np.array([], dtype=np.float32)
        return np.stack(vectors, axis=0)

    def filter(
        self,
        task: Optional[str] = None,
        subtask: Optional[str] = None,
        success_only: bool = False,
        fail_only: bool = False,
    ) -> List[Experience]:
        """Filter experiences by criteria."""
        result = []
        for exp in self.experiences:
            if task is not None and exp.task != task:
                continue
            if subtask is not None and exp.subtask != subtask:
                continue
            if success_only and not exp.success:
                continue
            if fail_only and not exp.fail:
                continue
            result.append(exp)
        return result

    def get_by_symbolic_filter(
        self,
        filter_key: tuple,
        relaxed: bool = False,
    ) -> List[int]:
        """Get experience indices matching a symbolic filter key."""
        if filter_key in self._symbolic_index:
            return list(self._symbolic_index[filter_key])
        return []

    def get_all_with_symbolic_state(self) -> List[int]:
        """Get indices of all experiences that have symbolic_state set."""
        return [i for i, exp in enumerate(self.experiences) if exp.symbolic_state is not None]

    def rebuild_symbolic_index(self):
        """Rebuild the symbolic index from all experiences."""
        self._symbolic_index = {}
        for idx, exp in enumerate(self.experiences):
            filter_key = exp.get_symbolic_filter_key()
            if filter_key is not None:
                if filter_key not in self._symbolic_index:
                    self._symbolic_index[filter_key] = []
                self._symbolic_index[filter_key].append(idx)

    def get_stats(self) -> Dict:
        """Get statistics about the memory bank."""
        if not self.experiences:
            return {"total": 0}

        successes = sum(1 for e in self.experiences if e.success)
        fails = sum(1 for e in self.experiences if e.fail)

        by_task = {}
        by_subtask = {}
        for exp in self.experiences:
            by_task[exp.task] = by_task.get(exp.task, 0) + 1
            by_subtask[exp.subtask] = by_subtask.get(exp.subtask, 0) + 1

        with_symbolic = sum(1 for e in self.experiences if e.symbolic_state is not None)

        return {
            "total": len(self.experiences),
            "successes": successes,
            "fails": fails,
            "success_rate": successes / len(self.experiences) if self.experiences else 0,
            "by_task": by_task,
            "by_subtask": by_subtask,
            "with_symbolic_state": with_symbolic,
            "symbolic_buckets": len(self._symbolic_index),
        }

    def save(self, path: Path):
        """Save memory bank to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "name": self.name,
            "experiences": [exp.to_dict() for exp in self.experiences],
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: Path) -> "MemoryBank":
        """Load memory bank from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        bank = cls(name=data.get("name", "default"))
        for exp_dict in data.get("experiences", []):
            exp = Experience.from_dict(exp_dict)
            bank.add(exp)
        return bank

    def save_pt(self, path: Path):
        """Save memory bank as PyTorch file (faster for large memories)."""
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "name": self.name,
            "experiences": [exp.to_dict() for exp in self.experiences],
        }
        torch.save(data, path)

    @classmethod
    def load_pt(cls, path: Path) -> "MemoryBank":
        """Load memory bank from PyTorch file."""
        import torch

        data = torch.load(path, map_location="cpu", weights_only=False)
        bank = cls(name=data.get("name", "default"))
        for exp_dict in data.get("experiences", []):
            exp = Experience.from_dict(exp_dict)
            bank.add(exp)
        return bank
