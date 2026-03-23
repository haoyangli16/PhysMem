"""
Write-back policy for PhysMem.

Handles when and how to write new experiences back to memory.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Set

from physmem.core.experience import Experience, MemoryBank
from physmem.core.retriever import Retriever


@dataclass
class WritebackConfig:
    """Configuration for write-back policy."""
    on_failure: bool = True
    on_success: bool = False
    on_surprise: bool = False
    deduplicate: bool = True
    novelty_threshold: float = 0.1
    max_memory_size: int = 10000
    recency_bias: bool = True


class WritebackPolicy:
    """Policy for writing new experiences back to memory (test-time learning)."""

    def __init__(
        self,
        memory: MemoryBank,
        retriever: Retriever,
        config: Optional[WritebackConfig] = None,
    ):
        self.memory = memory
        self.retriever = retriever
        self.config = config or WritebackConfig()
        self._written_eids: Set[str] = set()

    def should_write(
        self,
        experience: Experience,
        predicted_success: Optional[bool] = None,
    ) -> bool:
        if self.config.on_failure and experience.fail:
            return True
        if self.config.on_success and experience.success:
            return True
        if self.config.on_surprise and predicted_success is not None:
            if experience.success != predicted_success:
                return True
        return False

    def write(self, experience: Experience, force: bool = False) -> bool:
        if len(self.memory) >= self.config.max_memory_size:
            if self.config.recency_bias:
                self._prune_oldest()
            else:
                return False

        if self.config.deduplicate and not force:
            if not self._is_novel(experience):
                return False

        eid = self.memory.add(experience)
        self._written_eids.add(eid)
        self.retriever.rebuild_index()
        return True

    def _is_novel(self, experience: Experience) -> bool:
        if experience.state_vec is None:
            return True
        if self.retriever.index is None or self.retriever.index.n_vectors == 0:
            return True
        result = self.retriever.retrieve(experience.get_query_vector(), k=1)
        if result.is_empty:
            return True
        return result.distances.min() > self.config.novelty_threshold

    def _prune_oldest(self, n: int = 1):
        """Remove oldest experiences to make room."""
        pass

    def get_stats(self) -> Dict:
        return {
            "total_written": len(self._written_eids),
            "memory_size": len(self.memory),
        }
