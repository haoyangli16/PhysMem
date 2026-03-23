"""
Retriever for PhysMem.

Retrieves relevant experiences from memory based on query similarity.
Supports symbolic state filtering for state-query based retrieval.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np

from physmem.core.experience import Experience, MemoryBank
from physmem.core.index import FAISSIndex, build_index_from_vectors


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""

    experiences: List[Experience]
    distances: np.ndarray
    indices: np.ndarray

    def __len__(self):
        return len(self.experiences)

    @property
    def is_empty(self) -> bool:
        return len(self.experiences) == 0

    def get_success_rate(self) -> float:
        if not self.experiences:
            return 0.0
        return sum(1 for e in self.experiences if e.success) / len(self.experiences)

    def get_similarity_weighted_success_rate(self) -> float:
        """Success rate weighted by similarity (inverse distance)."""
        if not self.experiences or len(self.distances) == 0:
            return 0.0
        n = min(len(self.experiences), len(self.distances))
        if n == 0:
            return 0.0
        eps = 1e-6
        dists = self.distances[:n].flatten()
        similarities = 1.0 / (dists + eps)
        similarities = similarities / np.sum(similarities)
        return sum(
            sim * (1.0 if self.experiences[i].success else 0.0)
            for i, sim in enumerate(similarities)
        )


class Retriever:
    """
    Retriever for finding relevant experiences from memory.

    Supports k-NN retrieval based on state vector similarity,
    filtering by task/subtask, and random retrieval.
    """

    def __init__(self, memory: MemoryBank, use_gpu: bool = False, seed: int = 0):
        self.memory = memory
        self.use_gpu = use_gpu
        self.index: Optional[FAISSIndex] = None
        self._indexed_indices: List[int] = []
        self.rng = np.random.default_rng(seed)
        self._default_task: Optional[str] = None
        self._default_subtask: Optional[str] = None

    def build_index(
        self,
        task: Optional[str] = None,
        subtask: Optional[str] = None,
    ):
        """Build FAISS index from memory, optionally filtering by task/subtask."""
        self._default_task = task
        self._default_subtask = subtask

        vectors = []
        self._indexed_indices = []

        for i, exp in enumerate(self.memory.experiences):
            if task is not None and exp.task != task:
                continue
            if subtask is not None and exp.subtask != subtask:
                continue
            if exp.state_vec is None:
                continue
            vectors.append(exp.get_query_vector())
            self._indexed_indices.append(i)

        if vectors:
            vectors = np.stack(vectors, axis=0).astype(np.float32)
            self.index = build_index_from_vectors(vectors, use_gpu=self.use_gpu)
        else:
            self.index = FAISSIndex(dim=1, use_gpu=self.use_gpu)

    def rebuild_index(self):
        """Rebuild index with the same filters as the initial build."""
        self.build_index(task=self._default_task, subtask=self._default_subtask)

    def retrieve(self, query: np.ndarray, k: int = 5) -> RetrievalResult:
        """Retrieve k nearest experiences."""
        if self.index is None or self.index.n_vectors == 0:
            return RetrievalResult(
                experiences=[],
                distances=np.array([], dtype=np.float32),
                indices=np.array([], dtype=np.int64),
            )

        query = np.asarray(query, dtype=np.float32)
        if query.ndim == 1:
            query = query[None, :]

        distances, indices = self.index.search(query, k)

        valid_exps = []
        valid_dists = []
        valid_indices = []

        for j in range(indices.shape[1]):
            idx = int(indices[0, j])
            if 0 <= idx < len(self._indexed_indices):
                mem_idx = self._indexed_indices[idx]
                if 0 <= mem_idx < len(self.memory.experiences):
                    valid_exps.append(self.memory.experiences[mem_idx])
                    valid_dists.append(distances[0, j])
                    valid_indices.append(mem_idx)

        return RetrievalResult(
            experiences=valid_exps,
            distances=np.array(valid_dists, dtype=np.float32),
            indices=np.array(valid_indices, dtype=np.int64),
        )

    def retrieve_filtered(
        self,
        query: np.ndarray,
        symbolic_filter_key: Optional[tuple] = None,
        k: int = 5,
    ) -> RetrievalResult:
        """Retrieve with symbolic state filtering: filter first, then rank by similarity."""
        if symbolic_filter_key is not None:
            candidate_indices = self.memory.get_by_symbolic_filter(symbolic_filter_key)
            if not candidate_indices:
                return self.retrieve(query, k)

            # Build temporary index from candidates only
            vectors = []
            valid_indices = []
            for idx in candidate_indices:
                exp = self.memory.experiences[idx]
                if exp.state_vec is not None:
                    vectors.append(exp.get_query_vector())
                    valid_indices.append(idx)

            if not vectors:
                return self.retrieve(query, k)

            vectors = np.stack(vectors, axis=0).astype(np.float32)
            temp_index = build_index_from_vectors(vectors, use_gpu=self.use_gpu)

            query = np.asarray(query, dtype=np.float32)
            if query.ndim == 1:
                query = query[None, :]

            distances, indices = temp_index.search(query, min(k, len(vectors)))

            result_exps = []
            result_dists = []
            result_indices = []
            for j in range(indices.shape[1]):
                idx = int(indices[0, j])
                if 0 <= idx < len(valid_indices):
                    mem_idx = valid_indices[idx]
                    result_exps.append(self.memory.experiences[mem_idx])
                    result_dists.append(distances[0, j])
                    result_indices.append(mem_idx)

            return RetrievalResult(
                experiences=result_exps,
                distances=np.array(result_dists, dtype=np.float32),
                indices=np.array(result_indices, dtype=np.int64),
            )

        return self.retrieve(query, k)

    def retrieve_random(self, k: int = 5) -> RetrievalResult:
        """Random retrieval (baseline)."""
        if len(self.memory) == 0:
            return RetrievalResult(
                experiences=[], distances=np.array([]), indices=np.array([])
            )
        indices = self.rng.choice(len(self.memory), size=min(k, len(self.memory)), replace=False)
        exps = [self.memory.experiences[i] for i in indices]
        return RetrievalResult(
            experiences=exps,
            distances=np.zeros(len(exps), dtype=np.float32),
            indices=np.array(indices, dtype=np.int64),
        )
