"""
FAISS index for fast nearest neighbor retrieval.

Falls back to numpy brute-force if FAISS is not installed.
"""

from typing import Tuple
from pathlib import Path

import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class FAISSIndex:
    """FAISS-based index for fast nearest neighbor search."""

    def __init__(self, dim: int, use_gpu: bool = False):
        self.dim = dim
        self.use_gpu = use_gpu
        self.index = None
        self.n_vectors = 0

        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatL2(dim)
            if use_gpu and faiss.get_num_gpus() > 0:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        else:
            self._vectors = None

    def add(self, vectors: np.ndarray):
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        if len(vectors.shape) == 1:
            vectors = vectors.reshape(1, -1)

        if FAISS_AVAILABLE:
            self.index.add(vectors)
        else:
            if self._vectors is None:
                self._vectors = vectors
            else:
                self._vectors = np.vstack([self._vectors, vectors])

        self.n_vectors += len(vectors)

    def search(self, query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        query = np.ascontiguousarray(query.astype(np.float32))
        if len(query.shape) == 1:
            query = query.reshape(1, -1)

        k = min(k, self.n_vectors)
        if k == 0:
            return np.array([[]]), np.array([[]])

        if FAISS_AVAILABLE:
            distances, indices = self.index.search(query, k)
        else:
            if self._vectors is None or len(self._vectors) == 0:
                return np.array([[]]), np.array([[]])
            diff = self._vectors[np.newaxis, :, :] - query[:, np.newaxis, :]
            dists = np.sum(diff ** 2, axis=2)
            indices = np.argsort(dists, axis=1)[:, :k]
            distances = np.take_along_axis(dists, indices, axis=1)

        return distances, indices

    def reset(self):
        if FAISS_AVAILABLE:
            self.index.reset()
        else:
            self._vectors = None
        self.n_vectors = 0

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if FAISS_AVAILABLE:
            if self.use_gpu:
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpu_index, str(path))
            else:
                faiss.write_index(self.index, str(path))
        else:
            np.save(path, self._vectors)

    @classmethod
    def load(cls, path: Path, use_gpu: bool = False) -> "FAISSIndex":
        path = Path(path)
        if FAISS_AVAILABLE and path.suffix != ".npy":
            index = faiss.read_index(str(path))
            obj = cls(dim=index.d, use_gpu=use_gpu)
            if use_gpu and faiss.get_num_gpus() > 0:
                res = faiss.StandardGpuResources()
                obj.index = faiss.index_cpu_to_gpu(res, 0, index)
            else:
                obj.index = index
            obj.n_vectors = index.ntotal
            return obj
        else:
            vectors = np.load(path)
            dim = vectors.shape[1] if len(vectors.shape) > 1 else 0
            obj = cls(dim=dim, use_gpu=False)
            obj._vectors = vectors
            obj.n_vectors = len(vectors)
            return obj


def build_index_from_vectors(vectors: np.ndarray, use_gpu: bool = False) -> FAISSIndex:
    """Build a FAISS index from a matrix of vectors."""
    if len(vectors) == 0:
        return FAISSIndex(dim=1, use_gpu=use_gpu)
    dim = vectors.shape[1]
    index = FAISSIndex(dim=dim, use_gpu=use_gpu)
    index.add(vectors)
    return index
