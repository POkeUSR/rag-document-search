from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import numpy as np


@dataclass
class SearchResult:
    indices: np.ndarray  # shape (k,)
    scores: np.ndarray  # shape (k,)


@runtime_checkable
class Indexer(Protocol):
    dim: int
    kind: Literal["faiss", "numpy"]

    def add(self, embeddings: np.ndarray) -> None: ...
    def search(self, query_embedding: np.ndarray, k: int = 4) -> SearchResult: ...
    def save(self, path: str | Path) -> None: ...


class NumpyIndexer:
    """
    Fallback when FAISS isn't available (common on Windows without Docker/Conda).
    Cosine similarity via dot product on normalized embeddings.
    """

    kind: Literal["numpy"] = "numpy"

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._emb: np.ndarray | None = None

    def add(self, embeddings: np.ndarray) -> None:
        emb = np.asarray(embeddings, dtype=np.float32)
        if emb.ndim != 2 or emb.shape[1] != self.dim:
            raise ValueError(f"Expected embeddings of shape (n, {self.dim})")
        self._emb = emb

    def search(self, query_embedding: np.ndarray, k: int = 4) -> SearchResult:
        if self._emb is None:
            return SearchResult(indices=np.array([], dtype=np.int64), scores=np.array([], dtype=np.float32))

        q = np.asarray(query_embedding, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if q.shape[1] != self.dim:
            raise ValueError(f"Query embedding dim mismatch: expected {self.dim}")

        scores = (self._emb @ q[0]).astype(np.float32)  # inner product
        k = min(int(k), int(scores.shape[0]))
        if k <= 0:
            return SearchResult(indices=np.array([], dtype=np.int64), scores=np.array([], dtype=np.float32))
        idx = np.argpartition(-scores, kth=k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return SearchResult(indices=idx.astype(np.int64), scores=scores[idx])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"type": "numpy", "dim": self.dim}), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, embeddings: np.ndarray) -> "NumpyIndexer":
        dim = int(embeddings.shape[1])
        obj = cls(dim=dim)
        obj.add(embeddings)
        return obj


class FaissIndexer:
    """
    Cosine similarity via IndexFlatIP + normalized embeddings.
    """

    kind: Literal["faiss"] = "faiss"

    def __init__(self, dim: int) -> None:
        import faiss

        self.dim = dim
        self._faiss = faiss
        self.index = faiss.IndexFlatIP(dim)

    def add(self, embeddings: np.ndarray) -> None:
        emb = np.asarray(embeddings, dtype=np.float32)
        if emb.ndim != 2 or emb.shape[1] != self.dim:
            raise ValueError(f"Expected embeddings of shape (n, {self.dim})")
        self.index.add(emb)

    def search(self, query_embedding: np.ndarray, k: int = 4) -> SearchResult:
        q = np.asarray(query_embedding, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if q.shape[1] != self.dim:
            raise ValueError(f"Query embedding dim mismatch: expected {self.dim}")

        scores, indices = self.index.search(q, k)
        return SearchResult(indices=indices[0], scores=scores[0])

    def save(self, path: str | Path) -> None:
        self._faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "FaissIndexer":
        import faiss

        idx = faiss.read_index(str(path))
        dim = idx.d
        obj = cls(dim=dim)
        obj.index = idx
        return obj


def create_indexer(dim: int) -> Indexer:
    """
    Prefer FAISS; fall back to NumPy if FAISS isn't importable.
    """
    try:
        return FaissIndexer(dim=dim)
    except Exception:
        return NumpyIndexer(dim=dim)
