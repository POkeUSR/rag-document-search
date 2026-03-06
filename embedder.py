from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol

import numpy as np


@dataclass
class EmbedderConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    normalize: bool = True
    device: str = "cpu"


class _Backend(Protocol):
    def encode(self, texts: list[str], batch_size: int, normalize: bool) -> np.ndarray: ...
    @property
    def dim(self) -> int: ...


class _SentenceTransformersBackend:
    def __init__(self, model_name: str, device: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            model_name,
            device=device,
            model_kwargs={
                "low_cpu_mem_usage": False,
                "device_map": None,
            },
        )

    @property
    def dim(self) -> int:
        v = self._model.encode(["dim_probe"], normalize_embeddings=True, show_progress_bar=False)
        return int(v.shape[1])

    def encode(self, texts: list[str], batch_size: int, normalize: bool) -> np.ndarray:
        arr = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=normalize,
        )
        return np.asarray(arr, dtype=np.float32)


class _HFMeanPoolingBackend:
    """
    Fallback backend (no sentence-transformers) in case ST hits meta-tensor issues.
    Uses Transformers AutoModel with mean pooling.
    """

    def __init__(self, model_name: str, device: str) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self._device = device
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self._model = AutoModel.from_pretrained(
            model_name,
            low_cpu_mem_usage=False,
            device_map=None,
        )
        self._model.eval()
        if device:
            self._model.to(device)

    @property
    def dim(self) -> int:
        # hidden size is stable and cheap to access
        return int(getattr(self._model.config, "hidden_size"))

    def encode(self, texts: list[str], batch_size: int, normalize: bool) -> np.ndarray:
        torch = self._torch
        out_chunks: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                enc = {k: v.to(self._device) for k, v in enc.items()}
                m = self._model(**enc)
                last = m.last_hidden_state  # (b, t, h)
                mask = enc["attention_mask"].unsqueeze(-1).expand(last.size()).float()
                summed = (last * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                emb = summed / counts
                if normalize:
                    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                out_chunks.append(emb.cpu().numpy().astype(np.float32))
        return np.vstack(out_chunks) if out_chunks else np.zeros((0, self.dim), dtype=np.float32)


class Embedder:
    _BACKEND_CACHE: dict[tuple[str, str], _Backend] = {}

    def __init__(self, config: EmbedderConfig | None = None) -> None:
        self.config = config or EmbedderConfig()
        key = (self.config.model_name, self.config.device)
        if key not in self._BACKEND_CACHE:
            try:
                self._BACKEND_CACHE[key] = _SentenceTransformersBackend(
                    model_name=self.config.model_name,
                    device=self.config.device,
                )
            except NotImplementedError:
                # Meta-tensor error on some torch/transformers builds: fall back
                self._BACKEND_CACHE[key] = _HFMeanPoolingBackend(
                    model_name=self.config.model_name,
                    device=self.config.device,
                )
        self._backend = self._BACKEND_CACHE[key]

    @property
    def dim(self) -> int:
        return int(self._backend.dim)

    def embed_texts(self, texts: Iterable[str], batch_size: int = 32) -> np.ndarray:
        return self._backend.encode(list(texts), batch_size=batch_size, normalize=self.config.normalize)


def save_embeddings(path: str, embeddings: np.ndarray) -> None:
    np.save(path, embeddings)


def load_embeddings(path: str) -> np.ndarray:
    return np.load(path)
