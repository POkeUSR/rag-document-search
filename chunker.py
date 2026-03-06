from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class Chunk:
    chunk_id: int
    text: str
    source: str
    page: int | None
    start_token: int
    end_token: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TokenChunker:
    """
    Token-based chunking to satisfy "500–1000 tokens" requirement.
    Uses a HF tokenizer (same family as SentenceTransformers default).
    """

    def __init__(
        self,
        tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size_tokens: int = 500,
        overlap_ratio: float = 0.15,
    ) -> None:
        if chunk_size_tokens < 10:
            raise ValueError("chunk_size_tokens is too small")
        if not (0.0 <= overlap_ratio < 1.0):
            raise ValueError("overlap_ratio must be in [0, 1)")

        self.tokenizer_name = tokenizer_name
        self.chunk_size_tokens = chunk_size_tokens
        self.overlap_ratio = overlap_ratio

        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        # We use the tokenizer only for splitting, not for feeding the encoder model directly.
        # Avoid noisy warnings when encoding long texts for chunking.
        self._tokenizer.model_max_length = 10**9

    def _encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)

    def _decode(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=True).strip()

    def chunk_pages(self, pages: Iterable[Any]) -> list[Chunk]:
        """
        pages: iterable of objects with fields: text, source, page
        (compatible with loader.LoadedPage)
        """
        chunks: list[Chunk] = []
        chunk_id = 0

        overlap = int(self.chunk_size_tokens * self.overlap_ratio)
        step = max(1, self.chunk_size_tokens - overlap)

        for page_obj in pages:
            text = getattr(page_obj, "text", "") or ""
            source = getattr(page_obj, "source", "unknown")
            page = getattr(page_obj, "page", None)
            if not text.strip():
                continue

            token_ids = self._encode(text)
            if not token_ids:
                continue

            start = 0
            n = len(token_ids)
            while start < n:
                end = min(n, start + self.chunk_size_tokens)
                window = token_ids[start:end]
                chunk_text = self._decode(window)
                if chunk_text:
                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            text=chunk_text,
                            source=str(source),
                            page=page if isinstance(page, int) else None,
                            start_token=start,
                            end_token=end,
                        )
                    )
                    chunk_id += 1
                if end >= n:
                    break
                start += step

        return chunks
