from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from chunker import Chunk, TokenChunker
from embedder import Embedder, EmbedderConfig, load_embeddings, save_embeddings
from indexer import FaissIndexer, Indexer, NumpyIndexer, create_indexer
from loader import load_book


NO_INFO_ANSWER = "В книге нет информации по данному вопросу."


@dataclass(frozen=True)
class BookArtifacts:
    book_id: str
    book_dir: Path
    index_path: Path
    embeddings_path: Path
    chunks_path: Path
    manifest_path: Path


@dataclass
class BuildConfig:
    chunk_size_tokens: int = 1000
    overlap_ratio: float = 0.18
    top_k_default: int = 8
    top_k_retrieve: int = 10
    top_k_rerank: int = 5
    similarity_threshold: float = 0.25
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    rerank_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class BookStore:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        (self.root_dir / "_latest.json").touch(exist_ok=True)

    def new_book_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def artifacts(self, book_id: str) -> BookArtifacts:
        d = self.root_dir / book_id
        return BookArtifacts(
            book_id=book_id,
            book_dir=d,
            index_path=d / "index.faiss",
            embeddings_path=d / "embeddings.npy",
            chunks_path=d / "chunks.json",
            manifest_path=d / "manifest.json",
        )

    def set_latest(self, book_id: str) -> None:
        p = self.root_dir / "_latest.json"
        p.write_text(json.dumps({"book_id": book_id}, ensure_ascii=False), encoding="utf-8")

    def get_latest(self) -> str | None:
        p = self.root_dir / "_latest.json"
        try:
            data = json.loads(p.read_text(encoding="utf-8") or "{}")
            return data.get("book_id")
        except Exception:
            return None


@dataclass
class LoadedBook:
    config: BuildConfig
    artifacts: BookArtifacts
    chunks: list[Chunk]
    embeddings: np.ndarray
    indexer: Indexer
    embedder: Embedder
    tfidf: Any | None = None  # lazy baseline cache
    reranker: Any | None = None  # lazy cross-encoder cache


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _llm_mode() -> Literal["openai", "local"]:
    mode = (os.getenv("RAG_LLM") or "").strip().lower()
    if mode in {"openai", "local"}:
        return mode  # type: ignore[return-value]
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "local"


class RAGService:
    def __init__(self, data_dir: str | Path = "data/index") -> None:
        self.store = BookStore(data_dir)
        self._cache: dict[str, LoadedBook] = {}
        self._local_pipeline = None
        self._author_re = re.compile(
            r"\b([A-Za-zА-Яа-яЁё]\.)\s*([A-Za-zА-Яа-яЁё]\.)\s*([A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё-]{2,})\b"
        )
        self._chapter_q_re = re.compile(r"\b(глава|раздел|подраздел|тема)\b", re.I)

    def _norm(self, s: str) -> str:
        s = (s or "").strip().lower()
        s = s.replace("ё", "е")
        s = re.sub(r"\s+", " ", s)
        return s

    def _is_author_question(self, q: str) -> bool:
        s = (q or "").strip().lower()
        return ("кто автор" in s) or ("автор" in s and "книг" in s) or ("кто написал" in s)

    def _try_answer_author(self, loaded: LoadedBook) -> tuple[str | None, list[dict[str, Any]]]:
        # Usually author/title is in the very beginning of the book.
        for c in loaded.chunks[:20]:
            m = self._author_re.search(c.text)
            if not m:
                continue
            initials1, initials2, surname = m.group(1), m.group(2), m.group(3)
            author = f"{initials1.upper()} {initials2.upper()} {surname[:1].upper()}{surname[1:]}"
            src = {
                "chunk_id": c.chunk_id,
                "page": c.page,
                "source": c.source,
                "score": 1.0,
                "text": c.text,
            }
            return f"Автор: {author}.", [src]
        return None, []

    def _is_chapter_existence_question(self, q: str) -> bool:
        s = self._norm(q)
        if not self._chapter_q_re.search(s):
            return False
        return ("есть" in s) or ("в книге" in s) or ("присутств" in s) or ("имеетс" in s)

    def _extract_chapter_phrase(self, q: str) -> str | None:
        s = (q or "").strip()
        m = re.search(r"(?:глава|раздел|подраздел|тема)\s+(.+)$", s, flags=re.I)
        cand = m.group(1) if m else s
        cand = cand.strip().strip(" ?!.:;\"'“”«»()[]{}")
        cand = re.sub(r"^(эта|этот|это|про)\s+", "", cand, flags=re.I).strip()
        return cand if len(cand) >= 6 else None

    def _find_phrase_hits(self, loaded: LoadedBook, phrase: str, max_hits: int = 5) -> list[dict[str, Any]]:
        target = self._norm(phrase)
        hits: list[dict[str, Any]] = []
        if not target:
            return hits
        for c in loaded.chunks:
            if target in self._norm(c.text):
                hits.append(
                    {
                        "chunk_id": c.chunk_id,
                        "page": c.page,
                        "source": c.source,
                        "score": 1.0,
                        "text": c.text,
                    }
                )
                if len(hits) >= max_hits:
                    break
        return hits

    def _ensure_tfidf(self, loaded: LoadedBook) -> Any:
        if loaded.tfidf is not None:
            return loaded.tfidf
        from tfidf_baseline import TfidfBaseline

        base = TfidfBaseline()
        base.fit([c.text for c in loaded.chunks])
        loaded.tfidf = base
        return base

    def _ensure_reranker(self, loaded: LoadedBook) -> Any:
        """Lazy load cross-encoder for reranking."""
        if loaded.reranker is not None:
            return loaded.reranker
        try:
            from sentence_transformers import CrossEncoder

            reranker = CrossEncoder(loaded.config.rerank_model_name)
            loaded.reranker = reranker
            return reranker
        except Exception as e:
            import logging

            logging.warning(f"Failed to load reranker: {e}. Reranking disabled.")
            return None

    def rerank(self, question: str, sources: list[dict[str, Any]], loaded: LoadedBook) -> list[dict[str, Any]]:
        """Rerank retrieved sources using cross-encoder and return top-k best."""
        if not sources:
            return sources

        reranker = self._ensure_reranker(loaded)
        if reranker is None:
            # Fallback: return as-is if reranker not available
            return sources[: loaded.config.top_k_rerank]

        try:
            # Prepare pairs for reranking
            pairs = [[question, s["text"]] for s in sources]
            scores = reranker.predict(pairs)

            # Add rerank scores to sources
            for s, score in zip(sources, scores):
                s["rerank_score"] = float(score)

            # Sort by rerank score
            reranked = sorted(sources, key=lambda x: x["rerank_score"], reverse=True)
            return reranked[: loaded.config.top_k_rerank]
        except Exception as e:
            import logging

            logging.warning(f"Reranking failed: {e}. Returning original top-k.")
            return sources[: loaded.config.top_k_rerank]

    def build(self, book_path: str | Path, config: BuildConfig | None = None) -> str:
        cfg = config or BuildConfig()
        book_id = self.store.new_book_id()
        art = self.store.artifacts(book_id)
        art.book_dir.mkdir(parents=True, exist_ok=True)

        pages = load_book(book_path)
        chunker = TokenChunker(
            tokenizer_name=cfg.tokenizer_name,
            chunk_size_tokens=cfg.chunk_size_tokens,
            overlap_ratio=cfg.overlap_ratio,
        )
        chunks = chunker.chunk_pages(pages)

        embedder = Embedder(EmbedderConfig(model_name=cfg.embed_model_name, normalize=True))
        embeddings = embedder.embed_texts([c.text for c in chunks], batch_size=32)

        indexer = create_indexer(dim=int(embeddings.shape[1]))
        indexer.add(embeddings)
        index_type = getattr(indexer, "kind", "unknown")

        # Persist
        save_embeddings(str(art.embeddings_path), embeddings)
        indexer.save(str(art.index_path))
        art.chunks_path.write_text(
            json.dumps([c.to_dict() for c in chunks], ensure_ascii=False),
            encoding="utf-8",
        )
        art.manifest_path.write_text(
            json.dumps(
                {
                    "created_at": _now_iso(),
                    "source_file": str(Path(book_path)),
                    "config": asdict(cfg),
                    "counts": {"pages": len(pages), "chunks": len(chunks)},
                    "embedding_dim": int(embeddings.shape[1]),
                    "index_type": index_type,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        self.store.set_latest(book_id)
        self._cache.pop(book_id, None)
        return book_id

    def load(self, book_id: str) -> LoadedBook:
        if book_id in self._cache:
            return self._cache[book_id]

        art = self.store.artifacts(book_id)
        if not art.book_dir.exists():
            raise FileNotFoundError(f"Unknown book_id: {book_id}")

        manifest = json.loads(art.manifest_path.read_text(encoding="utf-8"))
        cfg = BuildConfig(**manifest.get("config", {}))

        chunks_raw = json.loads(art.chunks_path.read_text(encoding="utf-8"))
        chunks = [Chunk(**c) for c in chunks_raw]
        embeddings = load_embeddings(str(art.embeddings_path)).astype(np.float32)
        index_type = (manifest.get("index_type") or "faiss").strip().lower()
        if index_type == "faiss":
            try:
                indexer = FaissIndexer.load(str(art.index_path))
            except Exception:
                indexer = NumpyIndexer.load(str(art.index_path), embeddings=embeddings)
        else:
            indexer = NumpyIndexer.load(str(art.index_path), embeddings=embeddings)
        embedder = Embedder(EmbedderConfig(model_name=cfg.embed_model_name, normalize=True))

        loaded = LoadedBook(
            config=cfg,
            artifacts=art,
            chunks=chunks,
            embeddings=embeddings,
            indexer=indexer,
            embedder=embedder,
        )
        self._cache[book_id] = loaded
        return loaded

    def retrieve(self, book_id: str, question: str, k: int | None = None, k_rerank: int | None = None, with_reranking: bool = True) -> list[dict[str, Any]]:
        loaded = self.load(book_id)
        use_k = int(k or loaded.config.top_k_retrieve)
        q_emb = loaded.embedder.embed_texts([question], batch_size=1)[0]
        res = loaded.indexer.search(q_emb, k=use_k)

        out: list[dict[str, Any]] = []
        for idx, score in zip(res.indices.tolist(), res.scores.tolist()):
            if idx < 0 or idx >= len(loaded.chunks):
                continue
            c = loaded.chunks[idx]
            out.append(
                {
                    "chunk_id": c.chunk_id,
                    "page": c.page,
                    "source": c.source,
                    "score": float(score),
                    "text": c.text,
                }
            )

        # Rerank if enabled
        if with_reranking and out:
            # Temporarily set top_k_rerank for reranking
            original_k = loaded.config.top_k_rerank
            if k_rerank is not None:
                loaded.config.top_k_rerank = int(k_rerank)
            out = self.rerank(question, out, loaded)
            loaded.config.top_k_rerank = original_k

        return out

    def answer(self, question: str, book_id: str | None = None, k: int | None = None, k_rerank: int | None = None, with_reranking: bool = True) -> dict[str, Any]:
        use_book_id = book_id or self.store.get_latest()
        if not use_book_id:
            return {"question": question, "answer": NO_INFO_ANSWER, "sources": []}

        loaded = self.load(use_book_id)
        top_k = int(k or loaded.config.top_k_retrieve)

        # Deterministic author extraction (no LLM, strictly from book text).
        if self._is_author_question(question):
            author_ans, author_sources = self._try_answer_author(loaded)
            if author_ans:
                return {"question": question, "answer": author_ans, "sources": author_sources}

        # Deterministic "does chapter/section exist?" lookup by exact phrase mention in the book.
        if self._is_chapter_existence_question(question):
            phrase = self._extract_chapter_phrase(question)
            if phrase:
                hits = self._find_phrase_hits(loaded, phrase, max_hits=5)
                if hits:
                    return {
                        "question": question,
                        "answer": f"В тексте книги встречается «{phrase}» (см. источники).",
                        "sources": hits,
                    }

        # Retrieve WITHOUT reranking first to check similarity threshold
        sources_raw = self.retrieve(use_book_id, question, k=top_k, k_rerank=k_rerank, with_reranking=False)
        
        if not sources_raw:
            return {"question": question, "answer": NO_INFO_ANSWER, "sources": []}

        # Check threshold on raw cosine scores (before reranking)
        best = max(s["score"] for s in sources_raw)
        if best < loaded.config.similarity_threshold:
            # Try TF-IDF baseline as a fallback retriever (often helps for metadata-like queries).
            try:
                from tfidf_baseline import format_sources

                tfidf = self._ensure_tfidf(loaded)
                tf = tfidf.search(question, k=top_k)
                tf_sources = format_sources(loaded.chunks, tf.indices, tf.scores)
                if tf_sources:
                    sources_raw = tf_sources
            except Exception:
                pass

            best2 = max((s.get("score", 0.0) for s in sources_raw), default=0.0)
            if best2 < loaded.config.similarity_threshold:
                return {"question": question, "answer": NO_INFO_ANSWER, "sources": sources_raw}

        # NOW apply reranking if enabled
        if with_reranking:
            sources = self.rerank(question, sources_raw, loaded)
        else:
            sources = sources_raw

        context_blocks: list[str] = []
        for s in sources:
            tag = f"chunk={s['chunk_id']}"
            if s.get("page"):
                tag += f", page={s['page']}"
            # Clean text from tokenizer artifacts
            clean_text = s['text'].replace('##', '')
            context_blocks.append(f"[{tag}]\n{clean_text}")

        context = "\n\n".join(context_blocks)
        prompt = (
            "Ты отвечаешь на вопрос строго по предоставленному контексту из книги.\n"
            f"Если в контексте нет ответа, ответь ровно: {NO_INFO_ANSWER}\n\n"
            f"КОНТЕКСТ:\n{context}\n\n"
            f"ВОПРОС: {question}\n\n"
            "ОТВЕТ:"
        )

        mode = _llm_mode()
        try:
            if mode == "openai":
                ans = self._answer_openai(prompt)
            else:
                ans = self._answer_local(prompt)
        except Exception:
            # Safe fallback: return an extractive answer from retrieved context
            # (still strictly inside book context).
            ans = sources[0]["text"][:800].strip() if sources else NO_INFO_ANSWER

        ans = (ans or "").strip()
        if not ans:
            ans = NO_INFO_ANSWER

        return {"question": question, "answer": ans, "sources": sources}

    def _answer_openai(self, prompt: str) -> str:
        try:
            from openai import OpenAI
        except Exception as e:  # pragma: no cover
            raise RuntimeError("OpenAI client not installed. pip install openai") from e

        client = OpenAI(organization=os.getenv("OPENAI_ORG_ID"))
        model = os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=350,
            messages=[
                {"role": "system", "content": "Отвечай только по контексту. Не добавляй факты от себя."},
                {"role": "user", "content": prompt},
            ],
        )
        return (resp.choices[0].message.content or "").strip()

    def _answer_local(self, prompt: str) -> str:
        model_name = os.getenv("LOCAL_LLM_MODEL") or "google/flan-t5-small"
        if self._local_pipeline is None:
            try:
                from transformers import pipeline
            except Exception as e:  # pragma: no cover
                raise RuntimeError("transformers not installed. pip install transformers") from e

            self._local_pipeline = pipeline(
                task="text2text-generation",
                model=model_name,
                device=-1,
            )

        out = self._local_pipeline(prompt, max_new_tokens=200, do_sample=False)
        if not out:
            return ""
        return (out[0].get("generated_text") or "").strip()
