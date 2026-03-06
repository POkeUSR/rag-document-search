from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from rag import BuildConfig, RAGService
from tfidf_baseline import TfidfBaseline, format_sources


def setup_logging() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def cmd_upload(args: argparse.Namespace) -> None:
    svc = RAGService(data_dir=args.index_dir)
    cfg = BuildConfig(
        chunk_size_tokens=args.chunk_size_tokens,
        overlap_ratio=args.overlap_ratio,
    )
    book_id = svc.build(args.path, config=cfg)
    loaded = svc.load(book_id)
    print(
        json.dumps(
            {
                "book_id": book_id,
                "source_file": str(Path(args.path)),
                "chunks": len(loaded.chunks),
                "embedding_dim": loaded.indexer.dim,
                "index_dir": str(loaded.artifacts.book_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_ask(args: argparse.Namespace) -> None:
    svc = RAGService(data_dir=args.index_dir)
    question = " ".join(args.question) if isinstance(args.question, list) else str(args.question)
    result = svc.answer(question=question, book_id=args.book_id, k=args.k)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=(os.getenv("LOG_LEVEL", "info")).lower(),
    )

def cmd_tfidf(args: argparse.Namespace) -> None:
    svc = RAGService(data_dir=args.index_dir)
    book_id = args.book_id or svc.store.get_latest()
    if not book_id:
        print(json.dumps({"error": "No book indexed yet. Upload/build first."}, ensure_ascii=False))
        return

    loaded = svc.load(book_id)
    baseline = TfidfBaseline()
    baseline.fit([c.text for c in loaded.chunks])
    question = " ".join(args.question)
    res = baseline.search(question, k=args.k)
    sources = format_sources(loaded.chunks, res.indices, res.scores)
    print(json.dumps({"book_id": book_id, "question": question, "sources": sources}, ensure_ascii=False, indent=2))


def cmd_compare(args: argparse.Namespace) -> None:
    svc = RAGService(data_dir=args.index_dir)
    book_id = args.book_id or svc.store.get_latest()
    if not book_id:
        print(json.dumps({"error": "No book indexed yet. Upload/build first."}, ensure_ascii=False))
        return

    loaded = svc.load(book_id)
    question = " ".join(args.question)

    rag_sources = svc.retrieve(book_id, question, k=args.k)
    baseline = TfidfBaseline()
    baseline.fit([c.text for c in loaded.chunks])
    tf = baseline.search(question, k=args.k)
    tf_sources = format_sources(loaded.chunks, tf.indices, tf.scores)

    print(
        json.dumps(
            {
                "book_id": book_id,
                "question": question,
                "rag": {"sources": rag_sources},
                "tfidf": {"sources": tf_sources},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Book RAG QA (PDF/TXT) with FAISS + ST")
    p.add_argument("--index-dir", default=os.getenv("RAG_INDEX_DIR") or "data/index")

    sub = p.add_subparsers(dest="cmd", required=True)

    up = sub.add_parser("upload", help="Build index for a book file")
    up.add_argument("path", help="Path to .pdf/.txt/.docx")
    up.add_argument("--chunk-size-tokens", type=int, default=500)
    up.add_argument("--overlap-ratio", type=float, default=0.15)
    up.set_defaults(func=cmd_upload)

    ask = sub.add_parser("ask", help="Ask a question (uses latest book by default)")
    ask.add_argument("question", nargs="+")
    ask.add_argument("--book-id", default=None)
    ask.add_argument("-k", type=int, default=4)
    ask.set_defaults(func=cmd_ask)

    sv = sub.add_parser("serve", help="Run REST API (FastAPI)")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--reload", action="store_true")
    sv.set_defaults(func=cmd_serve)

    tf = sub.add_parser("tfidf", help="TF-IDF baseline retrieval (top-k chunks)")
    tf.add_argument("question", nargs="+")
    tf.add_argument("--book-id", default=None)
    tf.add_argument("-k", type=int, default=5)
    tf.set_defaults(func=cmd_tfidf)

    cp = sub.add_parser("compare", help="Compare RAG retrieval vs TF-IDF baseline")
    cp.add_argument("question", nargs="+")
    cp.add_argument("--book-id", default=None)
    cp.add_argument("-k", type=int, default=5)
    cp.set_defaults(func=cmd_compare)

    return p


def enable_utf8_io() -> None:
    """
    Make CLI output readable on Windows terminals (avoid mojibake for Cyrillic).
    Safe no-op on platforms/Python builds that don't support reconfigure.
    """
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass


def main() -> None:
    load_dotenv()
    enable_utf8_io()
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

