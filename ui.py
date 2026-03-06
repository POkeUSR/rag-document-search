from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import streamlit as st

from rag import NO_INFO_ANSWER, BuildConfig, RAGService


APP_DATA_DIR = Path("data")
UPLOADS_DIR = APP_DATA_DIR / "uploads"
INDEX_DIR = APP_DATA_DIR / "index"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_resource
def get_service() -> RAGService:
    return RAGService(data_dir=INDEX_DIR)


def list_books() -> list[str]:
    if not INDEX_DIR.exists():
        return []
    ids: list[str] = []
    for p in INDEX_DIR.iterdir():
        if p.is_dir() and (p / "manifest.json").exists():
            ids.append(p.name)
    return sorted(ids)


def save_upload(uploaded_file) -> Path:
    name = Path(uploaded_file.name).name
    dst = UPLOADS_DIR / name
    dst.write_bytes(uploaded_file.getbuffer())
    return dst


def format_source(s: dict[str, Any]) -> str:
    bits = [f"chunk {s.get('chunk_id')}"]
    if s.get("page"):
        bits.append(f"page {s.get('page')}")
    if "rerank_score" in s:
        bits.append(f"rerank score {s.get('rerank_score'):.3f}")
    else:
        bits.append(f"score {s.get('score'):.3f}")
    return " | ".join(bits)


st.set_page_config(page_title="Book RAG Chat", layout="wide")
st.title("Book RAG Chat")

svc = get_service()

with st.sidebar:
    st.header("Книга")

    uploaded = st.file_uploader("Загрузить файл (PDF/TXT/DOCX)", type=["pdf", "txt", "docx"])

    st.subheader("Чанкинг")
    chunk_size = st.slider("Размер чанка (токены)", min_value=900, max_value=1300, value=1000, step=50)
    overlap = st.slider("Overlap (%)", min_value=15, max_value=20, value=18, step=1) / 100.0

    st.subheader("Поиск & Reranking")
    k = st.slider("Top-k retrieval", min_value=6, max_value=10, value=8, step=1)
    k_rerank = st.slider("Top-k после reranking", min_value=3, max_value=5, value=5, step=1)

    st.subheader("Ответ")
    threshold = st.slider(
        "Порог релевантности (similarity threshold)",
        min_value=0.0,
        max_value=0.6,
        value=0.25,
        step=0.01,
        help="Если лучший score ниже порога, система вернёт: 'В книге нет информации по данному вопросу.'",
    )

    if "book_id" not in st.session_state:
        st.session_state.book_id = svc.store.get_latest()

    books = list_books()
    selected = st.selectbox("Выбрать проиндексированную книгу", options=["(latest)"] + books)
    if selected == "(latest)":
        st.session_state.book_id = svc.store.get_latest()
    else:
        st.session_state.book_id = selected

    build_clicked = st.button("Собрать/пересобрать индекс из загруженного файла", type="primary", disabled=uploaded is None)

    if build_clicked and uploaded is not None:
        path = save_upload(uploaded)
        cfg = BuildConfig(
            chunk_size_tokens=int(chunk_size),
            overlap_ratio=float(overlap),
            similarity_threshold=float(threshold),
        )
        with st.status("Индексирую книгу (эмбеддинги + индекс)…", expanded=False):
            book_id = svc.build(path, config=cfg)
        st.session_state.book_id = book_id
        st.success(f"Готово. book_id={book_id}")
        st.caption(str(path))

    st.divider()
    st.subheader("Настройки LLM")
    st.caption("Управляется через переменные окружения: RAG_LLM, OPENAI_API_KEY, LOCAL_LLM_MODEL.")
    st.code("RAG_LLM=local\nLOCAL_LLM_MODEL=google/flan-t5-small\n\n# or\nRAG_LLM=openai\nOPENAI_API_KEY=...\nOPENAI_MODEL=gpt-4o-mini")


if "messages" not in st.session_state:
    st.session_state.messages = []


book_id = st.session_state.book_id

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Чат")
    if not book_id:
        st.info("Сначала загрузите книгу и соберите индекс (слева).")

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Введите вопрос по книге…", disabled=not bool(book_id))
    if prompt and book_id:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Думаю…"):
                result = svc.answer(question=prompt, book_id=book_id, k=int(k), k_rerank=int(k_rerank))
            st.markdown(result["answer"])
            if result.get("answer") == NO_INFO_ANSWER and result.get("sources"):
                try:
                    best = max(float(s.get("score", 0.0)) for s in result["sources"])
                except Exception:
                    best = None
                try:
                    index_threshold = float(svc.load(book_id).config.similarity_threshold)
                except Exception:
                    index_threshold = None

                if best is not None and index_threshold is not None and best < index_threshold:
                    extra = ""
                    if abs(index_threshold - float(threshold)) > 1e-6:
                        extra = f" (порог индекса={index_threshold:.2f}, в UI сейчас={threshold:.2f})"
                    st.caption(
                        f"Низкая уверенность: лучший score={best:.3f} ниже порога.{extra} "
                        "Попробуйте уменьшить порог, увеличить Top-k или переформулировать вопрос."
                    )
        st.session_state.messages.append({"role": "assistant", "content": result["answer"], "result": result})

with col2:
    st.subheader("Источники (top-k)")
    if st.session_state.messages:
        last = next((m for m in reversed(st.session_state.messages) if m.get("role") == "assistant" and m.get("result")), None)
        if last:
            res = last["result"]
            st.caption(f"book_id: {book_id}")
            for i, s in enumerate(res.get("sources", []), start=1):
                with st.expander(f"{i}. {format_source(s)}", expanded=(i == 1)):
                    st.text(s.get("source", ""))
                    st.write(s.get("text", ""))
        else:
            st.info("Задайте вопрос, чтобы увидеть источники.")
    else:
        st.info("Задайте вопрос, чтобы увидеть источники.")

    st.divider()
    st.subheader("Сервисная информация")
    if book_id:
        try:
            loaded = svc.load(book_id)
            st.json(
                {
                    "book_id": book_id,
                    "chunks": len(loaded.chunks),
                    "embedding_dim": loaded.indexer.dim,
                    "config": asdict(loaded.config),
                }
            )
        except Exception as e:
            st.error(str(e))
