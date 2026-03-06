# Book QA (RAG) по загруженной книге

Система Question‑Answering по содержимому книги: загрузка файла → разбиение на чанки → эмбеддинги → векторный поиск → генерация ответа **строго по найденному контексту**.

Поддерживаются форматы: **PDF, TXT** (и дополнительно DOCX).

## Архитектура (модули)

- `loader.py`: загрузка PDF/TXT/DOCX, извлечение и очистка текста (в т.ч. удаление HTML‑тегов, нормализация переносов строк)
- `chunker.py`: разбиение на чанки **500–1000 токенов** (по умолчанию 500) с overlap **10–20%** (по умолчанию 15%) + метаданные (chunk_id, page, source)
- `embedder.py`: эмбеддинги через **SentenceTransformers**, сохранение в `embeddings.npy`
- `indexer.py`: векторный индекс **FAISS** (если доступен) с сохранением/загрузкой; на Windows без FAISS — прозрачный fallback на NumPy‑поиск
- `rag.py`: retrieval top‑k (3–5) + сборка промпта и генерация ответа через **OpenAI API** или локальную open‑source LLM
- `api.py`: REST API (`/upload`, `/ask`)
- `main.py`: запуск через CLI (`upload`, `ask`, `serve`)

## Быстрый старт (Python)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 1) Запуск REST API

```bash
python main.py serve --host 127.0.0.1 --port 8000
```

### 2) Загрузка книги

```bash
curl -X POST "http://127.0.0.1:8000/upload" -F "file=@book.docx"
```

Можно загружать `.pdf` или `.txt` аналогично.

### 3) Вопрос

```bash
curl -X POST "http://127.0.0.1:8000/ask" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"О чем эта книга?\",\"k\":4}"
```

Ответ API:

```json
{
  "question": "...",
  "answer": "...",
  "sources": [
    {"chunk_id": 0, "page": 1, "source": "...", "score": 0.53, "text": "..."}
  ]
}
```

## Запуск без API (CLI)

Построить индекс:

```bash
python main.py upload book.docx
```

Задать вопрос (используется последний загруженный `book_id`):

```bash
python main.py ask "Какие ключевые идеи?"
```

## Web UI (Streamlit)

Запуск веб‑интерфейса чата:

```bash
streamlit run ui.py
```

Дальше откройте страницу, загрузите `book.pdf` / `book.txt` / `book.docx`, нажмите сборку индекса и задавайте вопросы. Справа отображаются **источники** (чанки) с `chunk_id/page/score`.

## LLM режимы (RAG генерация)

Выбор режима:

- `RAG_LLM=openai` — использовать OpenAI (нужен `OPENAI_API_KEY`)
- `RAG_LLM=local` — использовать локальную open‑source модель (по умолчанию `google/flan-t5-small`)

Пример `.env`:

```bash
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini
RAG_LLM=openai
```

Для локальной модели:

```bash
RAG_LLM=local
LOCAL_LLM_MODEL=google/flan-t5-small
```

Если система не находит ответа в retrieved контексте, возвращается строка:

**`В книге нет информации по данному вопросу.`**

## Замечание про FAISS на Windows

`faiss-cpu` обычно проще ставится на Linux. В этом проекте:

- на Linux/в Docker будет использован **FAISS**
- на Windows без FAISS автоматически включится fallback‑поиск на NumPy (функциональность сохраняется)

## Технологии

- Python 3.10+
- FastAPI + Uvicorn
- SentenceTransformers (embeddings)
- FAISS (vector index) + fallback
- OpenAI API или open‑source LLM (Transformers)
- python-dotenv, logging
