from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal


@dataclass(frozen=True)
class LoadedPage:
    text: str
    source: str
    page: int | None  # 1-based for PDFs, None for non-paged sources


HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"[ \t]+")
MANY_NEWLINES_RE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """
    Minimal, predictable cleanup:
    - remove HTML tags (if any)
    - normalize newlines to \\n
    - collapse excessive whitespace
    - trim
    """
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = HTML_TAG_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text)
    text = MANY_NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def _load_txt(path: Path) -> list[LoadedPage]:
    data = path.read_bytes()
    raw: str | None = None
    for enc in ("utf-8-sig", "utf-8", "cp1251", "latin-1"):
        try:
            raw = data.decode(enc)
            break
        except Exception:
            continue
    if raw is None:
        raw = data.decode("utf-8", errors="ignore")

    text = clean_text(raw)
    return [LoadedPage(text=text, source=str(path), page=None)]


def _load_pdf(path: Path) -> list[LoadedPage]:
    try:
        import fitz  # PyMuPDF
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PyMuPDF is required for PDF support. Install with: pip install pymupdf"
        ) from e

    pages: list[LoadedPage] = []
    with fitz.open(str(path)) as doc:
        for i, page in enumerate(doc, start=1):
            text = clean_text(page.get_text("text"))
            if text:
                pages.append(LoadedPage(text=text, source=str(path), page=i))
    return pages


def _load_docx(path: Path) -> list[LoadedPage]:
    try:
        import docx  # python-docx
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "python-docx is required for DOCX support. Install with: pip install python-docx"
        ) from e

    document = docx.Document(str(path))
    parts: list[str] = []
    for p in document.paragraphs:
        if p.text and p.text.strip():
            parts.append(p.text.strip())
    text = clean_text("\n".join(parts))
    return [LoadedPage(text=text, source=str(path), page=None)]


SupportedFormat = Literal[".txt", ".pdf", ".docx"]


def load_book(path: str | Path) -> list[LoadedPage]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    suffix = p.suffix.lower()
    if suffix == ".txt":
        return _load_txt(p)
    if suffix == ".pdf":
        return _load_pdf(p)
    if suffix == ".docx":
        return _load_docx(p)
    raise ValueError(f"Unsupported format: {suffix}. Supported: .txt, .pdf, .docx")


def iter_text(pages: Iterable[LoadedPage]) -> str:
    return "\n\n".join([p.text for p in pages if p.text])
