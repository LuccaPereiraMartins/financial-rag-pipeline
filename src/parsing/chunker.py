"""Chunk construction with metadata tagging."""

import hashlib
import re
from dataclasses import dataclass

import tiktoken

from src.config import Config
from src.parsing.pdf_parser import ParsedDocument

_ENC = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


@dataclass
class Chunk:
    chunk_id: str
    text: str
    company: str
    doc_type: str
    content_type: str
    source_file: str
    doc_date: str
    page: int
    file_hash: str

    def metadata(self) -> dict:
        return {
            "company": self.company,
            "doc_type": self.doc_type,
            "content_type": self.content_type,
            "source_file": self.source_file,
            "doc_date": self.doc_date,
            "page": self.page,
            "file_hash": self.file_hash,
        }


def _make_chunk_id(source_file: str, page: int, content_type: str, text: str, index: int = 0) -> str:
    digest = hashlib.sha256(
        f"{source_file}|{page}|{content_type}|{index}|{text}".encode()
    ).hexdigest()
    return digest[:24]


def _split_prose(text: str, chunk_size: int, overlap: int) -> list[str]:
    parts = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if not parts:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if current:
            chunks.append("\n\n".join(current).strip())
            current, current_tokens = [], 0

    for part in parts:
        pt = count_tokens(part)
        if pt > chunk_size:
            flush()
            tokens = _ENC.encode(part)
            step = max(chunk_size - overlap, 1)
            for i in range(0, len(tokens), step):
                chunks.append(_ENC.decode(tokens[i : i + chunk_size]))
            continue
        if current and current_tokens + pt > chunk_size:
            flush()
        current.append(part)
        current_tokens += pt
    flush()
    return [c for c in chunks if c]


def _split_oversized(text: str, max_tokens: int = 6000) -> list[str]:
    if count_tokens(text) <= max_tokens:
        return [text]
    tokens = _ENC.encode(text)
    return [_ENC.decode(tokens[i : i + max_tokens]) for i in range(0, len(tokens), max_tokens)]


def document_to_chunks(doc: ParsedDocument) -> list[Chunk]:
    chunks: list[Chunk] = []
    source_file = doc.path.name
    # Prefix every chunk so filename/date are in the embedding space itself
    header = f"[source: {source_file} | {doc.company} | {doc.doc_type}"
    if doc.doc_date:
        header += f" | {doc.doc_date}"
    header += "]\n\n"

    for unit in doc.units:
        if unit.content_type == "table":
            texts = _split_oversized(unit.text)
        else:
            texts = []
            for t in _split_prose(unit.text, Config.CHUNK_SIZE_TOKENS, Config.CHUNK_OVERLAP_TOKENS):
                texts.extend(_split_oversized(t))

        for idx, text in enumerate(texts):
            if not text.strip():
                continue
            body = header + text
            chunks.append(
                Chunk(
                    chunk_id=_make_chunk_id(source_file, unit.page, unit.content_type, body, index=idx),
                    text=body,
                    company=doc.company,
                    doc_type=doc.doc_type,
                    content_type=unit.content_type,
                    source_file=source_file,
                    doc_date=doc.doc_date,
                    page=unit.page,
                    file_hash=doc.file_hash,
                )
            )
    return chunks
