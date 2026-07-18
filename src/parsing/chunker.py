"""Fixed-size word chunking with light overlap."""

import hashlib
from dataclasses import dataclass

from src.config import Config
from src.parsing.pdf_parser import ParsedDocument


@dataclass
class Chunk:
    chunk_id: str
    text: str
    company: str
    doc_type: str
    source_file: str
    doc_date: str
    page: int
    file_hash: str

    def metadata(self) -> dict:
        return {
            "company": self.company,
            "doc_type": self.doc_type,
            "source_file": self.source_file,
            "doc_date": self.doc_date,
            "page": self.page,
            "file_hash": self.file_hash,
        }


def _make_chunk_id(source_file: str, page: int, text: str, index: int = 0) -> str:
    digest = hashlib.sha256(
        f"{source_file}|{page}|{index}|{text}".encode()
    ).hexdigest()
    return digest[:24]


def _split_words(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= size:
        return [text.strip()]
    step = max(size - overlap, 1)
    out = []
    for i in range(0, len(words), step):
        piece = " ".join(words[i : i + size]).strip()
        if piece:
            out.append(piece)
        if i + size >= len(words):
            break
    return out


def document_to_chunks(doc: ParsedDocument) -> list[Chunk]:
    chunks: list[Chunk] = []
    source_file = doc.path.name
    # Prefix every chunk so filename/date are in the embedding space itself
    header = f"[source: {source_file} | {doc.company} | {doc.doc_type}"
    if doc.doc_date:
        header += f" | {doc.doc_date}"
    header += "]\n\n"

    size = Config.CHUNK_SIZE_WORDS
    overlap = Config.CHUNK_OVERLAP_WORDS

    for page in doc.pages:
        for idx, text in enumerate(_split_words(page.text, size, overlap)):
            body = header + text
            chunks.append(
                Chunk(
                    chunk_id=_make_chunk_id(source_file, page.page, body, index=idx),
                    text=body,
                    company=doc.company,
                    doc_type=doc.doc_type,
                    source_file=source_file,
                    doc_date=doc.doc_date,
                    page=page.page,
                    file_hash=doc.file_hash,
                )
            )
    return chunks
