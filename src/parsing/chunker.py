"""Chunk construction: LangChain recursive split for prose; tables kept whole.

# LangChain RecursiveCharacterTextSplitter prefers paragraph/line breaks over mid-sentence cuts.
# Tables stay as one chunk each so numeric rows aren't sliced apart
"""

import hashlib
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

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
    kind: str  # prose | table — useful in retrieval results, not a filter

    def metadata(self) -> dict:
        return {
            "company": self.company,
            "doc_type": self.doc_type,
            "source_file": self.source_file,
            "doc_date": self.doc_date,
            "page": self.page,
            "file_hash": self.file_hash,
            "kind": self.kind,
        }


def _make_chunk_id(source_file: str, page: int, kind: str, text: str, index: int) -> str:
    digest = hashlib.sha256(
        f"{source_file}|{page}|{kind}|{index}|{text}".encode()
    ).hexdigest()
    return digest[:24]


_PROSE_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=Config.CHUNK_SIZE,
    chunk_overlap=Config.CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""], # set the order in which to split the text
)

# Rare huge tables: split on row boundaries only
_TABLE_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=Config.CHUNK_SIZE * 2,
    chunk_overlap=0,
    separators=["\n", " "],
)


def document_to_chunks(doc: ParsedDocument) -> list[Chunk]:
    chunks: list[Chunk] = []
    source_file = doc.path.name
    # Prefix every chunk so filename/date are in the embedding space itself
    header = f"[source: {source_file} | {doc.company} | {doc.doc_type}"
    if doc.doc_date:
        header += f" | {doc.doc_date}"
    header += "]\n\n"

    for block in doc.blocks:
        if block.kind == "table":
            pieces = (
                [block.text]
                if len(block.text) <= Config.CHUNK_SIZE * 2
                else _TABLE_SPLITTER.split_text(block.text)
            )
        else:
            pieces = _PROSE_SPLITTER.split_text(block.text)

        for idx, text in enumerate(pieces):
            if not text.strip():
                continue
            body = header + text
            chunks.append(
                Chunk(
                    chunk_id=_make_chunk_id(source_file, block.page, block.kind, body, idx),
                    text=body,
                    company=doc.company,
                    doc_type=doc.doc_type,
                    source_file=source_file,
                    doc_date=doc.doc_date,
                    page=block.page,
                    file_hash=doc.file_hash,
                    kind=block.kind,
                )
            )
    return chunks
