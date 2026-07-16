"""Retrieval tools for the Strands agent."""

import json
from typing import Optional

from strands import tool

from src.index.vector_store import VectorStore

_STORE: VectorStore | None = None


def set_store(store: VectorStore) -> None:
    global _STORE
    _STORE = store


def get_store() -> VectorStore:
    if _STORE is None:
        raise RuntimeError("Vector store not initialized")
    return _STORE


@tool
def retrieve_chunks(
    query: str,
    company: Optional[str] = None,
    content_type: Optional[str] = None,
    doc_type: Optional[str] = None,
    k: int = 8,
) -> str:
    """Semantic search over earnings PDFs.

    Args:
        query: What to look for (include period cues in the query text when relevant).
        company: Optional filter: "Lam Research", "KLA", "Applied Materials", "ASML".
        content_type: Optional "prose" or "table" (prefer table for numeric figures).
        doc_type: Optional "earnings_release" or "earnings_call_transcript".
        k: Number of chunks to return.
    """
    hits = get_store().query(
        query=query,
        k=k,
        company=company,
        content_type=content_type,
        doc_type=doc_type,
    )
    results = []
    for h in hits:
        meta = h.get("metadata") or {}
        results.append(
            {
                "chunk_id": h["chunk_id"],
                "text": h["text"],
                "company": meta.get("company"),
                "doc_type": meta.get("doc_type"),
                "content_type": meta.get("content_type"),
                "source_file": meta.get("source_file"),
                "doc_date": meta.get("doc_date"),
                "page": meta.get("page"),
            }
        )
    return json.dumps({"results": results}, ensure_ascii=False)


@tool
def list_documents(company: Optional[str] = None) -> str:
    """List indexed source files (company, filename, doc_type, doc_date).

    Use this to see what is in the corpus before searching.
    If a company (e.g. TSMC) is absent, abstain.
    """
    rows = get_store().list_documents(company=company)
    return json.dumps({"documents": rows, "count": len(rows)}, ensure_ascii=False)


@tool
def get_full_chunk(chunk_id: str) -> str:
    """Fetch full text + metadata for a chunk_id from retrieve_chunks."""
    hit = get_store().get_chunk(chunk_id)
    if hit is None:
        return json.dumps({"error": f"chunk_id not found: {chunk_id}"})
    return json.dumps(hit, ensure_ascii=False)
