"""Retrieval tools for the Strands agent."""

import json

from strands import tool

from src.config import Config
from src.index.vector_store import VectorStore

# Lazy singleton: tools are registered as bare functions, so they can't close over
# a store created in answer_question. get_store() once per tool call is cheap.
_STORE: VectorStore | None = None


def get_store() -> VectorStore:
    global _STORE
    if _STORE is None:
        _STORE = VectorStore()
    if _STORE.collection.count() == 0:
        raise ValueError('Index is empty.')
    return _STORE


@tool
def retrieve_chunks(
    query: str,
    company: str | None = None,
    doc_type: str | None = None,
    k: int = Config.RETRIEVAL_TOP_K,
) -> str:
    """Semantic search over earnings PDFs (prose and tables).

    Args:
        query: What to look for (include period cues in the query text when relevant).
        company: Optional filter: "Lam Research", "KLA", "Applied Materials", "ASML".
        doc_type: Optional "earnings_release" or "earnings_call_transcript".
        k: Number of chunks to return.
    """
    # NOTE: retrieval metadata such as company, doc_type, would make search much more efficient
    # when corpus scales
    hits = get_store().query(
        query=query,
        k=k,
        company=company,
        doc_type=doc_type,
    )
    results = []
    # present chunks back in useful way for agent to use
    for h in hits:
        meta = h.get("metadata") or {}
        results.append(
            {
                "chunk_id": h["chunk_id"],
                "text": h["text"],
                "company": meta.get("company"),
                "doc_type": meta.get("doc_type"),
                "source_file": meta.get("source_file"),
                "doc_date": meta.get("doc_date"),
                "page": meta.get("page"),
                "kind": meta.get("kind"),
            }
        )
    return json.dumps({"results": results}, ensure_ascii=False)


@tool
def get_full_chunk(chunk_id: str) -> str:
    """Fetch full text + metadata for a chunk_id from retrieve_chunks."""
    hit = get_store().get_chunk(chunk_id)
    if hit is None:
        return json.dumps({"error": f"chunk_id not found: {chunk_id}"})
    return json.dumps(hit, ensure_ascii=False)
