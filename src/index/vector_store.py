"""Chroma persistent vector store."""

import hashlib
from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.config import Settings

from src.config import Config
from src.index.embeddings import Embedder
from src.parsing.chunker import Chunk


class VectorStore:
    def __init__(self, index_dir: str | None = None, embedder: Embedder | None = None):
        self.index_dir = Path(index_dir or Config.INDEX_DIR)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or Embedder()
        self._client = chromadb.PersistentClient(
            path=str(self.index_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self._client.get_or_create_collection(
            name=Config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def ingested_file_hashes(self) -> set[str]:
        if self.collection.count() == 0:
            return set()
        result = self.collection.get(include=["metadatas"])
        return {
            m["file_hash"]
            for m in (result.get("metadatas") or [])
            if m and m.get("file_hash")
        }

    def delete_by_file_hash(self, file_hash: str) -> None:
        self.collection.delete(where={"file_hash": file_hash})

    def upsert_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        seen: set[str] = set()
        unique: list[Chunk] = []
        for c in chunks:
            if c.chunk_id in seen:
                c.chunk_id = hashlib.sha256(
                    f"{c.chunk_id}|{len(unique)}|{c.text[:80]}".encode()
                ).hexdigest()[:24]
            seen.add(c.chunk_id)
            unique.append(c)

        ids = [c.chunk_id for c in unique]
        documents = [c.text for c in unique]
        metadatas = [c.metadata() for c in unique]
        embeddings = self.embedder.embed(documents)

        for i in range(0, len(ids), 100):
            self.collection.upsert(
                ids=ids[i : i + 100],
                documents=documents[i : i + 100],
                metadatas=metadatas[i : i + 100],
                embeddings=embeddings[i : i + 100],
            )
        return len(unique)

    def query(
        self,
        query: str,
        k: int = 8,
        company: str | None = None,
        doc_type: str | None = None,
    ) -> list[dict[str, Any]]:
        where = _build_where(company, doc_type)
        kwargs: dict[str, Any] = {
            "query_embeddings": [self.embedder.embed_one(query)],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        result = self.collection.query(**kwargs)
        hits = []
        for i, cid in enumerate((result.get("ids") or [[]])[0]):
            hits.append(
                {
                    "chunk_id": cid,
                    "text": (result.get("documents") or [[]])[0][i],
                    "metadata": (result.get("metadatas") or [[]])[0][i],
                    "distance": (result.get("distances") or [[]])[0][i],
                }
            )
        return hits

    def get_chunk(self, chunk_id: str) -> Optional[dict[str, Any]]:
        result = self.collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        if not result.get("ids"):
            return None
        return {
            "chunk_id": result["ids"][0],
            "text": (result.get("documents") or [""])[0],
            "metadata": (result.get("metadatas") or [{}])[0],
        }


def _build_where(
    company: str | None,
    doc_type: str | None,
) -> dict | None:
    clauses = []
    if company:
        clauses.append({"company": company})
    if doc_type:
        clauses.append({"doc_type": doc_type})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}
