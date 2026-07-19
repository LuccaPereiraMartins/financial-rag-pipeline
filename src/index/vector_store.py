"""Chroma persistent vector store."""

from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.config import Settings

from src.config import Config
from src.index.embeddings import Embedder
from src.parsing.chunker import Chunk


# NOTE: in practice, you would use a persistent vector database like S3 Vectors or Pinecone
# and ingestion would be separate to the agent & retrieval process.
# NOTE: Production alternative: upload docs to S3 and let a managed service (e.g. Amazon Bedrock
# Knowledge Bases) handle parsing/OCR/chunking/embedding — this local Chroma path is the
# case-study stand-in.

class VectorStore:
    def __init__(self ):
        self.index_dir = Path(Config.INDEX_DIR) # we shouldn't be changing the index so not in init args
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = Embedder() # nor for embedding process
        self._client = chromadb.PersistentClient(
            path=str(self.index_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self._client.get_or_create_collection(
            name=Config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}, # set retrieval algorithm and distance metric
        )

    def stored_file_hash(self, source_file: str) -> Optional[str]:
        """Return the file_hash stored for this source_file, or None if not ingested.

        """
        result = self.collection.get(
            where={"source_file": source_file},
            limit=1,
            include=["metadatas"],
        )
        metas = result.get("metadatas") or []
        if not metas or not metas[0]:
            return None
        return metas[0].get("file_hash")

    def delete_by_source_file(self, source_file: str) -> None:
        self.collection.delete(where={"source_file": source_file})

    def upsert_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [c.metadata() for c in chunks]
        embeddings = self.embedder.embed(documents)

        # batch upserts are more efficient
        for i in range(0, len(ids), 100):
            self.collection.upsert(
                ids=ids[i : i + 100],
                documents=documents[i : i + 100],
                metadatas=metadatas[i : i + 100],
                embeddings=embeddings[i : i + 100],
            )
        return len(chunks)

    def query(
        self,
        query: str,
        k: int = 8,
        company: str | None = None,
        doc_type: str | None = None,
    ) -> list[dict[str, Any]]:
        # Chroma metadata filter (`where`). Needed so retrieve_chunks can scope by
        # company/doc_type. Syntax is awkward when combining filters ($and).
        where: dict[str, Any] | None = None
        if company and doc_type:
            where = {"$and": [{"company": company}, {"doc_type": doc_type}]}
        elif company:
            where = {"company": company}
        elif doc_type:
            where = {"doc_type": doc_type}

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
