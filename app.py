"""Earnings corpus API: REST (Swagger) + FastMCP tools."""

from fastapi import FastAPI, HTTPException
from fastmcp import FastMCP
from pydantic import BaseModel, Field

from src.agent.schemas import AgentAnswer
from src.agent.strands_agent import answer_question
from src.agent.tools import get_store
from src.config import Config

# --- shared helpers (REST + MCP) ---


def _search(
    query: str,
    company: str | None = None,
    doc_type: str | None = None,
    k: int = Config.RETRIEVAL_TOP_K,
) -> list[dict]:
    hits = get_store().query(query=query, k=k, company=company, doc_type=doc_type)
    results = []
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
                "distance": h.get("distance"),
            }
        )
    return results


def _get_chunk(chunk_id: str) -> dict:
    hit = get_store().get_chunk(chunk_id)
    if hit is None:
        return {"error": f"chunk_id not found: {chunk_id}"}
    return hit


def _ask(question: str) -> dict:
    result: AgentAnswer = answer_question(question)
    return {
        "abstained": result.abstained,
        "answer": result.answer,
        "citations": [
            {"document": c.document, "page": c.page, "quote": c.quote}
            for c in result.citations
        ],
    }


# --- MCP tools for LLM / MCP clients ---

mcp = FastMCP("Earnings Corpus")


@mcp.tool
def retrieve_chunks(
    query: str,
    company: str | None = None,
    doc_type: str | None = None,
    k: int = Config.RETRIEVAL_TOP_K,
) -> list[dict]:
    """Semantic search over indexed financial PDFs (prose and tables)."""
    return _search(query, company=company, doc_type=doc_type, k=k)


@mcp.tool
def get_full_chunk(chunk_id: str) -> dict:
    """Fetch full text + metadata for a chunk_id from retrieve_chunks."""
    return _get_chunk(chunk_id)


@mcp.tool
def ask(question: str) -> dict:
    """Grounded Q&A over the indexed corpus (same behaviour as POST /ask)."""
    return _ask(question)


mcp_app = mcp.http_app(path="/")

app = FastAPI(
    title="Earnings Corpus API",
    description="Ask grounded questions or search the vector index. MCP tools at /mcp.",
    lifespan=mcp_app.lifespan,
)
app.mount("/mcp", mcp_app)


# --- REST schemas ---


class AskRequest(BaseModel):
    question: str = Field(description="Question over the indexed financial corpus")


class CitationOut(BaseModel):
    document: str
    page: int
    quote: str


class AskResponse(BaseModel):
    abstained: bool
    answer: str | None = None
    citations: list[CitationOut] = Field(default_factory=list)


class SearchRequest(BaseModel):
    query: str
    company: str | None = None
    doc_type: str | None = None
    k: int = Config.RETRIEVAL_TOP_K


class SearchResponse(BaseModel):
    results: list[dict]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_http(body: AskRequest) -> AskResponse:
    try:
        payload = _ask(body.question)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return AskResponse(
        abstained=payload["abstained"],
        answer=payload["answer"],
        citations=[CitationOut(**c) for c in payload["citations"]],
    )


@app.post("/search", response_model=SearchResponse)
def search(body: SearchRequest) -> SearchResponse:
    try:
        results = _search(
            body.query,
            company=body.company,
            doc_type=body.doc_type,
            k=body.k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return SearchResponse(results=results)


# entrypoint for local dev
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
