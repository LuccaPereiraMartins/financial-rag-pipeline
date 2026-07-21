"""Pydantic schemas for agent answers."""

from pydantic import BaseModel, Field


class Citation(BaseModel):
    document: str = Field(description="Source PDF filename")
    page: int = Field(description="1-based page number")
    quote: str = Field(description="Short verbatim quote from the source")


class AgentAnswer(BaseModel):
    abstained: bool = Field(description="True if the corpus does not contain the answer")
    answer: str | None = Field(default=None, description="Grounded answer; null if abstained")
    citations: list[Citation] = Field(default_factory=list)
