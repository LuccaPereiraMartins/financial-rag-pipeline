"""Strands agent: retrieve + think, then structured answer."""

from strands import Agent
from strands.models.openai import OpenAIModel
from strands_tools import think

from src.agent.schemas import AgentAnswer
from src.agent.tools import get_full_chunk, list_documents, retrieve_chunks, set_store
from src.config import Config
from src.index.vector_store import VectorStore

SYSTEM_PROMPT = """You answer questions ONLY from the semiconductor earnings corpus
(Lam Research, KLA, Applied Materials, ASML).

Rules:
- Never use parametric knowledge. Every claim must come from retrieved chunks.
- Call list_documents and/or retrieve_chunks before answering.
- Prefer content_type="table" for numeric figures from earnings releases.
- Filenames and doc_date are in chunk text/metadata — use them to resolve periods
  (companies have different fiscal calendars; read the source, don't invent mappings).
- Use the think tool when comparing periods or aggregating across companies.
- If the entity/period isn't in the corpus, or evidence is missing, abstain.
- Citations: source filename, page, short verbatim quote from a retrieved chunk.

Return AgentAnswer structured output.
"""


def answer_question(question: str, store: VectorStore | None = None) -> AgentAnswer:
    store = store or VectorStore()
    set_store(store)

    agent = Agent(
        model=OpenAIModel(model_id=Config.OPENAI_MODEL, params={"temperature": 0}),
        system_prompt=SYSTEM_PROMPT,
        tools=[retrieve_chunks, list_documents, get_full_chunk, think],
        callback_handler=None,
        structured_output_model=AgentAnswer,
    )
    result = agent(question)
    answer = result.structured_output
    if answer is None:
        return AgentAnswer(abstained=True, answer=None, citations=[])
    if answer.abstained or not answer.answer or not answer.citations:
        return AgentAnswer(abstained=True, answer=None, citations=[])
    return answer


def format_answer(answer: AgentAnswer) -> str:
    if answer.abstained or not answer.answer:
        return "ABSTAINED: Not found in the provided documents."
    lines = [f"ANSWER: {answer.answer}", "CITATIONS:"]
    for c in answer.citations:
        quote = c.quote.replace("\n", " ").strip()
        lines.append(f'- {c.document}, p.{c.page}: "{quote}"')
    return "\n".join(lines)
