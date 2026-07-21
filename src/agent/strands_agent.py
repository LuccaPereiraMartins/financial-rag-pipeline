"""Strands agent: retrieve + think, then structured answer."""

from strands import Agent
from strands.models.openai import OpenAIModel
from strands_tools import think

from src.agent.schemas import AgentAnswer
from src.agent.tools import get_full_chunk, retrieve_chunks
from src.config import Config

# agent is quite thin, so put prompt here instead of another module.
SYSTEM_PROMPT = """You are a financial research assistant. Answer questions ONLY from the
documents currently indexed in the vector store (earnings materials and related filings
ingested into this system).

Rules:
- Never use parametric / general knowledge. Every claim must come from retrieved chunks.
- If the question is about an entity or topic that is clearly not covered by what you can
  retrieve from this index, abstain immediately — do not invent an answer. You may skip
  tools when it is obvious the corpus cannot help (e.g. a consumer brand with no filings here).
- Otherwise call retrieve_chunks before answering. Scope by company when the question names one
  and that company exists in the index metadata.
- If retrieval finds nothing relevant, abstain.
- Filenames and doc_date are in chunk text/metadata — use them to resolve reporting periods
  (issuers often use different fiscal calendars; read the source, don't invent mappings).
- Use the think tool only when comparing periods or aggregating across companies/documents.
- Citations: source filename, page, short verbatim quote from a retrieved chunk.

Return AgentAnswer structured output.
"""


def answer_question(question: str) -> AgentAnswer:
    _model = OpenAIModel(
        # NOTE: could use global openai client if needed (eg async or rate limited)
        # NOTE: we do not pass temperature to avoid having to support different models
        model_id=Config.OPENAI_MODEL,
    )

    agent = Agent(
        model=_model,
        system_prompt=SYSTEM_PROMPT,
        tools=[retrieve_chunks, get_full_chunk, think],
        callback_handler=None,
        structured_output_model=AgentAnswer,
    )

    result = agent(question)
    answer = result.structured_output
    # in case of any errors, abstain. NOTE: could instead state it's an error if better
    if answer is None:
        return AgentAnswer(abstained=True, answer=None, citations=[])
    if answer.abstained or not answer.answer or not answer.citations:
        return AgentAnswer(abstained=True, answer=None, citations=[])
    return answer


def format_answer(answer: AgentAnswer) -> str:
    # some light formatting for the answer, pulling parts out of the response class
    if answer.abstained or not answer.answer:
        return "ABSTAINED: Not found in the provided documents."
    lines = [f"ANSWER: {answer.answer}", "CITATIONS:"]
    for c in answer.citations:
        quote = c.quote.replace("\n", " ").strip()
        lines.append(f'- {c.document}, p.{c.page}: "{quote}"')
    return "\n".join(lines)
