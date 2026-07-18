"""Strands agent: retrieve + think, then structured answer."""

from strands import Agent
from strands.models.openai import OpenAIModel
from strands_tools import think

from src.agent.schemas import AgentAnswer
from src.agent.tools import get_full_chunk, retrieve_chunks
from src.config import Config

# agent is quite thin, so put prompt here instead of another module.
# evaluation states only on provided material, but the prompt could be made more generalised when needed
SYSTEM_PROMPT = """You answer questions ONLY from the semiconductor earnings corpus
(Lam Research, KLA, Applied Materials, ASML).

Rules:
- Never use parametric knowledge. Every claim must come from retrieved chunks.
- Call retrieve_chunks before answering. Scope by company when the question names one.
- If a company is outside this corpus (e.g. TSMC), or retrieval finds nothing relevant, abstain.
- Filenames and doc_date are in chunk text/metadata — use them to resolve periods
  (companies have different fiscal calendars; read the source, don't invent mappings).
- Use the think tool when comparing periods or aggregating across companies.
- Citations: source filename, page, short verbatim quote from a retrieved chunk.

Return AgentAnswer structured output.
"""


def answer_question(question: str) -> AgentAnswer:
    _model = OpenAIModel(
        # could use global openai client if needed (eg async or rate limited)
        model_id=Config.OPENAI_MODEL, 
        params={"temperature": 0} # NOTE: temperature 0 is default, but good to be explicit
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
