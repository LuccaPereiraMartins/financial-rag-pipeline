# Financial Earnings RAG

CLI-first RAG for grounded Q&A over financial filings (earnings releases and call transcripts).

Answers cite sources, or the system abstains when the index does not support an answer.

## Setup (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

A pre-built Chroma index is included at `data/index/` so you can run `ask.py` without re-ingesting.

## Ingest

```powershell
python ingest.py --input data/reports
```

Idempotent: unchanged PDFs (same content hash per filename) are skipped. Use `--force` to rebuild.

Company, doc type, and date come from folder layout / filename at ingest (stand-in for production metadata). Tables are kept as whole chunks; prose uses LangChain `RecursiveCharacterTextSplitter`.

## Ask

```powershell
python ask.py "What net bookings did ASML report for Q3 2024, and how did management characterise them on the call?"
```

Output:

```
ANSWER: ...
CITATIONS:
- <document>, p.<n>: "..."
```

or:

```
ABSTAINED: Not found in the provided documents.
```

## Evaluation

Lightweight golden checks for the five case-study example questions:

```powershell
python -m src.evaluation
```

## API + MCP

```powershell
python app.py
```

- Swagger: http://localhost:8000/docs
- `POST /ask` — grounded agent answer
- `POST /search` — raw vector search
- MCP at http://localhost:8000/mcp/ — tools: `ask`, `retrieve_chunks`, `get_full_chunk`

## Docker

```powershell
docker compose up --build
```

Serves the API on port **8000** using `.env` for `OPENAI_API_KEY`. CI (`.github/workflows/docker-build.yml`) only checks that the image builds.

## Architecture (short)

```
PDFs → pdfplumber → chunks → OpenAI embeddings → Chroma
                         ↑
ask.py / POST /ask / MCP ask → Strands agent → retrieve tools → grounded answer
```

- Models: `gpt-5.4-nano` (agent) + `text-embedding-3-small` (embeddings) in `src/config.py`.
- The agent is a general financial research assistant over whatever is indexed.
- Fiscal calendars differ by issuer. We do **not** normalise periods to a shared calendar at ingest; the agent relies on filename/`doc_date` cues and wording inside the PDF (e.g. “quarter ended …”), so ambiguous asks like “results 2025” can be imperfect.
- Production direction: managed ingest (e.g. Bedrock Knowledge Bases + S3) and period/company normalising, real vector DB, OCR for tables/figures, separate ingest vs agent vs UI services — see `writeup.tex`.

## Write-up

One-page LaTeX write-up: `writeup.tex` and `writeup.pdf`.
