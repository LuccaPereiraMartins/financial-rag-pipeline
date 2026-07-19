# Semiconductor Earnings RAG

CLI-first RAG over earnings releases and call transcripts for Lam Research, KLA, Applied Materials, and ASML (2024–25).

## Setup (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

## Ingest

```powershell
python ingest.py --input data/reports
```

Re-runnable and idempotent: unchanged PDFs (same content hash per filename) are skipped. Use `--force` to rebuild.

A pre-built index lives in `data/index/` after ingest, so `ask.py` works without re-ingesting.

## Ask

```powershell
python ask.py "What net bookings did ASML report for Q3 2024, and how did management characterise them on the call?"
```

Output is either:

```
ANSWER: ...
CITATIONS:
- <document>, p.<n>: "..."
```

or:

```
ABSTAINED: Not found in the provided documents.
```

## Notes

- Models: `gpt-4o-mini` (agent) + `text-embedding-3-small` (embeddings), set in `src/config.py`.
- Vector store: local Chroma at `data/index/`.
- Company / doc type / date come from folder layout and filename at ingest (stand-in for production CMS metadata).
- Fiscal calendars differ by company; the agent resolves periods from filenames, `doc_date`, and source text.
