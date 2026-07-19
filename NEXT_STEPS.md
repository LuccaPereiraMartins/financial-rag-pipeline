# Next steps

## Done (first draft)

- `ingest.py` / `ask.py` wired from repo root
- PDF parse → LangChain recursive prose chunks + whole-table blocks → Chroma at `data/index/`
- Strands agent: `retrieve_chunks`, `get_full_chunk`, `think`
- Metadata from folder/filename at ingest (production stand-in for CMS / Bedrock KB + S3)

## Still missing

### 1. One re-ingest after parsing/chunking change (cost)

Chunking + metadata schema changed. When ready, run **once**:

```powershell
python ingest.py --input data/reports --force
```

Then freeze the index while iterating on agent/prompt only.

### 2. Smoke the five case-study examples

1. ASML Q3 2024 net bookings + call characterisation  
2. China % of revenue, Dec 2024 quarterly results (multi-company)  
3. KLA services revenue, Dec 2024 quarter (table)  
4. Lam revenue guidance Fiscal 2Q25 → 3Q25  
5. TSMC 2025 capex (must abstain)

### 3. Submission polish

- [ ] `.env.example` with blank `OPENAI_API_KEY=`
- [ ] Ship `data/index/` in the zip
- [ ] `writeup.pdf`

### 4. Optional (secondary)

- [ ] Thin API (`POST /ask`)
- [ ] Small eval set
- [ ] Dockerfile

## Design notes

- Abstain via empty/irrelevant retrieve + known corpus companies in the prompt (no document-catalog tool).
- No fiscal-calendar module: agent reads filename/`doc_date` + content.
- Tables kept as whole chunks; prose via LangChain `RecursiveCharacterTextSplitter`.
- Metadata known at ingest: see `pdf_parser.py` module docstring.

## Things to consider

- tracks costs (bulk ingestion, delta ingestion, single agent run (llm + embedding query))
- dockerfile, lightweight UI
- latex report and PDF version
- consider github actions for CI/CD if easily done, just check if health deploys
- API layer can have the pure retrieval tools exposed, and the ask endpoint exposed

## Files left to review

- whole repo should be written and understandable from first look
- flatten any thin modules where possible
