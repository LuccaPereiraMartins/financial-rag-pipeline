# Next steps

## Done (first draft)

- `ingest.py` / `ask.py` wired from repo root
- PDF parse → chunk (prose + tables) → Chroma at `data/index/`
- Strands agent: `retrieve_chunks`, `list_documents`, `get_full_chunk`, `think`
- Light metadata: `company`, `doc_type`, `doc_date`, `source_file` (filename also in chunk text)

## Still missing

### 1. One re-ingest, then freeze (cost)

The index on disk still reflects the **old** schema (`calendar_period`, etc.). After you’re happy with parsing/chunking/agent changes, run **once**:

```powershell
python ingest.py --input data/reports --force
```

Then commit `data/index/` and **don’t re-ingest during iteration** — tune the agent, prompts, and tools against the frozen index. Only re-ingest when chunking/metadata actually change, or when new PDFs are added (normal ingest skips unchanged files).

### 2. Smoke the five case-study examples

Run manually (space questions out to avoid rate limits):

1. ASML Q3 2024 net bookings + call characterisation  
2. China % of revenue, Dec 2024 quarterly results (multi-company)  
3. KLA services revenue, Dec 2024 quarter (table)  
4. Lam revenue guidance Fiscal 2Q25 → 3Q25  
5. TSMC 2025 capex (must abstain)

Fix retrieval/prompt only unless answers clearly need better chunks.

### 3. Submission polish

- [ ] Update [README.md](README.md) (drop stale fiscal-calendar note; Windows setup is mostly there)
- [ ] `.env.example` with blank `OPENAI_API_KEY=`
- [ ] Ship `data/index/` in the zip (pre-built index)
- [ ] `writeup.pdf` (architecture, models/cost, limitations, next steps, where AI was used)

### 4. Optional (secondary)

- [ ] Thin `api.py` (`POST /ask`) sharing `answer_question`
- [ ] `eval/qa_set.json` + small `run_eval.py` (6–8 pairs, not a big harness)
- [ ] Dockerfile

## Deliberately not doing now

- Hardcoded fiscal-calendar module (agent + filename/`doc_date` + `think`)
- Heavy citation validator / retry loops (trust structured output + manual smoke)
- Parquet index (Chroma is enough for this corpus and metadata filters)

## Cost tips

- **Embeddings** dominate ingest cost — avoid `--force` while experimenting on agent code.
- **Agent**: `think` and multi-hop retrieval add LLM calls; use sparingly in prompt if budget is tight.
- Consider `gpt-4o-mini` throughout; bump model only if smoke tests fail on reasoning, not retrieval.


## Things to consider

- NOTEs and TODOs throughout the code
- tracks costs (bulk ingestion, delta ingestion, single agent run)
- dockerfile, lightweight UI
- check what we need tiktoken package for?
- latex report and PDF version