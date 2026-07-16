"""Ingestion pipeline: PDF folder → chunks → Chroma index."""

import argparse
import json
import sys
from pathlib import Path

from openai import OpenAI

from src.config import Config
from src.index.vector_store import VectorStore
from src.parsing.chunker import document_to_chunks
from src.parsing.pdf_parser import _file_hash, parse_pdf


# TODO: remove this, state that we would have this info when we ingest the data and shouldn't
# be wasting LLM calls on this
def _llm_classify(filename: str, page1_text: str) -> dict:
    client = OpenAI()
    resp = client.chat.completions.create(
        model=Config.OPENAI_MODEL,
        messages=[
            {
                "role": "user",
                "content": (
                    "Classify this semiconductor earnings PDF. Return JSON with keys: "
                    "company (string), doc_type (earnings_release|earnings_call_transcript).\n\n"
                    f"Filename: {filename}\n\nPage 1 text:\n{page1_text[:2500]}"
                ),
            }
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(resp.choices[0].message.content or "{}")


def ingest_folder(input_dir: str | Path, force: bool = False) -> dict:
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input folder not found: {input_path}")

    store = VectorStore()
    existing = set() if force else store.ingested_file_hashes()
    pdfs = sorted(input_path.rglob("*.pdf"))
    stats = {"found": len(pdfs), "processed": 0, "skipped": 0, "chunks_added": 0, "errors": []}

    for pdf in pdfs:
        try:
            fhash = _file_hash(pdf)
            if fhash in existing and not force:
                stats["skipped"] += 1
                continue

            doc = parse_pdf(pdf, llm_fallback=_llm_classify)
            chunks = document_to_chunks(doc)
            if force or fhash in existing:
                store.delete_by_file_hash(fhash)
            n = store.upsert_chunks(chunks)
            stats["processed"] += 1
            stats["chunks_added"] += n
            print(f"[ok] {pdf.name} | {doc.company} | {doc.doc_type} | {doc.doc_date} | {n} chunks")
        except Exception as exc:
            stats["errors"].append({"file": str(pdf), "error": str(exc)})
            print(f"[error] {pdf}: {exc}", file=sys.stderr)

    print(
        f"\nDone. found={stats['found']} processed={stats['processed']} "
        f"skipped={stats['skipped']} chunks_added={stats['chunks_added']} "
        f"errors={len(stats['errors'])}"
    )
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest earnings PDFs into the vector index")
    parser.add_argument("--input", required=True, help="Folder of PDFs (searched recursively)")
    parser.add_argument("--force", action="store_true", help="Re-ingest even if already present")
    args = parser.parse_args(argv)
    ingest_folder(args.input, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
