"""Ingestion pipeline: PDF folder → chunks → Chroma index.

Production assumption: each document arrives with known company / doc_type /
doc_date (see src/parsing/pdf_parser.py module docstring). Here we derive those
from folder layout + filename conventions for the case-study corpus.
"""

import argparse
import re
import sys
from pathlib import Path

from src.index.vector_store import VectorStore
from src.parsing.chunker import document_to_chunks
from src.parsing.pdf_parser import file_hash, parse_pdf

# NOTE: Metadata is resolved below from path/filename in a hardcoded way to match our corpus
# in practice, our production ingestion handles this. It could be handled with LLMs, at a cost


def _doc_type_from_filename(name: str) -> str:
    lower = name.lower()
    if "earnings_call" in lower or "earnings call" in lower:
        return "earnings_call_transcript"
    return "earnings_release"


def _doc_date_from_filename(name: str) -> str:
    m = re.search(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})", name)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    months = {
        "january": "01", "february": "02", "march": "03", "april": "04",
        "may": "05", "june": "06", "july": "07", "august": "08",
        "september": "09", "october": "10", "november": "11", "december": "12",
    }
    m = re.search(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)"
        r"[_-](\d{1,2})[_-](20\d{2})",
        name,
        flags=re.IGNORECASE,
    )
    if m:
        return f"{m.group(3)}-{months[m.group(1).lower()]}-{int(m.group(2)):02d}"
    return ""


def ingest_folder(input_dir: str | Path, force: bool = False) -> dict:
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input folder not found: {input_path}")

    store = VectorStore()
    pdfs = sorted(input_path.rglob("*.pdf"))
    stats = {"found": len(pdfs), "processed": 0, "skipped": 0, "chunks_added": 0, "errors": []}

    for pdf in pdfs:
        try:
            fhash = file_hash(pdf)
            stored_hash = None if force else store.stored_file_hash(pdf.name)
            if stored_hash == fhash:
                stats["skipped"] += 1
                continue

            # Stand-in for production metadata attached at upload time
            company = pdf.parent.name
            doc_type = _doc_type_from_filename(pdf.name)
            doc_date = _doc_date_from_filename(pdf.name)

            doc = parse_pdf(pdf, company=company, doc_type=doc_type, doc_date=doc_date)
            chunks = document_to_chunks(doc)
            # Drop prior chunks for this filename (content changed or --force)
            if stored_hash is not None or force:
                store.delete_by_source_file(pdf.name)
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
