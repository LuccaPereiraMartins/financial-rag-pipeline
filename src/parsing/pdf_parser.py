"""PDF text (+ table) extraction.

Production assumption
---------------------
In a real pipeline we would NOT reverse-engineer company / doc type / period from
raw PDF bytes. Documents would arrive from a controlled upload with metadata
already attached (company ticker, doc_type, report date / fiscal period, source URL).

This case-study corpus is a bulk folder of PDFs, so `ingest.py` supplies that
metadata from a reasonable stand-in:
  - company  = parent folder name (ASML, KLA, Lam Research, Applied Materials)
  - doc_type = filename convention (Earnings_Call* → transcript, else release)
  - doc_date = date embedded in the filename when present
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pdfplumber


@dataclass
class PageText:
    text: str
    page: int


@dataclass
class ParsedDocument:
    path: Path
    file_hash: str
    company: str
    doc_type: str  # earnings_release | earnings_call_transcript
    doc_date: str  # YYYY-MM-DD when known, else ""
    pages: list[PageText] = field(default_factory=list)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def table_to_markdown(table: list[list[Optional[str]]]) -> str:
    if not table:
        return ""
    rows = []
    for row in table:
        cells = [("" if c is None else str(c).replace("\n", " ").strip()) for c in row]
        if any(cells):
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    normalized = [r + [""] * (width - len(r)) for r in rows]
    header = normalized[0]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in normalized[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def parse_pdf(
    path: Path,
    *,
    company: str,
    doc_type: str,
    doc_date: str = "",
) -> ParsedDocument:
    """Extract page text. Metadata must be provided by the caller (ingest)."""
    path = Path(path)
    pages: list[PageText] = []

    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            parts: list[str] = []
            prose = (page.extract_text() or "").strip()
            if prose:
                parts.append(prose)
            # Append tables as markdown into the same page stream — no separate
            # prose/table chunking path. Fixed-size chunking may split a table;
            # acceptable tradeoff vs maintaining two pipelines.
            for table in page.extract_tables() or []:
                md = table_to_markdown(table)
                if md.strip():
                    parts.append(md)
            if parts:
                pages.append(PageText(text="\n\n".join(parts), page=i + 1))

    return ParsedDocument(
        path=path,
        file_hash=file_hash(path),
        company=company,
        doc_type=doc_type,
        doc_date=doc_date,
        pages=pages,
    )
