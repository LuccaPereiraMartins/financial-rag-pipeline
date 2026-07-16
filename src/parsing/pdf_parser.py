"""PDF text + table extraction and light document metadata."""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pdfplumber

COMPANY_ALIASES = {
    "Lam Research": ["lam research", "lam research corp", "lrcx"],
    "KLA": ["kla corporation", "kla corp", "kla-tencor", " kla "],
    "Applied Materials": ["applied materials", "applied material", "amat"],
    "ASML": ["asml holding", "asml"],
}


@dataclass
class PageUnit:
    text: str
    page: int
    content_type: str  # "prose" | "table"


@dataclass
class ParsedDocument:
    path: Path
    file_hash: str
    company: str
    doc_type: str  # earnings_release | earnings_call_transcript
    doc_date: str  # YYYY-MM-DD from filename when present, else ""
    units: list[PageUnit] = field(default_factory=list)


def _file_hash(path: Path) -> str:
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


def detect_company(filename: str, page1_text: str = "", parent: str = "") -> str:
    blob = f"{filename}\n{parent}\n{page1_text}".lower().replace("_", " ").replace("-", " ")
    scores = {}
    for company, aliases in COMPANY_ALIASES.items():
        for alias in aliases:
            if alias in blob:
                scores[company] = scores.get(company, 0) + len(alias)
    if scores:
        return max(scores, key=scores.get)
    return "Unknown"


def detect_doc_type(filename: str, page1_text: str) -> str:
    blob = f"{filename}\n{page1_text}".lower()
    if any(h in blob for h in ("earnings_call", "earnings call", "conference call", "prepared remarks")):
        return "earnings_call_transcript"
    if any(h in blob for h in ("earnings release", "press release", "financial results", "exhibit 99")):
        return "earnings_release"
    return "earnings_release"


def date_from_filename(filename: str) -> str:
    """Pull a release/call date from the filename when present. Empty if none."""
    m = re.search(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})", filename)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    # Quarter-Ended-December-29-2024
    months = {
        "january": "01", "february": "02", "march": "03", "april": "04",
        "may": "05", "june": "06", "july": "07", "august": "08",
        "september": "09", "october": "10", "november": "11", "december": "12",
    }
    m = re.search(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)"
        r"[_-](\d{1,2})[_-](20\d{2})",
        filename,
        flags=re.IGNORECASE,
    )
    if m:
        return f"{m.group(3)}-{months[m.group(1).lower()]}-{int(m.group(2)):02d}"
    return ""


def parse_pdf(path: Path, llm_fallback: Any = None) -> ParsedDocument:
    path = Path(path)
    units: list[PageUnit] = []
    page1_text = ""

    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_num = i + 1
            text = page.extract_text() or ""
            if page_num == 1:
                page1_text = text

            tables = page.extract_tables() or []
            prose = text.strip()
            if prose:
                units.append(PageUnit(text=prose, page=page_num, content_type="prose"))
            for table in tables:
                md = table_to_markdown(table)
                if md.strip():
                    units.append(PageUnit(text=md, page=page_num, content_type="table"))

    company = detect_company(path.name, page1_text, parent=path.parent.name)
    if company == "Unknown" and llm_fallback is not None:
        try:
            result = llm_fallback(path.name, page1_text)
            company = detect_company(result.get("company", ""), "") or "Unknown"
        except Exception:
            pass

    return ParsedDocument(
        path=path,
        file_hash=_file_hash(path),
        company=company,
        doc_type=detect_doc_type(path.name, page1_text),
        doc_date=date_from_filename(path.name),
        units=units,
    )
