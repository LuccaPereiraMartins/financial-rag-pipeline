"""Lightweight golden checks for the five case-study example questions.

Run: python -m src.evaluation
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

from src.agent.strands_agent import answer_question, format_answer
from src.agent.tools import get_store


@dataclass
class EvalCase:
    name: str
    question: str
    must_abstain: bool = False
    # All of these substrings must appear in the answer (case-insensitive).
    must_include: list[str] = field(default_factory=list)
    # At least one of these must appear (case-insensitive).
    must_include_any: list[str] = field(default_factory=list)


# Key numbers taken from the indexed earnings releases / calls.
CASES: list[EvalCase] = [
    EvalCase(
        name="asml_q3_2024_bookings",
        question=(
            "What net bookings did ASML report for Q3 2024, and how did management "
            "characterise them on the call?"
        ),
        must_include_any=[
            "2,633", "2633", "2.633", "€2.6", "EUR 2.6", "EUR2.6", "2.6 billion", "2.6",
        ],
    ),
    EvalCase(
        name="china_dec_2024_share",
        question=(
            "Which companies quantified China as a share of revenue in their "
            "December 2024 quarterly results, and what were the figures?"
        ),
        must_include=["China"],
        must_include_any=["%", "percent"],
    ),
    EvalCase(
        name="kla_services_dec_2024",
        question=(
            "What was KLA's services revenue in its December 2024 quarter, "
            "per the earnings release?"
        ),
        must_include_any=["667", "667.4", "667,389"],
    ),
    EvalCase(
        name="lam_guidance_2q25_3q25",
        question=(
            "Did Lam Research's quarterly revenue guidance increase or decrease "
            "between its Fiscal 2Q25 and Fiscal 3Q25 results, and by how much?"
        ),
        must_include=["increase"],
        must_include_any=["4.30", "4.3", "4.65", "350", "0.35"],
    ),
    EvalCase(
        name="tsmc_capex_abstain",
        question="What capex guidance did TSMC give for 2025?",
        must_abstain=True,
    ),
]


def _norm(text: str) -> str:
    return text.casefold()


def check_case(case: EvalCase) -> tuple[bool, str, str]:
    """Return (passed, detail, formatted_output)."""
    answer = answer_question(case.question)
    formatted = format_answer(answer)

    if case.must_abstain:
        ok = answer.abstained or not answer.answer
        detail = "expected abstain" if ok else f"expected abstain, got: {answer.answer!r}"
        return ok, detail, formatted

    if answer.abstained or not answer.answer:
        return False, "unexpected abstain", formatted

    if not answer.citations:
        return False, "missing citations", formatted

    body = _norm(answer.answer)
    for needle in case.must_include:
        if _norm(needle) not in body:
            return False, f"missing required substring {needle!r}", formatted

    if case.must_include_any:
        if not any(_norm(n) in body for n in case.must_include_any):
            return False, f"missing any of {case.must_include_any!r}", formatted

    return True, "ok", formatted


def main() -> int:
    get_store()

    print(
        f"Running {len(CASES)} cases with {__import__('src.config', fromlist=['Config']).Config.OPENAI_MODEL}. "
        "Expect ~30s per case (reasoning model + tool loops)."
    )
    passed = 0
    for case in CASES:
        print(f"\n=== {case.name} ===", flush=True)
        print(f"Q: {case.question}", flush=True)
        try:
            ok, detail, formatted = check_case(case)
        except Exception as exc:  # noqa: BLE001 — keep suite running
            print(f"ERROR: {exc}", file=sys.stderr)
            ok, detail, formatted = False, f"exception: {exc}", ""
        if formatted:
            print(formatted)
        status = "PASS" if ok else "FAIL"
        print(f"--> {status}: {detail}", flush=True)
        if ok:
            passed += 1

    total = len(CASES)
    print(f"\nSummary: {passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
