"""CLI entrypoint: python ask.py "<question>" """

import argparse
import sys

from src.agent.strands_agent import answer_question, format_answer
from src.index.vector_store import VectorStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask a question over the earnings corpus")
    parser.add_argument("question", help="Question to answer")
    args = parser.parse_args(argv)

    store = VectorStore()
    if store.collection.count() == 0:
        print("ERROR: Index is empty. Run: python ingest.py --input data/reports", file=sys.stderr)
        return 1

    print(format_answer(answer_question(args.question, store=store)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
