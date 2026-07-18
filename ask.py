"""CLI entrypoint: python ask.py "<question>" """

import argparse
import sys

from src.agent.strands_agent import answer_question, format_answer

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask a question over the earnings corpus")
    parser.add_argument("question", help="Question to answer")
    args = parser.parse_args(argv)
    print(format_answer(answer_question(args.question)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
