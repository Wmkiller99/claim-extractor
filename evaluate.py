"""Evaluate claim extraction against the labelled claims in data/eval_set.jsonl.

The LLM backend is the one selected by LLM_PROVIDER in .env (see .env.example).
"""

import argparse
import logging
import sys
from pathlib import Path

from claim_extractor.config import Settings, build_client
from claim_extractor.evaluation import format_report, load_cases, run_evaluation
from claim_extractor.llm.base import LLMError


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate claim extraction.")
    parser.add_argument("--data", type=Path, default=Path("data/eval_set.jsonl"))
    args = parser.parse_args()

    settings = Settings()
    # The entry point, not the library, decides how logs look.
    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    cases = load_cases(args.data)
    client = build_client(settings)

    try:
        results = run_evaluation(cases, client)
    except LLMError as exc:
        print(f"LLM backend error, evaluation stopped: {exc}", file=sys.stderr)
        return 1

    print(format_report(results, client.model))
    return 0


if __name__ == "__main__":
    sys.exit(main())
