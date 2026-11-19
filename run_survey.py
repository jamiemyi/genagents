"""Run the bundled survey agent using the project's Survey environment.

From the repository root:
    python run_survey.py                 # first four items
    python run_survey.py --all            # full 55-item questionnaire
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from environment.survey.survey import Survey


ROOT = Path(__file__).resolve().parent
DEFAULT_AGENT_ID = "01fd7d2a-0357-4c1b-9f3e-8eade2d537ae"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-id", default=DEFAULT_AGENT_ID)
    parser.add_argument(
        "--population",
        type=Path,
        default=ROOT / "agent_bank/populations/survey_agents",
    )
    parser.add_argument(
        "--questionnaire",
        type=Path,
        default=ROOT / "environment/survey/lsms_like_questionnaire.json",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "survey_run")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--all", action="store_true", help="run all questionnaire items")
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be a positive integer")
    with args.questionnaire.open() as stream:
        questionnaire = json.load(stream)
    if not isinstance(questionnaire, dict) or not questionnaire:
        parser.error("questionnaire must be a non-empty question-to-options object")

    selected = list(questionnaire.items())
    if not args.all:
        selected = selected[:args.batch_size]

    survey = Survey()
    survey.load_agents([{
        "population": str(args.population),
        "agent_id": args.agent_id,
    }])

    for start in range(0, len(selected), args.batch_size):
        batch = dict(selected[start:start + args.batch_size])
        survey.survey(batch, num_threads=1)
        survey.save(str(args.output))
        print(f"Saved {min(start + len(batch), len(selected))}/{len(selected)} responses")

    if isinstance(survey.responses, pd.DataFrame):
        print(survey.responses.to_string(index=False))
    print(f"Responses saved under {args.output}")


if __name__ == "__main__":
    main()
