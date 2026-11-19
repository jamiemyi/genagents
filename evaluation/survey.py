"""Predict held-out survey answers and compare them with observed distributions."""

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    options: tuple[str, ...]


def load_questions(path: Path) -> dict[str, Question]:
    data = json.loads(path.read_text())
    if not isinstance(data, list) or not data:
        raise ValueError("Questions must be a non-empty JSON list")
    questions = {}
    texts = set()
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Each question must be an object")
        question_id, text, options = (item.get(key) for key in ("id", "text", "options"))
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError("Each question needs a non-empty id")
        if question_id in questions:
            raise ValueError(f"Duplicate question id: {question_id}")
        normalized_text = " ".join(text.split()).casefold() if isinstance(text, str) else ""
        if not normalized_text or normalized_text in texts:
            raise ValueError(f"Question text is empty or repeated: {text!r}")
        if not isinstance(options, list) or len(options) < 2 or any(
            not isinstance(option, str) or not option.strip() for option in options
        ) or len(set(options)) != len(options):
            raise ValueError(f"Question {question_id} needs distinct, non-empty options")
        questions[question_id] = Question(question_id, text, tuple(options))
        texts.add(normalized_text)
    return questions


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if (not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames))
                or not required.issubset(reader.fieldnames)):
            raise ValueError(f"{path} needs columns: {', '.join(sorted(required))}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} has no data rows")
    return rows


def positive_weight(value: str, where: str) -> float:
    try:
        weight = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid weight at {where}: {value!r}") from None
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError(f"Weight must be positive and finite at {where}")
    return weight


def load_agents(path: Path) -> list[dict]:
    rows = read_csv(path, {"agent_id", "folder", "cohort"})
    agents, seen = [], set()
    for line_number, row in enumerate(rows, 2):
        agent_id, folder, cohort = (
            (row.get(key) or "").strip() for key in ("agent_id", "folder", "cohort")
        )
        if not agent_id or not folder or not cohort:
            raise ValueError(f"Missing agent_id, folder, or cohort at {path}:{line_number}")
        key = cohort, agent_id
        if key in seen:
            raise ValueError(f"Duplicate agent in cohort at {path}:{line_number}: {key}")
        seen.add(key)
        agents.append({
            "agent_id": agent_id,
            "folder": (path.parent / folder).resolve(),
            "cohort": cohort,
            "weight": positive_weight(row.get("weight") or "1", f"{path}:{line_number}"),
        })
    return agents


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def predict(questions_path: Path, agents_path: Path, output_path: Path,
            exclude_groups: tuple[str, ...] = ()) -> int:
    # Import only for prediction; scoring needs no model or ML dependencies.
    from genagents.survey_agent import SurveyAgent

    questions = load_questions(questions_path)
    agents = load_agents(agents_path)
    targets = {question.text: list(question.options) for question in questions.values()}
    rows = []
    for entry in agents:
        agent = SurveyAgent(entry["folder"])
        if str(agent.meta.get("id")) != entry["agent_id"]:
            raise ValueError(f"Agent ID does not match folder: {entry['folder']}")
        # Exclude every evaluation item from the prompt, including items not in
        # the current four-question block. Never use recorded_resp for evaluation.
        answers = agent.categorical_resp(
            targets, exclude_item_ids=tuple(questions), exclude_groups=exclude_groups
        )["responses"]
        if len(answers) != len(questions):
            raise ValueError(f"Wrong answer count from agent {entry['agent_id']}")
        for question, answer in zip(questions.values(), answers):
            rows.append({
                "agent_id": entry["agent_id"], "question_id": question.id,
                "cohort": entry["cohort"], "answer": answer,
                "weight": entry["weight"],
            })
    write_csv(output_path, ["agent_id", "question_id", "cohort", "answer", "weight"], rows)
    return len(rows)


def load_answers(path: Path, id_column: str,
                 questions: dict[str, Question]) -> dict[tuple[str, str], list[tuple[str, float]]]:
    rows = read_csv(path, {id_column, "question_id", "cohort", "answer"})
    groups = defaultdict(list)
    seen = set()
    for line_number, row in enumerate(rows, 2):
        person, question_id, cohort, answer = (
            (row.get(key) or "").strip()
            for key in (id_column, "question_id", "cohort", "answer")
        )
        if not person or not cohort or question_id not in questions:
            raise ValueError(f"Missing ID/cohort or unknown question at {path}:{line_number}")
        if answer not in questions[question_id].options:
            raise ValueError(f"Answer outside options at {path}:{line_number}: {answer!r}")
        key = cohort, question_id, person
        if key in seen:
            raise ValueError(f"Duplicate respondent and question at {path}:{line_number}")
        seen.add(key)
        weight = positive_weight(row.get("weight") or "1", f"{path}:{line_number}")
        groups[(cohort, question_id)].append((answer, weight))
    return groups


def distribution(rows: list[tuple[str, float]], options: tuple[str, ...]) -> dict[str, float]:
    counts = {option: 0.0 for option in options}
    for answer, weight in rows:
        counts[answer] += weight
    total = sum(counts.values())
    return {option: counts[option] / total for option in options}


def score(questions_path: Path, observed_path: Path, predicted_path: Path,
          output_dir: Path) -> dict:
    questions = load_questions(questions_path)
    observed = load_answers(observed_path, "participant_id", questions)
    predicted = load_answers(predicted_path, "agent_id", questions)
    if observed.keys() != predicted.keys():
        missing_predictions = sorted(observed.keys() - predicted.keys())
        missing_observations = sorted(predicted.keys() - observed.keys())
        raise ValueError(
            f"Question/cohort mismatch: missing predictions {missing_predictions}; "
            f"missing observations {missing_observations}"
        )
    if set(questions) != {question_id for _, question_id in observed}:
        raise ValueError("Some questions in the manifest have no observed or predicted rows")

    summary_rows, option_rows = [], []
    for cohort, question_id in sorted(observed):
        question = questions[question_id]
        observed_shares = distribution(observed[(cohort, question_id)], question.options)
        predicted_shares = distribution(predicted[(cohort, question_id)], question.options)
        tvd = 0.5 * sum(
            abs(observed_shares[option] - predicted_shares[option])
            for option in question.options
        )
        ranked_shares = sorted(predicted_shares.values(), reverse=True)
        summary_rows.append({
            "cohort": cohort, "question_id": question_id, "question": question.text,
            "observed_n": len(observed[(cohort, question_id)]),
            "predicted_n": len(predicted[(cohort, question_id)]),
            "observed_weight": sum(weight for _, weight in observed[(cohort, question_id)]),
            "predicted_weight": sum(weight for _, weight in predicted[(cohort, question_id)]),
            "n_options": len(question.options),
            "question_chars": len(question.text),
            "predicted_entropy": -sum(
                share * math.log(share) for share in ranked_shares if share > 0
            ),
            "predicted_max_share": ranked_shares[0],
            "predicted_top2_gap": ranked_shares[0] - ranked_shares[1],
            "tvd": tvd,
        })
        for option in question.options:
            option_rows.append({
                "cohort": cohort, "question_id": question_id, "option": option,
                "observed_share": observed_shares[option],
                "predicted_share": predicted_shares[option],
                "absolute_gap": abs(observed_shares[option] - predicted_shares[option]),
            })

    write_csv(output_dir / "summary.csv", list(summary_rows[0]), summary_rows)
    write_csv(output_dir / "options.csv", list(option_rows[0]), option_rows)
    report = {
        "question_cohort_pairs": len(summary_rows),
        "mean_tvd": sum(row["tvd"] for row in summary_rows) / len(summary_rows),
        "note": "Each question/cohort pair contributes equally to mean_tvd.",
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    predict_parser = subparsers.add_parser("predict", help="predict held-out questions")
    predict_parser.add_argument("--questions", type=Path, required=True)
    predict_parser.add_argument("--agents", type=Path, required=True)
    predict_parser.add_argument("--output", type=Path, required=True)
    predict_parser.add_argument("--exclude-group", action="append", default=[],
                                help="withhold a survey response group from the prompt")
    score_parser = subparsers.add_parser("score", help="compare predictions with observed data")
    score_parser.add_argument("--questions", type=Path, required=True)
    score_parser.add_argument("--observed", type=Path, required=True)
    score_parser.add_argument("--predicted", type=Path, required=True)
    score_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "predict":
            count = predict(args.questions, args.agents, args.output,
                            tuple(args.exclude_group))
            print(f"Wrote {count} predictions to {args.output}")
        else:
            report = score(args.questions, args.observed, args.predicted,
                           args.output_dir)
            print(json.dumps(report, indent=2))
    except (ValueError, OSError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
