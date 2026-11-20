"""Build and evaluate survey agents from mapped CSV, JSON, or JSONL records."""

import argparse
import csv
import hashlib
import json
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path


def _text(value):
    return "" if value is None else str(value).strip()


def _code(value):
    value = _text(value)
    if not value:
        return ""
    try:
        number = Decimal(value)
        if number.is_finite():
            return format(number.normalize(), "f")
    except InvalidOperation:
        pass
    return value


def _load_config(path):
    config = json.loads(path.read_text())
    if (not isinstance(config, dict) or not isinstance(config.get("source"), str) or
            not config["source"].strip()):
        raise ValueError("Config needs a source path")
    ids, questions = config.get("id_columns"), config.get("questions")
    if (not isinstance(ids, list) or not ids or
            any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids)):
        raise ValueError("id_columns must contain distinct column names")
    if not isinstance(questions, list) or not questions:
        raise ValueError("questions must be a non-empty list")
    seen_ids, seen_texts, seen_columns, train, test = set(), set(), set(), [], []
    for item in questions:
        if not isinstance(item, dict):
            raise ValueError("Each question mapping must be an object")
        qid, column, prompt, role = (item.get(key) for key in ("id", "column", "text", "role"))
        if any(not isinstance(x, str) or not x.strip() for x in (qid, column, prompt)):
            raise ValueError("Each question needs id, column, and text")
        if role not in ("train", "test"):
            raise ValueError(f"Question {qid} role must be train or test")
        normalized_text = " ".join(prompt.split()).casefold()
        if qid in seen_ids or column in seen_columns or normalized_text in seen_texts:
            raise ValueError(f"Repeated question id, column, or text: {qid}")
        seen_ids.add(qid)
        seen_columns.add(column)
        seen_texts.add(normalized_text)
        options = item.get("options")
        if (not isinstance(options, list) or len(options) < 2 or
                any(not isinstance(x, str) or not x.strip() for x in options) or
                len(set(options)) != len(options)):
            raise ValueError(f"Question {qid} needs distinct answer options")
        value_map = item.get("value_map", {})
        if not isinstance(value_map, dict) or any(value not in options for value in value_map.values()):
            raise ValueError(f"Question {qid} value_map must map source codes to options")
        normalized_map = {_code(key): value for key, value in value_map.items()}
        if len(normalized_map) != len(value_map):
            raise ValueError(f"Question {qid} has duplicate normalized source codes")
        missing = item.get("missing_values", [])
        if not isinstance(missing, list):
            raise ValueError(f"Question {qid} missing_values must be a list")
        item = {**item, "value_map": normalized_map,
                "missing_values": {_code(x) for x in missing}}
        (train if role == "train" else test).append(item)
    if not train or not test:
        raise ValueError("questions need at least one train and one test item")
    profile = config.get("profile_columns", {})
    if (not isinstance(profile, dict) or
            any(not isinstance(v, str) or not v for v in profile.values())):
        raise ValueError("profile_columns must map profile names to source columns")
    if ("data_origin" in config and
            (not isinstance(config["data_origin"], str) or not config["data_origin"].strip())):
        raise ValueError("data_origin must be non-empty text")
    test_columns = {q["column"] for q in test}
    protected = set(profile.values()) | set(ids)
    protected |= {config[key] for key in ("cohort_column", "weight_column") if config.get(key)}
    if protected & test_columns:
        raise ValueError("Held-out columns cannot appear in profile, ID, cohort, or weight")
    sample_size = config.get("sample_size")
    if sample_size is not None and (type(sample_size) is not int or sample_size < 1):
        raise ValueError("sample_size must be a positive integer or null")
    if type(config.get("sample_seed", 0)) is not int:
        raise ValueError("sample_seed must be an integer")
    prefix = config.get("agent_id_prefix", "survey")
    if (not isinstance(prefix, str) or not prefix or
            any(not (char.isalnum() or char in "._-") for char in prefix)):
        raise ValueError("agent_id_prefix has invalid characters")
    return config, (path.parent / config["source"]).resolve(), train, test


def _rows(source):
    suffix = source.suffix.lower()
    if suffix == ".csv":
        with source.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ValueError("Source CSV needs unique column names")
            for line, row in enumerate(reader, 2):
                yield line, row
    elif suffix == ".jsonl":
        with source.open(encoding="utf-8") as stream:
            for line, text in enumerate(stream, 1):
                if text.strip():
                    yield line, json.loads(text)
    elif suffix == ".json":
        records = json.loads(source.read_text())
        if not isinstance(records, list):
            raise ValueError("JSON source must be a list of respondent objects")
        for line, row in enumerate(records, 1):
            yield line, row
    else:
        raise ValueError("Source must be .csv, .json, or .jsonl")


def _answer(row, question, line):
    source_value = _text(row[question["column"]])
    raw = _code(source_value)
    if not raw or raw in question["missing_values"]:
        return None
    answer = (source_value if source_value in question["options"] else
              question["value_map"].get(raw, raw))
    if answer not in question["options"]:
        raise ValueError(f"Unknown answer {question['column']}={raw!r} at source row {line}")
    return answer


def _weight(row, column, line):
    value = _text(row[column]) if column else "1"
    try:
        weight = Decimal(value)
    except InvalidOperation:
        raise ValueError(f"Invalid weight at source row {line}: {value!r}") from None
    if not weight.is_finite() or weight <= 0:
        raise ValueError(f"Weight must be positive at source row {line}")
    return value


def prepare(config_path, output):
    """Write training inputs and held-out labels atomically; never build agents."""
    config_path, output = Path(config_path), Path(output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    config, source, train, test = _load_config(config_path)
    if not source.is_file():
        raise FileNotFoundError(f"Source not found: {source}")
    needed = (set(config["id_columns"]) | {q["column"] for q in train + test} |
              set(config.get("profile_columns", {}).values()))
    for key in ("cohort_column", "weight_column"):
        if config.get(key):
            needed.add(config[key])
    eligible, seen = [], set()
    skipped_missing_id = skipped_incomplete = 0
    for line, row in _rows(source):
        if not isinstance(row, dict):
            raise ValueError(f"Source row {line} must be an object")
        if not needed.issubset(row):
            raise ValueError(f"Source row {line} lacks columns: {sorted(needed - set(row))}")
        key = tuple(_text(row[column]) for column in config["id_columns"])
        if not all(key):
            skipped_missing_id += 1
            continue
        if key in seen:
            raise ValueError(f"Duplicate respondent key {key} at source row {line}")
        seen.add(key)
        test_answers = [_answer(row, question, line) for question in test]
        train_answers = [(question, _answer(row, question, line)) for question in train]
        train_answers = [(question, answer) for question, answer in train_answers
                         if answer is not None]
        cohort_column = config.get("cohort_column")
        cohort = _text(row[cohort_column]) if cohort_column else "all"
        if any(answer is None for answer in test_answers) or not train_answers or not cohort:
            skipped_incomplete += 1
            continue
        weight = _weight(row, config.get("weight_column"), line)
        agent_id = config.get("agent_id_prefix", "survey") + "-" + hashlib.sha256(
            json.dumps(key, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
        eligible.append((key, agent_id, cohort, weight, row, train_answers, test_answers))
    if not eligible:
        raise ValueError("No records have both training and held-out answers")
    if len({item[1] for item in eligible}) != len(eligible):
        raise ValueError("Agent ID hash collision")
    if config.get("sample_size") is not None:
        seed = config.get("sample_seed", 0)
        eligible.sort(key=lambda item: hashlib.sha256(
            f"{seed}:{item[1]}".encode("utf-8")).hexdigest())
        eligible = eligible[:config["sample_size"]]
    eligible.sort(key=lambda item: item[1])

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".survey-prepare-", dir=output.parent) as temp:
        staged = Path(temp) / "run"
        inputs = staged / "inputs"
        inputs.mkdir(parents=True)
        agents, observed, source_ids = [], [], []
        for key, agent_id, cohort, weight, row, train_answers, test_answers in eligible:
            profile = {name: _text(row[column]) for name, column in
                       config.get("profile_columns", {}).items() if _text(row[column])}
            builder_input = {
                "id": agent_id, "profile": profile,
                "data_origin": config.get("data_origin", str(source)),
                "responses": [{"question": q["text"], "answer": answer,
                               "options": q["options"]} for q, answer in train_answers],
            }
            (inputs / f"{agent_id}.json").write_text(json.dumps(builder_input, indent=2) + "\n")
            agents.append({"agent_id": agent_id, "folder": f"agents/{agent_id}",
                           "cohort": cohort, "weight": weight})
            source_ids.append({"agent_id": agent_id, **dict(zip(config["id_columns"], key))})
            for question, answer in zip(test, test_answers):
                observed.append({"participant_id": agent_id, "question_id": question["id"],
                                 "cohort": cohort, "answer": answer, "weight": weight})
        (staged / "questions.json").write_text(json.dumps([
            {"id": q["id"], "text": q["text"], "options": q["options"]} for q in test
        ], indent=2) + "\n")
        for name, fields, rows in (
            ("agents.csv", ["agent_id", "folder", "cohort", "weight"], agents),
            ("observed.csv", ["participant_id", "question_id", "cohort", "answer", "weight"], observed),
            ("source_ids.csv", ["agent_id", *config["id_columns"]], source_ids),
        ):
            with (staged / name).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        (staged / "manifest.json").write_text(json.dumps({
            "source": str(source), "agent_count": len(agents),
            "skipped_missing_id": skipped_missing_id,
            "skipped_incomplete": skipped_incomplete,
            "held_out_questions": [q["id"] for q in test],
            "sample_size": config.get("sample_size"),
            "sample_seed": config.get("sample_seed", 0),
        }, indent=2) + "\n")
        staged.rename(output)
    return len(eligible)


def build_agents(output):
    from build_survey_agent import build

    output = Path(output)
    sources = sorted((output / "inputs").glob("*.json"))
    if not sources:
        raise ValueError(f"No prepared respondent inputs under {output / 'inputs'}")
    for source in sources:
        build(source, output / "agents")
    return len(sources)


def predict_and_score(output):
    from evaluation.survey import predict, score

    output = Path(output)
    predict(output / "questions.json", output / "agents.csv", output / "predicted.csv")
    return score(output / "questions.json", output / "observed.csv",
                 output / "predicted.csv", output / "score")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "build", "evaluate", "all"))
    parser.add_argument("--config", type=Path, help="mapping JSON; required for prepare/all")
    parser.add_argument("--output", type=Path, required=True,
                        help="run directory; use a new path for prepare/all")
    args = parser.parse_args()
    if args.stage in ("prepare", "all") and args.config is None:
        parser.error("--config is required for prepare/all")
    if args.stage in ("prepare", "all"):
        print(f"Prepared {prepare(args.config, args.output)} respondent inputs")
    if args.stage in ("build", "all"):
        print(f"Built {build_agents(args.output)} survey agents")
    if args.stage in ("evaluate", "all"):
        print(f"Scores: {predict_and_score(args.output)}")


if __name__ == "__main__":
    main()
