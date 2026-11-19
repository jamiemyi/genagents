"""Build an interview-format agent from one respondent's LSMS answers."""

import argparse
import json
import re
from pathlib import Path


DEFAULT_POPULATION = Path(__file__).resolve().parent / "agent_bank/populations/survey_agents"
QUESTIONNAIRE = Path(__file__).resolve().parent / "environment/survey/lsms_like_questionnaire.json"


def _required_text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


def _validate_input(data, questionnaire):
    if not isinstance(data, dict):
        raise ValueError("Input must be a JSON object")
    agent_id = _required_text(data.get("id"), "id")
    if agent_id in (".", "..") or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", agent_id):
        raise ValueError("id may contain only letters, digits, dots, underscores, and hyphens")
    profile = data.get("profile", {})
    if not isinstance(profile, dict):
        raise ValueError("profile must be a JSON object")
    data_origin = data.get("data_origin", "respondent_reported")
    if not isinstance(data_origin, str) or not data_origin.strip():
        raise ValueError("data_origin must be non-empty text")
    responses = data.get("responses")
    if not isinstance(responses, list) or not responses:
        raise ValueError("responses must be a non-empty list")

    records, seen_questions = [], set()
    for index, item in enumerate(responses, 1):
        if not isinstance(item, dict):
            raise ValueError(f"responses[{index}] must be an object")
        question = _required_text(item.get("question"), f"responses[{index}].question")
        answer = _required_text(item.get("answer"), f"responses[{index}].answer")
        options = questionnaire.get(question)
        if options is None:
            raise ValueError(f"responses[{index}].question is not in the LSMS questionnaire")
        if answer not in options:
            raise ValueError(f"responses[{index}].answer must match one option exactly")
        if question in seen_questions:
            raise ValueError(f"Duplicate question at responses[{index}]")
        seen_questions.add(question)
        records.append({"question": question, "answer": answer})
    return agent_id, profile, records, data_origin.strip()


def build(source: Path, population: Path) -> Path:
    """Validate and create population/<id>; never overwrite an existing agent."""
    questionnaire = json.loads(QUESTIONNAIRE.read_text())
    agent_id, profile, records, data_origin = _validate_input(
        json.loads(source.read_text()), questionnaire)
    destination = population / agent_id
    if destination.exists():
        raise FileExistsError(f"Agent already exists: {destination}")

    from simulation_engine.gpt_structure import EMBEDDING_BACKEND, get_text_embedding

    name = profile.get("first_name", "Respondent")
    nodes, embeddings = [], {}
    for index, record in enumerate(records):
        content = f"Survey question: {record['question']}\n{name}'s response: {record['answer']}"
        nodes.append({
            "node_id": index, "node_type": "observation", "content": content,
            "importance": 80,
            "created": index + 1, "last_retrieved": index + 1,
            "pointer_id": None,
        })
        embeddings[content] = get_text_embedding(content)

    meta = {"id": agent_id, "agent_type": "survey", "ready": True,
            "embedding_backend": EMBEDDING_BACKEND, "item_count": len(records),
            "data_origin": data_origin}
    destination.mkdir(parents=True)
    memory_dir = destination / "memory_stream"
    memory_dir.mkdir()
    for name, data in (
        ("meta.json", meta), ("scratch.json", profile),
        ("memory_stream/nodes.json", nodes),
        ("memory_stream/embeddings.json", embeddings),
    ):
        (destination / name).write_text(json.dumps(data, indent=2) + "\n")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON profile and answered questions")
    parser.add_argument("--population", type=Path, default=DEFAULT_POPULATION,
                        help="directory for generated survey agent folders")
    args = parser.parse_args()
    try:
        destination = build(args.input, args.population)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"Created {destination}")


if __name__ == "__main__":
    main()
