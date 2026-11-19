"""Create one survey agent from a JSON profile and answered questionnaire.

See examples/survey_agent.example.json for the input format. The generated
agent works with SurveyAgent and with the memory-based Survey environment.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


DEFAULT_POPULATION = Path(__file__).resolve().parent / "agent_bank/populations/survey_agents"


def _required_text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


def _validate_input(data):
    if not isinstance(data, dict):
        raise ValueError("Input must be a JSON object")
    agent_id = _required_text(data.get("id"), "id")
    if agent_id in (".", "..") or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", agent_id):
        raise ValueError("id may contain only letters, digits, dots, underscores, and hyphens")
    profile = data.get("profile", {})
    if not isinstance(profile, dict):
        raise ValueError("profile must be a JSON object")
    responses = data.get("responses")
    if not isinstance(responses, list) or not responses:
        raise ValueError("responses must be a non-empty list")

    records, questionnaire, seen_ids, seen_questions = [], {}, set(), set()
    for index, item in enumerate(responses, 1):
        if not isinstance(item, dict):
            raise ValueError(f"responses[{index}] must be an object")
        item_id = _required_text(item.get("id"), f"responses[{index}].id")
        question = _required_text(item.get("question"), f"responses[{index}].question")
        answer = _required_text(item.get("answer"), f"responses[{index}].answer")
        options = item.get("options")
        if (not isinstance(options, list) or len(options) < 2
                or any(not isinstance(option, str) or not option.strip()
                       for option in options)
                or len(set(options)) != len(options)):
            raise ValueError(f"responses[{index}].options needs distinct non-empty choices")
        if answer not in options:
            raise ValueError(f"responses[{index}].answer must match one option exactly")
        normalized_question = " ".join(question.split()).casefold()
        if item_id in seen_ids or normalized_question in seen_questions:
            raise ValueError(f"Duplicate question id or text at responses[{index}]")
        seen_ids.add(item_id)
        seen_questions.add(normalized_question)

        group = item.get("group", "")
        if not isinstance(group, str):
            raise ValueError(f"responses[{index}].group must be text")
        status = _required_text(item.get("answer_status", "reported"),
                                f"responses[{index}].answer_status")
        refs = item.get("source_refs", [])
        if not isinstance(refs, list) or any(not isinstance(ref, str) for ref in refs):
            raise ValueError(f"responses[{index}].source_refs must be a list of text")
        records.append({
            "id": item_id, "group": group.strip(), "question": question,
            "answer": answer, "options": options, "answer_status": status,
            "source_refs": refs,
        })
        questionnaire[question] = options
    return agent_id, profile, records, questionnaire


def build(source: Path, population: Path) -> Path:
    """Validate and create population/<id>; never overwrite an existing agent."""
    agent_id, profile, records, questionnaire = _validate_input(
        json.loads(source.read_text())
    )
    destination = population / agent_id
    if destination.exists():
        raise FileExistsError(f"Agent already exists: {destination}")

    from simulation_engine.gpt_structure import EMBEDDING_BACKEND, get_text_embedding

    name = profile.get("first_name", "Respondent")
    nodes, embeddings = [], {}
    for index, record in enumerate(records):
        group = f" ({record['group']})" if record["group"] else ""
        content = (
            f"Survey question{group}: {record['question']}\n"
            f"{name}'s response ({record['answer_status']}): {record['answer']}"
        )
        nodes.append({
            "node_id": index, "node_type": "observation", "content": content,
            "importance": 80 if record["answer_status"] == "reported" else 60,
            "created": index + 1, "last_retrieved": index + 1,
            "pointer_id": None,
        })
        embeddings[content] = get_text_embedding(content)

    counts = dict(Counter(record["answer_status"] for record in records))
    meta = {
        "id": agent_id, "agent_type": "survey", "questionnaire": "questionnaire.json",
        "embedding_backend": EMBEDDING_BACKEND, "item_counts": counts,
    }
    destination.mkdir(parents=True)
    memory_dir = destination / "memory_stream"
    memory_dir.mkdir()
    for name, data in (
        ("meta.json", meta), ("scratch.json", profile),
        ("survey_responses.json", records), ("questionnaire.json", questionnaire),
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
