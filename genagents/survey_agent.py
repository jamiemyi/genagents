"""Survey-style simulation grounded in stored question/answer pairs.

This mirrors the paper's survey prompt structure, but the bundled example data
is converted from an interview and profile, not an original survey instrument.
"""

import json
import re
from pathlib import Path

from simulation_engine.gpt_structure import gpt_request


def _normalized(question):
    return re.sub(r"\s+", " ", question).strip().casefold()


def _json_response(text):
    if text.startswith("GENERATION ERROR:"):
        raise RuntimeError(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError(f"Model did not return JSON: {text[:300]}")
        return json.loads(match.group())


class SurveyAgent:
    """Answer new survey items using a person's stored prior answers."""

    def __init__(self, agent_folder):
        folder = Path(agent_folder)
        self.meta = json.loads((folder / "meta.json").read_text())
        self.items = json.loads((folder / "survey_responses.json").read_text())
        if not isinstance(self.items, list) or not all(
            isinstance(item, dict) and {"id", "question", "answer"} <= item.keys()
            for item in self.items
        ):
            raise ValueError("survey_responses.json must contain question/answer items")

    def _prior_answers(self, target_questions, exclude_item_ids=(), exclude_groups=()):
        excluded_questions = {_normalized(q) for q in target_questions}
        excluded_ids = set(exclude_item_ids)
        excluded_groups = set(exclude_groups)
        return [
            {"question": item["question"], "answer": item["answer"],
             "answer_status": item.get("answer_status", "unknown")}
            for item in self.items
            if item["id"] not in excluded_ids
            and item.get("group") not in excluded_groups
            and _normalized(item["question"]) not in excluded_questions
        ]

    def recorded_resp(self, questions):
        """Return stored answers by survey item id without making an LLM call."""
        by_question = {_normalized(item["question"]): item for item in self.items}
        answers = []
        for question, options in questions.items():
            item = by_question.get(_normalized(question))
            if item is None or item["answer"] not in options:
                raise ValueError(f"No stored answer for {question}")
            answers.append(item["answer"])
        return {"responses": answers}

    def categorical_resp(self, questions, *, exclude_item_ids=(), exclude_groups=()):
        """Return answers in input order; each item must match an option exactly.

        ``exclude_item_ids`` and ``exclude_groups`` let callers withhold an
        evaluation item or an entire trait block from the input context.
        """
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions must be a non-empty question-to-options dict")
        for question, options in questions.items():
            if not isinstance(question, str) or not question.strip():
                raise ValueError("Each question must be non-empty text")
            if not isinstance(options, list) or len(options) < 2 or not all(
                isinstance(option, str) and option for option in options
            ) or len(options) != len(set(options)):
                raise ValueError("Each question needs at least two distinct text options")

        entries = list(questions.items())
        prior = self._prior_answers(
            [question for question, _ in entries], exclude_item_ids, exclude_groups
        )
        responses = []
        for start in range(0, len(entries), 4):
            block = entries[start:start + 4]
            prompt = (
                "You are simulating a synthetic survey profile based on one "
                "person's interview. Use the prior answers below to predict "
                "how this profile would answer new "
                "questions. Reply with valid JSON only: an object with a "
                "'responses' array in question order. Copy one listed option "
                "verbatim for each answer.\n\n"
                "<synthetic_survey_profile>\n"
                + json.dumps(prior, ensure_ascii=False)
                + "\n</synthetic_survey_profile>\n\n"
                "<questions>\n"
                + json.dumps(
                    [{"question": question, "options": options} for question, options in block],
                    ensure_ascii=False,
                )
                + "\n</questions>"
            )
            parsed = _json_response(gpt_request(prompt))
            batch = parsed.get("responses") if isinstance(parsed, dict) else None
            if not isinstance(batch, list) or len(batch) != len(block):
                raise ValueError("Model returned the wrong number of responses")
            for answer, (_, options) in zip(batch, block):
                if answer not in options:
                    raise ValueError(f"Model returned an answer outside the options: {answer!r}")
            responses.extend(batch)
        return {"responses": responses}
