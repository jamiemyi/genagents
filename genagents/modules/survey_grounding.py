"""Answer survey questions from recorded responses and nearby survey evidence."""

import json

from genagents.modules.memory_stream import extract_relevance
from simulation_engine.gpt_structure import chat_safe_generate
from simulation_engine.llm_json_parser import extract_first_json_dict
from simulation_engine.settings import LLM_PROMPT_DIR, LLM_VERS


def _normalized(text):
  return " ".join(text.split()).casefold()


def _survey_records(agent):
  records = []
  for node in agent.memory_stream.seq_nodes:
    if not node.content.startswith("Survey question: "):
      continue
    question, separator, response = node.content.partition("\n")
    if not separator or ": " not in response:
      continue
    records.append((question.removeprefix("Survey question: "),
                    response.split(": ", 1)[1], node))
  return records


def _related_records(agent, records, question, count=6):
  if not records:
    return []
  nodes = [node for _, _, node in records]
  scores = extract_relevance(nodes, agent.memory_stream.embeddings, question)
  ranked = sorted(records, key=lambda record: scores[record[2].node_id], reverse=True)
  return [{"question": stored_question, "answer": answer}
          for stored_question, answer, _ in ranked[:count]]


def _predict(agent, question, options, evidence):
  prompt_file = f"{LLM_PROMPT_DIR}/generative_agent/interaction/categorical_resp/survey_v1.txt"
  prompt_input = [json.dumps(agent.scratch, ensure_ascii=False),
                  json.dumps(evidence, ensure_ascii=False),
                  json.dumps({"question": question, "options": options}, ensure_ascii=False)]

  def clean_up(response, prompt=""):
    parsed = extract_first_json_dict(response)
    if not isinstance(parsed, dict) or parsed.get("response") not in options:
      raise ValueError(f"Survey prediction must select a listed option for {question!r}")
    return parsed["response"], str(parsed.get("reasoning", ""))

  result, _, _, _ = chat_safe_generate(prompt_input, prompt_file, LLM_VERS,
                                       1, None, clean_up)
  return result


def categorical_survey_resp(agent, questions):
  """Use recorded answers directly; predict unseen answers one question at a time."""
  records = _survey_records(agent)
  answers, reasonings = [], []
  for question, options in questions.items():
    matching = [(answer, node) for stored_question, answer, node in records
                if _normalized(stored_question) == _normalized(question)]
    if matching and matching[0][0] in options:
      answers.append(matching[0][0])
      reasonings.append("Recorded survey answer for this question.")
      continue
    evidence = _related_records(agent, records, question)
    answer, reasoning = _predict(agent, question, options, evidence)
    answers.append(answer)
    reasonings.append(reasoning)
  return {"responses": answers, "reasonings": reasonings}
