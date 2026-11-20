import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from build_survey_agent import build, main
from genagents.genagents import GenerativeAgent


class SurveyBuilderTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.population = self.root / "survey_agents"
        self.input = self.root / "input.json"
        self.data = {
            "id": "respondent-001",
            "profile": {"first_name": "Avery", "last_name": "Example"},
            "responses": [
                {"question": "What age group are you in?", "answer": "25-34"},
                {"question": "What sex is recorded for you?", "answer": "Female"},
            ],
        }
        self.input.write_text(json.dumps(self.data))

    def test_build_creates_interview_format_files(self):
        with patch("simulation_engine.gpt_structure.get_text_embedding",
                   side_effect=lambda text: [float(len(text)), 1.0]):
            folder = build(self.input, self.population)
        self.assertEqual(folder, self.population / "respondent-001")
        nodes = json.loads((folder / "memory_stream/nodes.json").read_text())
        embeddings = json.loads((folder / "memory_stream/embeddings.json").read_text())
        self.assertEqual({str(p.relative_to(folder)) for p in folder.rglob("*.json")},
                         {"meta.json", "scratch.json", "memory_stream/nodes.json",
                          "memory_stream/embeddings.json"})
        self.assertEqual(len(nodes), 2)
        self.assertEqual(len(nodes), len(embeddings))
        for node in nodes:
            self.assertEqual(embeddings[node["content"]],
                             [float(len(node["content"])), 1.0])
        memory_agent = GenerativeAgent(agent_folder=str(folder))
        self.assertEqual(len(memory_agent.memory_stream.seq_nodes), 2)

    def test_invalid_answer_and_duplicate_id_do_not_write(self):
        self.data["responses"][0]["answer"] = "Maybe"
        self.data["responses"][0]["options"] = ["18-24", "25-34"]
        self.input.write_text(json.dumps(self.data))
        with self.assertRaisesRegex(ValueError, "match one option"):
            build(self.input, self.population)
        self.assertFalse(self.population.exists())

        self.data["responses"][0]["answer"] = "25-34"
        del self.data["responses"][0]["options"]
        self.data["responses"][1]["question"] = "What age group are you in?"
        self.data["responses"][1]["answer"] = "25-34"
        self.input.write_text(json.dumps(self.data))
        with self.assertRaisesRegex(ValueError, "Duplicate question"):
            build(self.input, self.population)
        self.assertFalse(self.population.exists())

    def test_arbitrary_questions_and_optional_questionnaire(self):
        self.data["responses"] = [
            {"question": "What did you enjoy this week?", "answer": "Visiting my sister."},
            {"question": "Would you go again?", "answer": "Yes", "options": ["Yes", "No"]},
        ]
        self.input.write_text(json.dumps(self.data))
        questionnaire = self.root / "questionnaire.json"
        questionnaire.write_text(json.dumps({"Would you go again?": ["Yes", "No"]}))
        with self.assertRaisesRegex(ValueError, "not in the questionnaire"):
            build(self.input, self.population, questionnaire)
        self.assertFalse(self.population.exists())

        with patch("simulation_engine.gpt_structure.get_text_embedding",
                   return_value=[1.0, 0.0]), patch(
                       "sys.argv", ["build_survey_agent.py", str(self.input),
                                    "--population", str(self.population)]):
            main()
        nodes = json.loads((self.population / "respondent-001" /
                            "memory_stream/nodes.json").read_text())
        self.assertIn("Visiting my sister.", nodes[0]["content"])

    def test_existing_agent_is_not_overwritten(self):
        folder = self.population / "respondent-001"
        folder.mkdir(parents=True)
        marker = folder / "keep.txt"
        marker.write_text("original")
        with self.assertRaises(FileExistsError):
            build(self.input, self.population)
        self.assertEqual(marker.read_text(), "original")

    def test_survey_answers_use_exact_record_then_related_records(self):
        with patch("simulation_engine.gpt_structure.get_text_embedding",
                   side_effect=lambda text: [float(len(text)), 1.0]):
            folder = build(self.input, self.population)
        agent = GenerativeAgent(agent_folder=str(folder))
        self.assertEqual(agent.agent_type, "survey")
        model_calls = []

        def fake_generate(prompt_input, prompt_file, model, repeat, fail_safe, clean_up):
            model_calls.append((prompt_input, prompt_file))
            result = clean_up('{"response": "Yes", "reasoning": "Related evidence."}')
            return result, "", prompt_input, fail_safe

        with patch("genagents.modules.survey_grounding.extract_relevance",
                   return_value={0: 0.9, 1: 0.1}), patch(
                       "genagents.modules.survey_grounding.chat_safe_generate",
                       side_effect=fake_generate):
            output = agent.categorical_resp({
                "What age group are you in?": ["18-24", "25-34"],
                "Are you currently enrolled in formal education?": ["Yes", "No"],
            })

        self.assertEqual(output["responses"], ["25-34", "Yes"])
        self.assertEqual(len(model_calls), 1)
        prompt_input, prompt_file = model_calls[0]
        self.assertTrue(prompt_file.endswith("survey_v1.txt"))
        self.assertEqual(json.loads(prompt_input[1])[0], {
            "question": "What age group are you in?", "answer": "25-34"})
        self.assertEqual(json.loads(prompt_input[2])["question"],
                         "Are you currently enrolled in formal education?")


if __name__ == "__main__":
    unittest.main()
