import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from build_survey_agent import build
from genagents.genagents import GenerativeAgent
from genagents.survey_agent import SurveyAgent


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
                {"id": "q1", "group": "housing", "question": "Own or rent?",
                 "answer": "Rent", "options": ["Own", "Rent"]},
                {"id": "q2", "question": "Commute by transit?",
                 "answer": "Yes", "options": ["Yes", "No"]},
            ],
        }
        self.input.write_text(json.dumps(self.data))

    def test_build_creates_both_survey_and_memory_views(self):
        with patch("simulation_engine.gpt_structure.get_text_embedding",
                   side_effect=lambda text: [float(len(text)), 1.0]):
            folder = build(self.input, self.population)
        self.assertEqual(folder, self.population / "respondent-001")
        records = json.loads((folder / "survey_responses.json").read_text())
        nodes = json.loads((folder / "memory_stream/nodes.json").read_text())
        embeddings = json.loads((folder / "memory_stream/embeddings.json").read_text())
        questionnaire = json.loads((folder / "questionnaire.json").read_text())
        self.assertEqual(len(records), len(nodes))
        self.assertEqual(len(nodes), len(embeddings))
        self.assertEqual(questionnaire["Own or rent?"], ["Own", "Rent"])
        for node in nodes:
            self.assertEqual(embeddings[node["content"]],
                             [float(len(node["content"])), 1.0])
        survey_agent = SurveyAgent(folder)
        self.assertEqual(survey_agent.recorded_resp({"Own or rent?": ["Own", "Rent"]})
                         ["responses"], ["Rent"])
        memory_agent = GenerativeAgent(agent_folder=str(folder))
        self.assertEqual(len(memory_agent.memory_stream.seq_nodes), 2)

    def test_invalid_answer_and_duplicate_id_do_not_write(self):
        self.data["responses"][0]["answer"] = "Maybe"
        self.input.write_text(json.dumps(self.data))
        with self.assertRaisesRegex(ValueError, "match one option"):
            build(self.input, self.population)
        self.assertFalse(self.population.exists())

        self.data["responses"][0]["answer"] = "Rent"
        self.data["responses"][1]["id"] = "q1"
        self.input.write_text(json.dumps(self.data))
        with self.assertRaisesRegex(ValueError, "Duplicate question"):
            build(self.input, self.population)
        self.assertFalse(self.population.exists())

    def test_existing_agent_is_not_overwritten(self):
        folder = self.population / "respondent-001"
        folder.mkdir(parents=True)
        marker = folder / "keep.txt"
        marker.write_text("original")
        with self.assertRaises(FileExistsError):
            build(self.input, self.population)
        self.assertEqual(marker.read_text(), "original")


if __name__ == "__main__":
    unittest.main()
