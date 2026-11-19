import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from build_survey_agent import build
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
        self.input.write_text(json.dumps(self.data))
        with self.assertRaisesRegex(ValueError, "match one option"):
            build(self.input, self.population)
        self.assertFalse(self.population.exists())

        self.data["responses"][0]["answer"] = "25-34"
        self.data["responses"][1]["question"] = "What age group are you in?"
        self.data["responses"][1]["answer"] = "25-34"
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
