import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.data_pipeline import build_agents, predict_and_score, prepare
from evaluation.survey import load_agents


class DataPipelineTest(unittest.TestCase):
    def test_prepare_keeps_held_out_answers_out_of_agents(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data_dir = root / "csv"
            data_dir.mkdir()
            with (data_dir / "people.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerows([
                    ["hhid", "indiv", "sector", "train", "heldout"],
                    ["10010", "1", "urban", "1.0", "2.0"],
                    ["10010", "2", "urban", "2.0", ""],
                ])
            config = root / "config.json"
            config.write_text(json.dumps({
                "source": "csv/people.csv",
                "id_columns": ["hhid", "indiv"], "cohort_column": "sector",
                "questions": [
                    {"id": "train", "column": "train", "text": "Training question?",
                     "role": "train", "options": ["Yes", "No"],
                     "value_map": {"1": "Yes", "2": "No"}},
                    {"id": "heldout", "column": "heldout", "text": "Held-out question?",
                     "role": "test", "options": ["Yes", "No"],
                     "value_map": {"1": "Yes", "2": "No"}},
                ],
            }))
            output = root / "run"
            self.assertEqual(prepare(config, output), 1)
            agent_input = json.loads(next((output / "inputs").glob("*.json")).read_text())
            self.assertEqual(agent_input["responses"], [{
                "question": "Training question?", "answer": "Yes", "options": ["Yes", "No"]}])
            self.assertNotIn("Held-out question?", json.dumps(agent_input))
            with (output / "observed.csv").open(newline="") as stream:
                observed = list(csv.DictReader(stream))
            self.assertEqual(observed[0]["answer"], "No")
            self.assertEqual(load_agents(output / "agents.csv")[0]["folder"],
                             (output / "agents" / agent_input["id"]).resolve())
            self.assertEqual(len(list((output / "agents").glob("*"))), 0)
            with patch("simulation_engine.gpt_structure.get_text_embedding",
                       return_value=[1.0, 0.0]):
                self.assertEqual(build_agents(output), 1)
            with patch("genagents.genagents.GenerativeAgent.categorical_resp",
                       return_value={"responses": ["No"]}):
                report = predict_and_score(output)
            self.assertEqual(report["mean_tvd"], 0)
            with self.assertRaises(FileExistsError):
                prepare(config, output)
            source = data_dir / "people.csv"
            source.write_text(source.read_text().replace("1.0,2.0", "7.0,2.0"))
            with self.assertRaisesRegex(ValueError, "Unknown answer"):
                prepare(config, root / "bad_run")
            self.assertFalse((root / "bad_run").exists())

    def test_jsonl_uses_same_mapping_and_seeded_sample(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "records.jsonl"
            source.write_text("\n".join(json.dumps({"person": str(i), "train": "Yes",
                                                     "test": "No"}) for i in range(5)))
            config = root / "config.json"
            config.write_text(json.dumps({
                "source": "records.jsonl", "id_columns": ["person"],
                "sample_size": 2, "sample_seed": 5,
                "questions": [
                    {"id": "train", "column": "train", "text": "Training question?",
                     "role": "train", "options": ["Yes", "No"]},
                    {"id": "test", "column": "test", "text": "Held-out question?",
                     "role": "test", "options": ["Yes", "No"]},
                ],
            }))
            prepare(config, root / "run1")
            source.write_text("\n".join(reversed(source.read_text().splitlines())))
            prepare(config, root / "run2")
            with (root / "run1/source_ids.csv").open(newline="") as stream:
                first = list(csv.DictReader(stream))
            with (root / "run2/source_ids.csv").open(newline="") as stream:
                second = list(csv.DictReader(stream))
            self.assertEqual(first, second)
            self.assertEqual(len(first), 2)


if __name__ == "__main__":
    unittest.main()
