import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.survey import predict, score


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class SurveyEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.questions = self.root / "questions.json"
        self.questions.write_text(json.dumps([
            {"id": "q1", "text": "Would you choose A?", "options": ["Yes", "No"]}
        ]))

    def test_weighted_tvd_and_option_shares(self):
        observed = self.root / "observed.csv"
        predicted = self.root / "predicted.csv"
        write_csv(observed,
                  ["participant_id", "question_id", "cohort", "answer", "weight"], [
                      {"participant_id": "p1", "question_id": "q1", "cohort": "c",
                       "answer": "Yes", "weight": 3},
                      {"participant_id": "p2", "question_id": "q1", "cohort": "c",
                       "answer": "No", "weight": 1},
                  ])
        write_csv(predicted,
                  ["agent_id", "question_id", "cohort", "answer", "weight"], [
                      {"agent_id": "a1", "question_id": "q1", "cohort": "c",
                       "answer": "Yes", "weight": 1},
                      {"agent_id": "a2", "question_id": "q1", "cohort": "c",
                       "answer": "No", "weight": 3},
                  ])
        report = score(self.questions, observed, predicted, self.root / "output")
        self.assertEqual(report["mean_tvd"], 0.5)
        with (self.root / "output/summary.csv").open(newline="") as stream:
            summary = next(csv.DictReader(stream))
        self.assertEqual(float(summary["predicted_max_share"]), 0.75)
        self.assertEqual(float(summary["predicted_top2_gap"]), 0.5)
        with (self.root / "output/options.csv").open(newline="") as stream:
            options = {row["option"]: row for row in csv.DictReader(stream)}
        self.assertEqual(float(options["Yes"]["observed_share"]), 0.75)
        self.assertEqual(float(options["Yes"]["predicted_share"]), 0.25)

    def test_rejects_missing_question_cohort_pairs(self):
        observed = self.root / "observed.csv"
        predicted = self.root / "predicted.csv"
        write_csv(observed,
                  ["participant_id", "question_id", "cohort", "answer"], [
                      {"participant_id": "p1", "question_id": "q1", "cohort": "one",
                       "answer": "Yes"}
                  ])
        write_csv(predicted,
                  ["agent_id", "question_id", "cohort", "answer"], [
                      {"agent_id": "a1", "question_id": "q1", "cohort": "two",
                       "answer": "Yes"}
                  ])
        with self.assertRaisesRegex(ValueError, "Question/cohort mismatch"):
            score(self.questions, observed, predicted, self.root / "output")

    def test_rejects_answer_outside_manifest_options(self):
        observed = self.root / "observed.csv"
        predicted = self.root / "predicted.csv"
        write_csv(observed,
                  ["participant_id", "question_id", "cohort", "answer"], [
                      {"participant_id": "p1", "question_id": "q1", "cohort": "c",
                       "answer": "Maybe"}
                  ])
        write_csv(predicted,
                  ["agent_id", "question_id", "cohort", "answer"], [
                      {"agent_id": "a1", "question_id": "q1", "cohort": "c",
                       "answer": "Yes"}
                  ])
        with self.assertRaisesRegex(ValueError, "Answer outside options"):
            score(self.questions, observed, predicted, self.root / "output")

    def test_predict_rejects_evaluation_question_in_memory(self):
        folder = self.root / "agent"
        (folder / "memory_stream").mkdir(parents=True)
        (folder / "meta.json").write_text(json.dumps({"id": "a1", "ready": True}))
        (folder / "memory_stream/nodes.json").write_text(json.dumps([
            {"content": "Survey question: Would you choose A?\nResponse: Yes"}
        ]))
        agents = self.root / "agents.csv"
        write_csv(agents, ["agent_id", "folder", "cohort"], [
            {"agent_id": "a1", "folder": "agent", "cohort": "c"}
        ])
        with self.assertRaisesRegex(ValueError, "appears in agent memory"):
            predict(self.questions, agents, self.root / "predicted.csv")

    def test_predict_uses_memory_agent_for_held_out_question(self):
        folder = self.root / "agent"
        (folder / "memory_stream").mkdir(parents=True)
        (folder / "meta.json").write_text(json.dumps({"id": "a1", "ready": True}))
        (folder / "memory_stream/nodes.json").write_text(json.dumps([
            {"content": "Survey question: What age group are you in?\nResponse: 25-34"}
        ]))
        agents = self.root / "agents.csv"
        write_csv(agents, ["agent_id", "folder", "cohort"], [
            {"agent_id": "a1", "folder": "agent", "cohort": "c"}
        ])
        with patch("genagents.genagents.GenerativeAgent") as agent_class:
            agent_class.return_value.categorical_resp.return_value = {"responses": ["Yes"]}
            self.assertEqual(predict(self.questions, agents, self.root / "predicted.csv"), 1)
            agent_class.return_value.categorical_resp.assert_called_once_with(
                {"Would you choose A?": ["Yes", "No"]})


if __name__ == "__main__":
    unittest.main()
