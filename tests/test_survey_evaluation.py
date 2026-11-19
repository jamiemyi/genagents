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

    def test_predict_withholds_all_evaluation_items_across_batches(self):
        items = [
            {"id": f"q{i}", "question": f"Target question {i}?", "answer": "Yes",
             "group": "target"}
            for i in range(5)
        ] + [{"id": "context", "question": "Context question?", "answer": "No",
              "group": "context"}]
        self.questions.write_text(json.dumps([
            {"id": f"q{i}", "text": f"Target question {i}?", "options": ["Yes", "No"]}
            for i in range(5)
        ]))
        folder = self.root / "agent"
        folder.mkdir()
        (folder / "meta.json").write_text(json.dumps({"id": "a1"}))
        (folder / "survey_responses.json").write_text(json.dumps(items))
        agents = self.root / "agents.csv"
        write_csv(agents, ["agent_id", "folder", "cohort"], [
            {"agent_id": "a1", "folder": "agent", "cohort": "c"}
        ])

        prompts = []

        def fake_request(prompt):
            prompts.append(prompt)
            profile = prompt.split("<synthetic_survey_profile>\n", 1)[1].split(
                "\n</synthetic_survey_profile>", 1
            )[0]
            prior = json.loads(profile)
            self.assertEqual([item["question"] for item in prior], ["Context question?"])
            targets = json.loads(prompt.split("<questions>\n", 1)[1].split(
                "\n</questions>", 1
            )[0])
            return json.dumps({"responses": ["Yes"] * len(targets)})

        with patch("genagents.survey_agent.gpt_request", side_effect=fake_request):
            count = predict(self.questions, agents, self.root / "predicted.csv")
        self.assertEqual(count, 5)
        self.assertEqual(len(prompts), 2)
        with (self.root / "predicted.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 5)
        self.assertEqual({row["answer"] for row in rows}, {"Yes"})


if __name__ == "__main__":
    unittest.main()
