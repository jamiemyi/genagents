# Survey evaluation

This workflow compares simulated categorical answers with observed survey
answers. It measures **total variation distance (TVD)** between answer
distributions for each question and cohort:

`TVD = 0.5 × sum(abs(observed_share - predicted_share))` over the listed options.

Zero means the distributions match; one means they have no overlap. The scorer
also writes each option's observed and predicted share. It uses Python's
standard library and does not call an LLM.

## Prepare three input files

Place private survey inputs in `evaluation/data/` (ignored by Git). You can use
other paths. These examples show the format, not real observations.

**`questions.json`** is a list of held-out questions with stable IDs and exact
answer options:

```json
[
  {
    "id": "q1",
    "text": "Would you use public transit next week?",
    "options": ["Yes", "No"]
  }
]
```

**`agents.csv`** lists the survey agents to simulate. `folder` is relative to
the CSV's directory, or absolute. `weight` is optional and defaults to 1.
Each agent must have `meta.json` and `survey_responses.json` in its folder.

```csv
agent_id,folder,cohort,weight
01fd7d2a-0357-4c1b-9f3e-8eade2d537ae,../../agent_bank/populations/survey_agents/01fd7d2a-0357-4c1b-9f3e-8eade2d537ae,demo,1
```

**`observed.csv`** has one observed answer per participant and question. Use
the same question IDs, cohort labels, and option strings as `questions.json`.
`weight` is optional and defaults to 1.

```csv
participant_id,question_id,cohort,answer,weight
person-1,q1,demo,Yes,1
```

The observed and simulated samples should represent the same population,
cohort, time period, question wording, and options. Use survey sampling weights
when available. A single agent, including this repo's synthetic demo agent,
does not establish population accuracy.

## Predict and score

Configure `simulation_engine/settings.py` for your LLM backend before running
`predict` (see the repo's main README). `score` needs no LLM. From the repo root:

```bash
python -m evaluation.survey predict \
  --questions evaluation/data/questions.json \
  --agents evaluation/data/agents.csv \
  --output evaluation/output/predicted.csv

python -m evaluation.survey score \
  --questions evaluation/data/questions.json \
  --observed evaluation/data/observed.csv \
  --predicted evaluation/output/predicted.csv \
  --output-dir evaluation/output/score
```

Prediction uses `SurveyAgent.categorical_resp`, never its recorded-answer
lookup or the separate `Survey.survey` memory-based runner. It withholds **all**
evaluation question IDs and matching question text
from every agent's prior-answer prompt. Add `--exclude-group GROUP` to the
predict command to withhold an entire response group as well; repeat the flag
for multiple groups. Build agents from training-only survey answers when
possible, and check for paraphrases or other fields that reveal the held-out
answers. A response already present in `survey_responses.json` is not an
independent test observation merely because it was copied into `observed.csv`.
Prediction sends the remaining prior answers to the configured LLM backend;
check that backend before using private participant data.

`predicted.csv` has `agent_id,question_id,cohort,answer,weight`. The scorer
rejects unknown options, duplicate respondent-question pairs, non-positive
weights, and question/cohort pairs missing from either input. It writes:

- `summary.csv`: TVD, sample sizes, total weights, and basic prediction features
  (option count, question length, entropy, maximum share, top-two gap) for each
  question/cohort. TVD is the observed-error label for future confidence work.
- `options.csv`: observed share, predicted share, and absolute gap by option.
- `metrics.json`: the unweighted mean TVD across question/cohort pairs.

This measures **observed error**. A confidence model that predicts error for
questions without observations would need many labeled question/cohort pairs,
separate training and held-out evaluation splits, and calibration checks. The
current workflow does not assign confidence labels.

Exclude missing answers before scoring, or list a nonresponse option explicitly
in `questions.json` if it is part of the outcome you want to measure.
