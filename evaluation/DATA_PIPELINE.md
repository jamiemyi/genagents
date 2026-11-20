# Survey agent data pipeline

`data_pipeline.py` prepares, builds, and evaluates survey agents from **any
tabular CSV, JSON list, or JSONL file with one row per respondent**. For data
spread across multiple files or record units, join or aggregate it into one
respondent row first. The pipeline never guesses question meanings or code
labels: a mapping JSON names the ID fields, questions, answer options, and
which answers to hold out.

Example mapping (paths are relative to the mapping file):

```json
{
  "source": "../private/responses.csv",
  "id_columns": ["respondent_id"],
  "profile_columns": {"age_group": "age_band"},
  "cohort_column": "region",
  "weight_column": "sample_weight",
  "sample_size": 100,
  "sample_seed": 42,
  "questions": [
    {"id": "q1", "column": "q1_code", "text": "Do you own a phone?",
     "role": "train", "options": ["Yes", "No"],
     "value_map": {"1": "Yes", "2": "No"}},
    {"id": "q2", "column": "q2_code", "text": "Do you use mobile banking?",
     "role": "test", "options": ["Yes", "No"],
     "value_map": {"1": "Yes", "2": "No"}, "missing_values": ["99"]}
  ]
}
```

The source can contain answer labels directly, in which case `value_map` is
optional. Blank values are missing; `missing_values` can exclude additional
codes. A respondent must have every test answer and at least one training
answer. Held-out columns cannot enter the profile, cohort, ID, or weight.
`sample_size` is optional; if set, a stable hash and `sample_seed` select the
same eligible people regardless of source row order. Set `sample_size` to
`null` or omit it to include all eligible records. `weight_column` is optional
and defaults to 1.

Run the stages separately from the repository root:

```bash
python -m evaluation.data_pipeline prepare \
  --config path/to/mapping.json --output evaluation/output/run1
python -m evaluation.data_pipeline build --output evaluation/output/run1
python -m evaluation.data_pipeline evaluate --output evaluation/output/run1
```

`prepare` writes respondent JSON inputs, `questions.json`, `agents.csv`,
`observed.csv`, `source_ids.csv`, and a manifest. It does not build agents or
call a model. `build` creates agents using only training answers. `evaluate`
predicts held-out answers with the configured model and writes TVD reports.
The `all` stage runs all three for unattended jobs. Use a fresh output path
for each run; preparation refuses to overwrite an existing run. Store outputs
under the ignored `evaluation/output/` directory when they contain respondent
data.

[nigeria_wave5.example.json](nigeria_wave5.example.json) shows one mapping
for the Nigeria household roster. It is an example of this general interface;
the pipeline itself has no Nigeria-specific fields. Roster answers may be
proxy reported, so treat the record unit carefully when interpreting results.
