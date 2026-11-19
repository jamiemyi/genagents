# Survey agents

`lsms_template/` mirrors an interview agent's four JSON files. Its profile and
memory are empty because no respondent answers have been supplied. It is not a
runnable respondent. The shared
[LSMS-inspired questionnaire](../../../environment/survey/lsms_like_questionnaire.json)
contains the questions and answer choices.

To build one respondent, create an input JSON file with their actual answers:

```json
{
  "id": "respondent-001",
  "profile": {"first_name": "Respondent"},
  "responses": [
    {"question": "<copy an exact LSMS question>",
     "answer": "<copy the respondent's selected option>"}
  ]
}
```

Run `python build_survey_agent.py path/to/input.json`. The builder checks
each question and answer against the shared questionnaire and writes only
`meta.json`, `scratch.json`, `memory_stream/nodes.json`, and
`memory_stream/embeddings.json`. Each answer becomes a memory and receives an
embedding. Use one input and ID per respondent. Leave evaluation questions out
of the input when testing prediction accuracy.
