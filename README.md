# genagents: Generative Agent Simulations of 1,000 People

![Cover Image](static_dir/cover3.png)

## Overview

This project introduces a novel agent architecture that simulates the attitudes and behaviors of real individuals by applying large language models (LLMs) to qualitative interviews about their lives. These agents replicate participants' responses on various social science measures, providing a foundation for new tools to investigate individual and collective behavior.

In the coming months, the authors at Stanford University plan to make available generative agents of 1,000 people—based on 2,000 hours of interviews—via a restricted API for research purposes. To support research while protecting participant privacy, this restricted access will offer a two-pronged system:

1. **Open Access to Aggregated Responses on Fixed Tasks**: Researchers can access aggregated data to analyze general trends and patterns.
2. **Restricted Access to Individual Responses on Open Tasks**: Researchers can request access to individual agents' responses for more detailed studies, subject to a review process ensuring ethical considerations are met.

This codebase offers two main components:

1. **Codebase for Creating and Interacting with Generative Agents**: Tools to build new agents based on your own data and interact with them. Query agents with surveys, experiments, and other stimuli to study their responses.
2. **Demographic Agent Banks**: A bank of over 3,000 agents created using demographic information from the General Social Survey (GSS) as a starting point to explore the codebase. *Note: The names and addresses are fictional.*

Additionally, to provide users with a sense of the interview-based agents, we offer an example generative agent of one of the authors, created using the same interview protocol used in our paper.

## Table of Contents

- [Installation](#installation)
  - [Requirements](#requirements)
  - [Dependencies](#dependencies)
  - [Configuration](#configuration)
  - [Using a subscription instead of an API key](#using-a-subscription-instead-of-an-api-key)
    - [Quick start](#quick-start)
    - [Using it from T3 Code](#using-it-from-t3-code)
- [Repository Structure](#repository-structure)
- [Usage](#usage)
  - [Creating a Generative Agent](#creating-a-generative-agent)
  - [Interacting with Agents](#interacting-with-agents)
    - [Categorical Responses](#categorical-responses)
    - [Numerical Responses](#numerical-responses)
    - [Open-Ended Questions](#open-ended-questions)
  - [Memory and Reflection](#memory-and-reflection)
    - [Adding Memories](#adding-memories)
    - [Reflection](#reflection)
  - [Saving and Loading Agents](#saving-and-loading-agents)
- [Interview Agent](#sample-agent)
- [Agent Bank Access](#agent-bank-access)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)
- [Contact](#contact)

## Installation

### Requirements

- Python 3.7 or higher
- An OpenAI API key with access to GPT-4 or GPT-3.5-turbo models, or a Claude or ChatGPT subscription (see [Using a subscription instead of an API key](#using-a-subscription-instead-of-an-api-key))

### Dependencies

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

### Configuration

Create a `settings.py` file in the `simulation_engine` folder (where `example-settings.py` is located). Place the following content in `settings.py`:

```python
from pathlib import Path

OPENAI_API_KEY = "YOUR_API_KEY"
KEY_OWNER = "YOUR_NAME"

DEBUG = False

MAX_CHUNK_SIZE = 4

LLM_VERS = "gpt-4o-mini"

BASE_DIR = f"{Path(__file__).resolve().parent.parent}"

POPULATIONS_DIR = f"{BASE_DIR}/agent_bank/populations"
LLM_PROMPT_DIR = f"{BASE_DIR}/simulation_engine/prompt_template"
```

Replace `"YOUR_API_KEY"` with your actual OpenAI API key and `"YOUR_NAME"` with your name.

### Using a subscription instead of an API key

You can run the agents through the Claude Code CLI (`claude`, Claude subscription) or the Codex CLI (`codex`, ChatGPT subscription) instead of the OpenAI API. Install the CLI you want, log in once (`claude`, or `codex login`), then add these lines to `settings.py` (see `example-settings.py` for all options):

```python
LLM_BACKEND = "claude_cli"         # or "codex_cli"; "openai" is the original API setup
CLAUDE_MODEL = "sonnet"            # optional; None uses the CLI's default model
CODEX_MODEL = None                 # optional
CLI_TIMEOUT = 300                  # seconds per call
EMBEDDING_BACKEND = "local"        # sentence-transformers, no API key needed
LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

With both settings off `"openai"`, `OPENAI_API_KEY` is not used and `openai` is not imported. Each LLM call runs the CLI headless with tools, MCP servers and file access turned off, so it only answers the prompt. `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `OPENAI_API_KEY` and `CODEX_API_KEY` are removed from the CLI's environment so it bills your subscription, not the API.

Memory retrieval compares stored embeddings with new ones, so they must come from the same model. After switching `EMBEDDING_BACKEND`, re-embed any agent that has memories (the GSS agents have none; the interview agent in this branch is already re-embedded for `"local"`):

```bash
python reembed_agents.py agent_bank/populations/interview_agent
```

The previous file is kept as `embeddings.json.bak`. To go back to `"openai"`, restore the backup or run the script again with that setting. (For the interview agent, the original OpenAI vectors are also in git history: `git checkout main -- agent_bank/populations/interview_agent`.)

Things to know:

- **Speed.** Every call starts a new CLI process, which adds a second or two. In our tests a call took about 3 to 9 seconds with `claude_cli` (Sonnet) and 4 to 17 seconds with `codex_cli`.
- **Rate limits.** Subscriptions have usage limits that reset over hours. Surveying hundreds of agents can hit them, and `Survey`/`Interview` start 50 threads by default, which means 50 CLI processes at once. Use a smaller `num_threads` and expect to spread large runs out. Calls that fail return `GENERATION ERROR: ...` as with the API.
- **Results differ from the paper.** The paper's results come from GPT-4o through the API. Other models give different answers, and the CLIs don't expose `temperature` or `max_tokens` (both are ignored), so don't expect to reproduce the published numbers.

#### Quick start

Use Python 3.12 or earlier (the pinned `numpy==1.26.2` doesn't install on newer versions). From the repo root:

```bash
python3.12 -m venv .venv            # or: uv venv -p 3.12 .venv
source .venv/bin/activate
pip install -r requirements.txt
cp simulation_engine/example-settings.py simulation_engine/settings.py
```

In `simulation_engine/settings.py`, set `LLM_BACKEND = "claude_cli"` (or `"codex_cli"`), `CLAUDE_MODEL = "sonnet"` and `EMBEDDING_BACKEND = "local"`. Then save this as `try_agent.py` in the repo root and run `python try_agent.py`:

```python
from genagents.genagents import GenerativeAgent

agent = GenerativeAgent(agent_folder="agent_bank/populations/interview_agent/01fd7d2a-0357-4c1b-9f3e-8eade2d537ae")

# Survey: categorical questions
survey = agent.categorical_resp({
  "Generally speaking, would you say most people can be trusted?": ["Can be trusted", "Can't be too careful", "Depends"],
  "How happy are you these days?": ["Very happy", "Pretty happy", "Not too happy"],
})
print(survey["responses"])

# Interview: open-ended answer
print(agent.utterance([["Interviewer", "Tell me about your favorite hobby."]]))
```

To use the code from another project folder instead, install this repo there in editable mode (`pip install -e "/path/to/genagents[local]"`). The `settings.py` in this repo still applies.

The first run downloads the small embedding model (about 90 MB). After that, this example takes about 30 seconds and prints something like `['Can be trusted', 'Pretty happy']` followed by a few sentences about drawing and painting.

#### Survey-style agent from the sample interview

Following the [paper's survey prompt structure](https://arxiv.org/pdf/2411.10109),
the `survey_agents` population contains a separate representation of the same
sample person as survey question/answer records. The example uses an original
55-item, [LSMS-inspired](https://documents1.worldbank.org/curated/en/452741468778781879/pdf/multi-page.pdf) household questionnaire covering demographics, education,
work, migration, housing, consumption, health, finances, and assets. Answers
are marked `reported` (stated in the source), `inferred` (a close implication),
or `imagined` (a plausible answer supplied for this example). **This is a
synthetic reconstruction, not a survey administered to the person or an official
LSMS instrument.** The paper's survey-only results do not apply to it.

Rebuild the files from the interview agent with:

```bash
python build_survey_agent.py \
  agent_bank/populations/interview_agent/01fd7d2a-0357-4c1b-9f3e-8eade2d537ae \
  agent_bank/populations/survey_agents/01fd7d2a-0357-4c1b-9f3e-8eade2d537ae
```

Run the first four LSMS-style questions with `python run_survey.py`. Use
`python run_survey.py --all` to run the full questionnaire. Each batch is saved
under `survey_run/` as `responses.csv`. These commands return stored answers;
`--mode predict` asks the LLM to predict instead.

Survey agents read `survey_responses.json` directly. Their `memory_stream/nodes.json`
is `[]` and `memory_stream/embeddings.json` is `{}`; they do not use vector
retrieval. Rebuilding the survey agent preserves those empty files. Running a
survey writes answers to `survey_run/responses.csv`; it does not add those
answers to the agent's stored profile automatically.

`Survey.survey` takes the same question-to-options dictionary as
`GenerativeAgent.categorical_resp`. Load the bundled questions and return the
stored answers:

```python
import json
from environment.survey.survey import Survey

survey = Survey()
survey.load_agents([{
    "population": "agent_bank/populations/survey_agents",
    "agent_id": "01fd7d2a-0357-4c1b-9f3e-8eade2d537ae"
}])
with open("environment/survey/lsms_like_questionnaire.json") as f:
    questions = json.load(f)
answers = survey.survey(questions, mode="recorded")
print(answers)
survey.save("survey_run")
```

To predict answers to new categorical questions from the synthetic profile:

```python
from environment.survey.survey import Survey

survey = Survey()
survey.load_agents([{
    "population": "agent_bank/populations/survey_agents",
    "agent_id": "01fd7d2a-0357-4c1b-9f3e-8eade2d537ae"
}])
result = survey.survey({
    "Generally speaking, would you say most people can be trusted?":
        ["Can be trusted", "Can't be too careful", "Depends"]
})
print(result)
```

In prediction mode, the survey agent passes its stored question/answer pairs
to the LLM in blocks of up to four new questions, withholding exact target
items. To reproduce the paper's survey condition, use actual survey responses
in place of the synthetic answers.

#### Using it from T3 Code

[T3 Code](https://github.com/pingdotgg/t3code) runs Claude Code or Codex in threads on your machine. Open this folder as a project, start a thread, and ask the agent to do the work in plain language. It writes and runs the Python for you. The simulation's own LLM calls go through `claude` or `codex` too, so the whole run uses your subscription. Example prompts:

- **Set up:** "Set up this repo for the subscription backend: make a Python 3.12 venv in `.venv`, install `requirements.txt`, and create `simulation_engine/settings.py` from `example-settings.py` with `LLM_BACKEND = "claude_cli"`, `CLAUDE_MODEL = "sonnet"` and `EMBEDDING_BACKEND = "local"`. Then run the Quick start example from the README."
- **Survey:** "Load the interview agent in `agent_bank/populations/interview_agent/01fd7d2a-0357-4c1b-9f3e-8eade2d537ae` and ask it these questions with `categorical_resp`: [your questions and answer options]. Show the answers in a table."
- **Numeric scale:** "Ask the interview agent 'On a scale of 1 to 10, how much do you trust the news media?' with `numerical_resp` (range 1 to 10), five times, and report the mean and spread."
- **Interview:** "Interview the interview agent for five turns about its childhood and career with `agent.utterance`. Keep the dialogue history between turns, write my follow-up questions based on its answers, and save the transcript to `interview.md`."
- **New agent:** "Create a new agent with `update_scratch` from this description: [description]. Add these memories with `remember()`: [memories]. Save it to `agent_bank/populations/my_agents/<name>`, then ask it [question]."
- **Compare backends:** "Run the same three survey questions on the interview agent once with `LLM_BACKEND = "claude_cli"` and once with `"codex_cli"`, and compare the answers side by side. Settings are read at import, so use a separate run for each."

Tips:

- **Permissions.** The thread's agent starts `claude` or `codex` again for every simulation call, and those calls need network access. If the thread runs in a sandbox that blocks network access (for example Codex's default sandbox), they fail with `GENERATION ERROR: ...`. Allow the command to run with network access, or run the script yourself in a terminal. We tested from a Claude Code thread only.
- **Long runs.** Each call takes 3 to 17 seconds, so ask the agent to run big surveys in the background and save results to a file as it goes.
- **Quota.** The thread's agent and the simulation share the same subscription limits.

## Repository Structure

- `genagents/`: Core module for creating and interacting with generative agents
  - `genagents.py`: Main class for the generative agent
  - `modules/`: Submodules for interaction and memory management
    - `interaction.py`: Handles agent interactions and responses
    - `memory_stream.py`: Manages the agent's memory and reflections
- `simulation_engine/`: Contains settings and global methods
  - `prompt_template/`: All LLM prompts used in this project
  - `settings.py`: Configuration settings for the simulation engine
  - `global_methods.py`: Helper functions used across modules
  - `gpt_structure.py`: Functions for interacting with the GPT models
  - `llm_json_parser.py`: Parses JSON outputs from language models
- `reembed_agents.py`: Re-embeds stored agent memories after changing `EMBEDDING_BACKEND`
- `agent_bank/`: Directory for storing agent data
  - `populations/`: Contains pre-generated agents
    - `gss_agents/`: Demographic agent data based on the GSS
    - `interview_agent/`: Example agent data (see [Interview Agent](#sample-agent))
- `README.md`: This readme file
- `requirements.txt`: List of Python dependencies

## Usage

### Creating a Generative Agent

To create a new generative agent, use the `GenerativeAgent` class from the `genagents` module:

```python
from genagents.genagents import GenerativeAgent

# Initialize a new agent
agent = GenerativeAgent()

# Update the agent's scratchpad with personal information
agent.update_scratch({
    "first_name": "John",
    "last_name": "Doe",
    "age": 30,
    "occupation": "Software Engineer",
    "interests": ["reading", "hiking", "coding"]
})
```

The `update_scratch` method allows you to add personal attributes to the agent, which are used in interactions.

### Interacting with Agents

#### Categorical Responses

You can ask the agent to respond to categorical survey questions:

```python
questions = {
    "Do you enjoy outdoor activities?": ["Yes", "No", "Sometimes"]
}

response = agent.categorical_resp(questions)
print(response["responses"])
```

#### Numerical Responses

For numerical questions:

```python
questions = {
    "On a scale of 1 to 10, how much do you enjoy coding?": [1, 10]
}

response = agent.numerical_resp(questions, float_resp=False)
print(response["responses"])
```

#### Open-Ended Questions

Have the agent generate open-ended responses:

```python
dialogue = [
    ("Interviewer", "Tell me about your favorite hobby."),
]

response = agent.utterance(dialogue)
print(response)
```

To have a continuing conversation in the terminal, run `main.py`:

```bash
python main.py interview                 # Example interview agent (the default)
python main.py jamie                     # Jamie's profile
python main.py gss                       # First GSS agent by ID
python main.py survey                    # Ask the survey agent a categorical question
python main.py survey --agent-id 01fd7d2a-0357-4c1b-9f3e-8eade2d537ae
python main.py --list-agents --limit 20  # Show GSS names and IDs
python main.py --list-agents --search Eric
python main.py gss --agent-id 05ced490-251d-4661-a324-502952c729df
python main.py path/to/saved/agent       # Any saved agent folder
```

Chats keep earlier turns in their conversation history. Survey mode prompts for
a question and answer choices separated by `|`, then predicts one listed answer
from the survey agent's stored responses. Type `exit` or `quit` to finish.

### Memory and Reflection

Agents have a memory stream that allows them to remember and reflect on experiences.

#### Adding Memories

```python
agent.remember("Went for a hike in the mountains.", time_step=1)
```

#### Reflection

Agents can reflect on their memories to form new insights:

```python
agent.reflect(anchor="outdoor activities", time_step=2)
```

### Saving and Loading Agents

You can save the agent's state to a directory for later use:

```python
agent.save("path/to/save_directory")
```

To load an existing agent:

```python
agent = GenerativeAgent(agent_folder="path/to/save_directory")
```

## Interview Agent

An interview agent is provided in the `agent_bank/populations/interview_agent/` directory. This agent includes a pre-populated memory stream and scratchpad information for demonstration purposes.

You can load and interact with the interview agent as follows:

```python
agent = GenerativeAgent(agent_folder="agent_bank/populations/interview_agent/01fd7d2a-0357-4c1b-9f3e-8eade2d537ae")

# Interact with the agent
questions = {
    "Do you enjoy outdoor activities?": ["Yes", "No", "Sometimes"]
}
response = agent.categorical_resp(questions)
print(response["responses"])
```

## Agent Bank Access

Due to participant privacy concerns, the full agent bank containing over 1,000 generative agents based on real interviews is not publicly available at the moment. However, we plan to make aggregated responses on fixed tasks accessible for general research use in the coming months. Researchers interested in accessing individual responses on open tasks can request restricted access by contacting the authors and following a review process that ensures ethical considerations are met.

## Contributing

We welcome contributions to enhance the functionality and usability of this project. If you are interested in contributing, please follow these steps:

1. **Fork the Repository**: Click on the "Fork" button at the top right corner of this page to create a copy of the repository on your GitHub account.
2. **Clone the Forked Repository**: Use `git clone` to clone the repository to your local machine.
3. **Create a New Branch**: Create a new branch for your feature or bug fix.
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make Changes**: Implement your feature or fix the bug.
5. **Commit Changes**: Commit your changes with a descriptive commit message.
   ```bash
   git commit -am "Add new feature: your feature name"
   ```
6. **Push to GitHub**: Push your changes to your forked repository.
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Submit a Pull Request**: Go to the original repository and create a pull request from your forked repository.

Please ensure that your code follows the project's coding conventions and includes relevant tests and documentation.

## License

This project is licensed under the [MIT License](LICENSE).

## References

Please refer to the original paper for detailed information on the methodology and findings:

- Park, J. S., Zou, C. Q., Shaw, A., Hill, B. M., Cai, C., Morris, M. R., Willer, R., Liang, P., & Bernstein, M. S. (2024). *Generative Agent Simulations of 1,000 People*.

## Acknowledgements
We thank Akaash Kolluri (Github: akaashkolluri) for the help setting up this open source repository. 

In addition, we thank Douglas Guilbeault, Amir Goldberg, Diyi Yang, Jeff Hancock, Serina Chang for their insights and discussions. 


## Contact

For questions or inquiries, please contact the corresponding author:

- **Joon Sung Park**: [joonspk@stanford.edu](mailto:joonspk@stanford.edu)
