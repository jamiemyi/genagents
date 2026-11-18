"""Talk to an agent or ask the survey agent categorical questions."""

import argparse
import json
from pathlib import Path

from genagents.genagents import GenerativeAgent
from genagents.survey_agent import SurveyAgent
from jamie_agent import create_jamie


ROOT = Path(__file__).resolve().parent
INTERVIEW_AGENT = (
    ROOT / "agent_bank/populations/interview_agent"
    / "01fd7d2a-0357-4c1b-9f3e-8eade2d537ae"
)
GSS_AGENTS = ROOT / "agent_bank/populations/gss_agents"
SURVEY_AGENTS = ROOT / "agent_bank/populations/survey_agents"


def gss_agent_folders():
    return sorted(folder for folder in GSS_AGENTS.iterdir()
                  if folder.is_dir() and (folder / "scratch.json").is_file())


def list_gss_agents(search, limit):
    matches = 0
    for folder in gss_agent_folders():
        with (folder / "scratch.json").open() as file:
            profile = json.load(file)
        name = " ".join(str(profile.get(key, "")) for key in
                        ("first_name", "last_name")).strip()
        if search and search.casefold() not in f"{folder.name} {name}".casefold():
            continue
        matches += 1
        if matches <= limit:
            print(f"{folder.name}  {name}")
    if matches > limit:
        print(f"Showing {limit} of {matches} matches. Use --search to narrow the list.")
    elif matches == 0:
        print("No matching GSS agents found.")


def load_agent(selection, agent_id=None):
    if selection == "jamie":
        return create_jamie()
    if selection == "interview":
        folder = INTERVIEW_AGENT
    elif selection == "gss":
        folders = gss_agent_folders()
        if not folders:
            raise ValueError(f"No GSS agents found in {GSS_AGENTS}")
        if agent_id and agent_id not in {item.name for item in folders}:
            raise ValueError(f"GSS agent not found: {agent_id}")
        folder = GSS_AGENTS / agent_id if agent_id else folders[0]
        if not agent_id:
            print(f"Using GSS agent {folder.name}. Pass --agent-id to choose another.")
    else:
        folder = Path(selection).expanduser()

    if not (folder / "scratch.json").is_file():
        raise ValueError(f"Agent not found: {folder}")
    if not (folder / "memory_stream/nodes.json").is_file() or not (
        folder / "memory_stream/embeddings.json"
    ).is_file():
        raise ValueError(f"Agent has incomplete memory files: {folder}")
    return GenerativeAgent(agent_folder=str(folder))


def load_survey_agent(agent_id=None):
    if not SURVEY_AGENTS.is_dir():
        raise ValueError(f"No survey agents found in {SURVEY_AGENTS}")
    folders = sorted(folder for folder in SURVEY_AGENTS.iterdir()
                     if folder.is_dir() and (folder / "meta.json").is_file()
                     and (folder / "survey_responses.json").is_file())
    if not folders:
        raise ValueError(f"No survey agents found in {SURVEY_AGENTS}")
    if agent_id and agent_id not in {folder.name for folder in folders}:
        raise ValueError(f"Survey agent not found: {agent_id}")
    folder = SURVEY_AGENTS / agent_id if agent_id else folders[0]
    if not agent_id:
        print(f"Using survey agent {folder.name}. Pass --agent-id to choose another.")
    return SurveyAgent(folder)


class Conversation:
    def __init__(self, agent, interviewer_name="Interviewer"):
        self.agent = (agent if isinstance(agent, GenerativeAgent)
                      else GenerativeAgent(agent_folder=agent))
        self.interviewer_name = interviewer_name
        self.conversation_history = []

    def start(self):
        name = self.agent.get_fullname()
        print(f"Starting conversation with {name}.\nType 'exit' to end the conversation.\n")
        while True:
            try:
                user_input = input(f"{self.interviewer_name}: ")
            except (EOFError, KeyboardInterrupt):
                print("\nConversation ended.")
                break
            if user_input.strip().lower() in ("exit", "quit"):
                print("Conversation ended.")
                break
            if not user_input.strip():
                continue
            self.conversation_history.append([self.interviewer_name, user_input])
            agent_response = self.agent.utterance(self.conversation_history)
            print(f"{name}: {agent_response}")
            self.conversation_history.append([name, agent_response])

#written such that survey agents have survey conversations, not sure if I like this
class SurveyConversation:
    def __init__(self, agent):
        self.agent = agent

    def start(self):
        print("Ask a categorical survey question. Separate answer choices with '|'.\n"
              "Type 'exit' or 'quit' to end.\n")
        while True:
            try:
                question = input("Question: ").strip()
                if question.lower() in ("exit", "quit"):
                    break
                if not question:
                    continue
                choices = input("Options (separate with |): ").strip()
                if choices.lower() in ("exit", "quit"):
                    break
            except (EOFError, KeyboardInterrupt):
                print("\nSurvey ended.")
                return
            options = [option.strip() for option in choices.split("|")]
            if len(options) < 2 or any(not option for option in options) or len(set(options)) != len(options):
                print("Enter at least two distinct, non-empty options.\n")
                continue
            try:
                answer = self.agent.categorical_resp({question: options})["responses"][0]
            except (RuntimeError, ValueError) as exc:
                print(f"Could not answer: {exc}\n")
                continue
            print(f"Answer: {answer}\n")
        print("Survey ended.")


def main():
    parser = argparse.ArgumentParser(
        description="Chat with an agent or ask the survey agent categorical questions."
    )
    parser.add_argument("agent", nargs="?", default="interview",
                        help="interview (default), jamie, gss, survey, or a saved agent folder")
    parser.add_argument("--agent-id", help="GSS or survey agent ID")
    parser.add_argument("--list-agents", action="store_true",
                        help="list GSS agent IDs and names without starting a chat")
    parser.add_argument("--search", default="", help="filter --list-agents by name or ID")
    parser.add_argument("--limit", type=int, default=20,
                        help="maximum agents to show with --list-agents (default: 20)")
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.agent_id and args.agent not in ("gss", "survey"):
        parser.error("--agent-id requires the gss or survey agent")
    if args.list_agents:
        list_gss_agents(args.search, args.limit)
        return
    if args.search:
        parser.error("--search requires --list-agents")
    try:
        agent = (load_survey_agent(args.agent_id) if args.agent == "survey"
                 else load_agent(args.agent, args.agent_id))
    except ValueError as exc:
        parser.error(str(exc))
    if args.agent == "survey":
        SurveyConversation(agent).start()
    else:
        Conversation(agent).start()


if __name__ == "__main__":
    main()
