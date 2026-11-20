"""Have an open-ended conversation with an interview, GSS, or survey agent."""

import argparse
import json
import random
from pathlib import Path

from genagents.genagents import GenerativeAgent


ROOT = Path(__file__).resolve().parent
POPULATIONS = {
    "interview": ROOT / "agent_bank/populations/interview_agent",
    "gss": ROOT / "agent_bank/populations/gss_agents",
    "survey": ROOT / "agent_bank/populations/survey_agents",
}
SAVED_AGENTS = {
    "jamie": ROOT / "agent_bank/populations/my_agents/jamie",
}


def gss_agent_folders():
    return agent_folders("gss")


def agent_folders(selection):
    population = POPULATIONS[selection]
    if not population.is_dir():
        return []
    folders = []
    for folder in population.iterdir():
        if not folder.is_dir() or not all((folder / name).is_file() for name in (
                "scratch.json", "memory_stream/nodes.json",
                "memory_stream/embeddings.json")):
            continue
        if selection == "survey":
            meta = folder / "meta.json"
            if not meta.is_file() or not json.loads(meta.read_text()).get("ready"):
                continue
        folders.append(folder)
    return sorted(folders)


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


def load_agent(selection, agent_id=None, rng=None):
    if selection in SAVED_AGENTS:
        folder = SAVED_AGENTS[selection]
    elif selection in POPULATIONS:
        folders = agent_folders(selection)
        if not folders:
            raise ValueError(f"No {selection} agents found in {POPULATIONS[selection]}")
        if agent_id:
            matches = [folder for folder in folders if folder.name == agent_id]
            if not matches:
                raise ValueError(f"{selection} agent not found: {agent_id}")
            folder = matches[0]
        else:
            folder = (rng or random).choice(folders)
            print(f"Using {selection} agent {folder.name}. Pass --agent-id to choose another.")
    else:
        folder = Path(selection).expanduser()

    if not (folder / "scratch.json").is_file():
        raise ValueError(f"Agent not found: {folder}")
    if not (folder / "memory_stream/nodes.json").is_file() or not (
        folder / "memory_stream/embeddings.json"
    ).is_file():
        raise ValueError(f"Agent has incomplete memory files: {folder}")
    return GenerativeAgent(agent_folder=str(folder))


class Conversation:
    def __init__(self, agent, interviewer_name="Interviewer"):
        self.agent = (agent if isinstance(agent, GenerativeAgent)
                      else GenerativeAgent(agent_folder=agent))
        self.interviewer_name = interviewer_name
        self.conversation_history = []

    def start(self):
        name = self.agent.get_fullname() or "Agent"
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

def main():
    parser = argparse.ArgumentParser(
        description="Chat with Jamie, an interview, GSS, survey, or saved agent."
    )
    parser.add_argument("agent", nargs="?", default="interview",
                        help="interview (default), jamie, gss, survey, or a saved agent folder")
    parser.add_argument("--agent-id", help="choose a specific agent instead of a random one")
    parser.add_argument("--seed", type=int, help="repeat the same random agent choice")
    parser.add_argument("--list-agents", action="store_true",
                        help="list GSS agent IDs and names without starting a chat")
    parser.add_argument("--search", default="", help="filter --list-agents by name or ID")
    parser.add_argument("--limit", type=int, default=20,
                        help="maximum agents to show with --list-agents (default: 20)")
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.agent_id and args.agent not in POPULATIONS:
        parser.error("--agent-id requires interview, gss, or survey")
    if args.list_agents:
        list_gss_agents(args.search, args.limit)
        return
    if args.search:
        parser.error("--search requires --list-agents")
    try:
        agent = load_agent(args.agent, args.agent_id, random.Random(args.seed))
    except ValueError as exc:
        parser.error(str(exc))
    Conversation(agent).start()


if __name__ == "__main__":
    main()
