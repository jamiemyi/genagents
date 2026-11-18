"""
Re-embed the memory nodes of one or more agents with the current
EMBEDDING_BACKEND, so stored vectors match the model used at query time.

Usage:
  python reembed_agents.py agent_bank/populations/interview_agent [more paths]

Each path can be an agent folder or any folder containing agents. The old
embeddings.json is kept as embeddings.json.bak (if no backup exists yet).
"""
import glob
import json
import os
import shutil
import sys

from simulation_engine.gpt_structure import get_text_embedding, EMBEDDING_BACKEND


def reembed_agent(agent_folder):
  memory_dir = os.path.join(agent_folder, "memory_stream")
  with open(os.path.join(memory_dir, "nodes.json")) as f:
    nodes = json.load(f)
  if not nodes:
    return 0

  embeddings = {}
  for node in nodes:
    if node["content"] not in embeddings:
      embeddings[node["content"]] = get_text_embedding(node["content"])

  embeddings_file = os.path.join(memory_dir, "embeddings.json")
  if os.path.exists(embeddings_file) and not os.path.exists(f"{embeddings_file}.bak"):
    shutil.copy(embeddings_file, f"{embeddings_file}.bak")
  with open(embeddings_file, "w") as f:
    json.dump(embeddings, f)
  return len(embeddings)


if __name__ == "__main__":
  if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)

  for path in sys.argv[1:]:
    pattern = os.path.join(path, "**", "memory_stream", "nodes.json")
    for nodes_file in sorted(glob.glob(pattern, recursive=True)):
      agent_folder = os.path.dirname(os.path.dirname(nodes_file))
      count = reembed_agent(agent_folder)
      print(f"{agent_folder}: {count} embeddings ({EMBEDDING_BACKEND})")
