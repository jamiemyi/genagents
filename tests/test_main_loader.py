import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main


def make_agent(population, name, ready=None):
    folder = population / name
    (folder / "memory_stream").mkdir(parents=True)
    (folder / "scratch.json").write_text("{}")
    (folder / "memory_stream/nodes.json").write_text("[]")
    (folder / "memory_stream/embeddings.json").write_text("{}")
    if ready is not None:
        (folder / "meta.json").write_text(json.dumps({"ready": ready}))
    return folder


class MainLoaderTest(unittest.TestCase):
    def test_all_types_use_same_loader_and_seed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            populations = {kind: root / kind for kind in ("interview", "gss", "survey")}
            interview = make_agent(populations["interview"], "person-1")
            make_agent(populations["gss"], "person-1")
            make_agent(populations["gss"], "person-2")
            survey = make_agent(populations["survey"], "person-1", ready=True)
            make_agent(populations["survey"], "template", ready=False)
            with patch.dict(main.POPULATIONS, populations, clear=True), patch.object(
                    main, "GenerativeAgent", side_effect=lambda agent_folder: Path(agent_folder)):
                self.assertEqual(main.load_agent("interview", rng=random.Random(3)), interview)
                self.assertEqual(main.load_agent("survey", rng=random.Random(3)), survey)
                self.assertEqual(main.load_agent("gss", agent_id="person-2"),
                                 populations["gss"] / "person-2")
                first = main.load_agent("gss", rng=random.Random(17))
                second = main.load_agent("gss", rng=random.Random(17))
                self.assertEqual(first, second)
                with self.assertRaisesRegex(ValueError, "survey agent not found"):
                    main.load_agent("survey", agent_id="template")

    def test_survey_cli_uses_regular_conversation(self):
        with patch("sys.argv", ["main.py", "survey", "--seed", "7"]), patch.object(
                main, "load_agent", return_value=object()) as load, patch.object(
                    main, "Conversation") as conversation:
            main.main()
        self.assertEqual(load.call_args.args[:2], ("survey", None))
        conversation.return_value.start.assert_called_once()


if __name__ == "__main__":
    unittest.main()
