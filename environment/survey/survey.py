import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from simulation_engine.settings import *
from simulation_engine.global_methods import *
from environment.environment import Environment 
from genagents.genagents import GenerativeAgent
from genagents.survey_agent import SurveyAgent


class Survey(Environment): 
  def __init__(self, saved_dir=None):
    super().__init__('survey', saved_dir)
    if not saved_dir: 
      self.responses = pd.DataFrame(columns=['agent_pid'])
    

  def _load_responses(self, responses_path):
    responses_path = os.path.join(responses_path, "responses.csv")

    if os.path.exists(responses_path):
      self.responses = pd.read_csv(responses_path)
      print(f"Loaded responses from {responses_path}")
    else:
      print(f"Responses file not found at {responses_path}")


  def _package_responses(self):
    if self.responses.empty:
      print("No responses to package.")
      return []
    
    columns = ['agent_pid'] + [col for col in self.responses.columns if col != 'agent_pid']
    return [self.responses.columns.tolist()] + self.responses[columns].values.tolist()


  def _save_responses(self, save_dir, packaged_responses):
    write_list_of_list_to_csv(packaged_responses, os.path.join(save_dir, "responses.csv"))


  def _administer_to_agent(self, agent_pid, questions, mode):
    population = self.agent_registry[agent_pid]["population"]
    agent_id = self.agent_registry[agent_pid]["agent_id"]

    agent_folder = os.path.join(population, agent_id)
    print (f"Generating {agent_pid}'s response")
    if os.path.isfile(os.path.join(agent_folder, "survey_responses.json")):
      agent = SurveyAgent(agent_folder)
      output = (agent.recorded_resp(questions) if mode == "recorded"
                else agent.categorical_resp(questions))
    else:
      if mode == "recorded":
        raise ValueError(f"Recorded answers are unavailable for {agent_folder}")
      agent = GenerativeAgent(population, agent_id)
      output = agent.categorical_resp(questions)
    output["agent_pid"] = agent_pid
    print (output)
    return output


  def _filter_agents(self, inclusion_criteria):
    if inclusion_criteria:
      # Apply inclusion criteria as filters on the DataFrame
      criteria = [(self.responses[question].isin(allowed_responses)) 
                for question, allowed_responses in inclusion_criteria.items()]
      mask = pd.concat(criteria, axis=1).all(axis=1)
      filtered_agents = self.responses[mask]['agent_pid'].unique()
      return list(filtered_agents)
    return list(self.agent_registry)


  def survey(self, questions, inclusion_criteria={}, num_threads=50, mode="predict"):
    if mode not in ("predict", "recorded"):
      raise ValueError("mode must be 'predict' or 'recorded'")
    filtered_agents = self._filter_agents(inclusion_criteria)

    if not filtered_agents:
      print("No agents meet the inclusion criteria.")
      return []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
      futures = [executor.submit(self._administer_to_agent, agent_pid, questions, mode)
                 for agent_pid in filtered_agents]
      outputs = [future.result() for future in futures]

    for output in outputs:
      response_data = {question: output["responses"][i] 
                       for i, question in enumerate(questions.keys())}
      response_data["agent_pid"] = output["agent_pid"]

      if (self.responses['agent_pid'] == output["agent_pid"]).any():
        idx = self.responses[self.responses['agent_pid'] == output["agent_pid"]].index
        for key, value in response_data.items():
            self.responses.loc[idx, key] = value
      else:
        self.responses = pd.concat([self.responses, 
                                    pd.DataFrame([response_data])], 
                                    ignore_index=True)

    return outputs
