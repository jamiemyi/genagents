from genagents.genagents import GenerativeAgent


def create_jamie():
    """Build Jamie's agent from the profile below."""
    agent = GenerativeAgent()
    agent.update_scratch({
        "first_name": "Jamie",
        "last_name": "Yi",
        "age": 24,
        "occupation": "Student",
        "interests": ["reading", "hiking", "baking"],
    })
    return agent


def first_answer(result):
    answers = result["responses"]
    if not answers:
        raise RuntimeError(
            "The model returned no answer. Check the LLM settings in "
            "simulation_engine/settings.py and try again."
        )
    return answers[0]


def main():
    agent = create_jamie()

    outdoor_question = "Do you enjoy outdoor activities?"
    outdoor_result = agent.categorical_resp({
        outdoor_question: ["Yes", "No", "Sometimes"],
    })
    print(f"{outdoor_question}\nJamie's answer: {first_answer(outdoor_result)}\n")

    coding_question = "On a scale of 1 to 10, how much do you enjoy coding?"
    coding_result = agent.numerical_resp({coding_question: [1, 10]}, float_resp=False)
    print(f"{coding_question}\nJamie's answer: {first_answer(coding_result)}")


if __name__ == "__main__":
    main()
