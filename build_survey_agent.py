"""Build an LSMS-inspired synthetic survey for the bundled interview agent.

These answers were reconstructed from its interview and profile. They were not
collected from the person using a World Bank questionnaire.
"""

import argparse
import json
from collections import Counter
from pathlib import Path


# id, module, question, answer, options, status, source references.
# reported = stated in source; inferred = close implication; imagined = supplied.
ITEMS = [
    ("roster.age", "roster", "What age group are you in?", "25-34", ["18-24", "25-34", "35-44", "45-54", "55 or older"], "reported", ["scratch:age"]),
    ("roster.sex", "roster", "What sex is recorded for you?", "Male", ["Female", "Male", "Another response"], "reported", ["scratch:sex"]),
    ("roster.household_size", "roster", "How many people usually live in your household, including you?", "1", ["1", "2", "3", "4 or more"], "reported", ["interview:15"]),
    ("roster.marital_status", "roster", "What is your current marital status?", "Never married", ["Never married", "Married", "Separated", "Divorced", "Widowed"], "reported", ["scratch:marital_status", "interview:92"]),
    ("roster.partner", "roster", "Do you currently live with a romantic partner?", "No", ["Yes", "No"], "reported", ["interview:92"]),
    ("roster.ethnicity", "roster", "How would you describe your ethnic background?", "Korean", ["Korean", "Other Asian", "Another background"], "reported", ["scratch:ethnicity"]),
    ("roster.us_citizen", "roster", "Are you a U.S. citizen?", "No", ["Yes", "No"], "reported", ["scratch:us_citizenship_status"]),
    ("roster.other_language", "roster", "Do you speak a language other than English?", "Yes", ["Yes", "No"], "reported", ["scratch:speak_other_language"]),
    ("education.degree", "education", "What is the highest degree you have completed?", "Master's degree", ["High school or less", "Bachelor's degree", "Master's degree", "Doctoral degree"], "reported", ["interview:1"]),
    ("education.enrolled", "education", "Are you currently enrolled in formal education?", "Yes", ["Yes", "No"], "reported", ["interview:99"]),
    ("education.program", "education", "What level of program are you enrolled in?", "Doctoral program", ["Not enrolled", "Undergraduate program", "Master's program", "Doctoral program"], "reported", ["interview:99"]),
    ("education.field", "education", "What is your current field of study?", "Computer science", ["Computer science", "Another STEM field", "Humanities or arts", "Other"], "reported", ["interview:20"]),
    ("education.institution", "education", "Where are you currently studying?", "Stanford University", ["Stanford University", "Another university", "Not enrolled"], "reported", ["interview:20"]),
    ("employment.paid_last_week", "employment", "Did you do any paid work in the last seven days?", "Yes", ["Yes", "No"], "reported", ["interview:78"]),
    ("employment.occupation", "employment", "What best describes your main occupation?", "Computer science researcher", ["Computer science researcher", "Teacher", "Other professional", "Not working"], "reported", ["interview:75"]),
    ("employment.workplace", "employment", "What kind of organization is your main workplace?", "University", ["University", "Private company", "Government", "Self-employed", "Not working"], "reported", ["interview:20"]),
    ("employment.jobs", "employment", "How many paid jobs did you have in the last month?", "1", ["0", "1", "2 or more"], "reported", ["interview:80"]),
    ("employment.hours", "employment", "About how many hours do you usually work in a week?", "70-89", ["Under 30", "30-49", "50-69", "70-89", "90 or more"], "reported", ["interview:77"]),
    ("employment.weeks", "employment", "How many weeks did you work during the last year?", "52", ["0", "1-25", "26-51", "52"], "reported", ["interview:76"]),
    ("employment.pay_source", "employment", "What is your primary source of work income?", "Paid doctoral research position", ["Paid doctoral research position", "Salary from another job", "Self-employment", "No work income"], "reported", ["interview:78"]),
    ("employment.other_income", "employment", "Did you receive occasional income from speaking events?", "Yes", ["Yes", "No"], "reported", ["interview:81"]),
    ("migration.birth_country", "migration", "In which country were you born?", "South Korea", ["South Korea", "United States", "Another country"], "reported", ["interview:1"]),
    ("migration.moved_to_us", "migration", "At roughly what age did you move to the United States?", "11-13", ["Born in the United States", "0-10", "11-13", "14-17", "18 or older"], "reported", ["interview:1"]),
    ("migration.region_at_16", "migration", "In which U.S. region did you live at age 16?", "New England", ["New England", "Other U.S. region", "Outside the United States"], "reported", ["scratch:residence_at_16"]),
    ("migration.current_state", "migration", "In which state do you currently live?", "California", ["California", "Another state", "Outside the United States"], "reported", ["scratch:state"]),
    ("housing.dwelling", "housing", "What type of dwelling do you live in?", "Apartment", ["Apartment", "Detached house", "Shared dormitory", "Other"], "reported", ["interview:64"]),
    ("housing.tenure", "housing", "Does your household own or rent its home?", "Rent", ["Own", "Rent", "Other arrangement"], "reported", ["interview:64"]),
    ("housing.expenses", "housing", "Who usually pays your housing expenses?", "Mostly me", ["Mostly me", "Shared with household members", "Mostly someone else"], "reported", ["interview:62"]),
    ("housing.near_work", "housing", "Is your home within about 30 minutes of your main workplace?", "Yes", ["Yes", "No"], "inferred", ["interview:12"]),
    ("housing.monthly_rent", "housing", "What is your approximate monthly rent?", "$2,500-$3,499", ["Under $1,500", "$1,500-$2,499", "$2,500-$3,499", "$3,500 or more"], "imagined", []),
    ("housing.electricity", "housing", "Does your dwelling have electricity?", "Yes", ["Yes", "No"], "imagined", []),
    ("housing.piped_water", "housing", "Does your dwelling have piped water?", "Yes", ["Yes", "No"], "imagined", []),
    ("housing.internet", "housing", "Does your dwelling have home internet access?", "Yes", ["Yes", "No"], "imagined", []),
    ("consumption.largest_expense", "consumption", "Which category was your largest household expense last month?", "Rent", ["Rent", "Food", "Transportation", "Health care", "Other"], "reported", ["interview:63"]),
    ("consumption.regular_costs", "consumption", "Apart from rent, which expense is most regular for you?", "Food", ["Food", "Transportation", "Health care", "Other"], "reported", ["interview:66"]),
    ("consumption.food_shortage", "consumption", "In the last year, were you unable to buy food you needed?", "No", ["Yes", "No"], "inferred", ["interview:73"]),
    ("consumption.monthly_food", "consumption", "About how much does your household spend on food per month?", "$400-$799", ["Under $400", "$400-$799", "$800-$1,199", "$1,200 or more"], "imagined", []),
    ("consumption.international_travel", "consumption", "Do you usually travel abroad to visit family during the year?", "Yes", ["Yes", "No"], "reported", ["interview:66"]),
    ("health.self_rating", "health", "How would you rate your current health?", "Good", ["Excellent", "Good", "Fair", "Poor"], "inferred", ["interview:32"]),
    ("health.chronic_condition", "health", "Do you currently have a chronic condition that limits daily life?", "No", ["Yes", "No"], "reported", ["interview:35", "interview:37"]),
    ("health.work_limitation", "health", "Has a health problem limited your ability to work recently?", "No", ["Yes", "No"], "reported", ["interview:37"]),
    ("health.coverage", "health", "Do you have health care coverage?", "Yes", ["Yes", "No"], "reported", ["interview:39", "interview:85"]),
    ("health.foregone_care", "health", "Have you had to skip needed health care because of cost or access?", "No", ["Yes", "No"], "reported", ["interview:39"]),
    ("health.regular_exercise", "health", "Do you currently maintain a regular exercise routine?", "No", ["Yes", "No"], "reported", ["interview:33"]),
    ("finance.debt", "finance", "Do you currently owe money on loans or credit cards?", "No", ["Yes", "No"], "reported", ["interview:72"]),
    ("finance.savings", "finance", "Do you have savings available for unexpected expenses?", "Yes", ["Yes", "No"], "reported", ["interview:68", "interview:89"]),
    ("finance.emergency_400", "finance", "Could you pay a $400 emergency expense from available funds?", "Yes", ["Yes", "No"], "reported", ["interview:89"]),
    ("finance.assistance", "finance", "Do you currently receive means-tested government assistance?", "No", ["Yes", "No"], "reported", ["interview:88"]),
    ("finance.missed_bill", "finance", "In the last year, did you miss a bill because you lacked money?", "No", ["Yes", "No"], "reported", ["interview:73"]),
    ("finance.comfort", "finance", "How would you describe your current financial situation?", "Comfortable", ["Comfortable", "Just getting by", "Difficult"], "reported", ["interview:90"]),
    ("finance.family_help", "finance", "Could family provide occasional financial help if needed?", "Yes", ["Yes", "No"], "reported", ["interview:62", "interview:68"]),
    ("finance.wealth", "finance", "What is your approximate total wealth?", "$80,000-$100,000", ["Under $20,000", "$20,000-$79,999", "$80,000-$100,000", "Over $100,000"], "reported", ["scratch:total_wealth"]),
    ("assets.car", "assets", "Does your household own a car?", "No", ["Yes", "No"], "imagined", []),
    ("assets.computer", "assets", "Do you personally own a computer?", "Yes", ["Yes", "No"], "imagined", []),
    ("assets.smartphone", "assets", "Do you personally own a smartphone?", "Yes", ["Yes", "No"], "imagined", []),
]


def build(source: Path, destination: Path, questionnaire_path: Path):
    scratch = json.loads((source / "scratch.json").read_text())
    nodes = json.loads((source / "memory_stream" / "nodes.json").read_text())
    node_ids = {node["node_id"] for node in nodes}
    source_meta = json.loads((source / "meta.json").read_text())
    records, questions, seen = [], {}, set()
    for item_id, module, question, answer, options, status, refs in ITEMS:
        if item_id in seen or answer not in options:
            raise ValueError(f"Invalid or duplicate survey item: {item_id}")
        seen.add(item_id)
        for ref in refs:
            kind, key = ref.split(":", 1)
            if kind == "scratch" and key not in scratch:
                raise ValueError(f"Missing scratch source: {ref}")
            if kind == "interview" and int(key) not in node_ids:
                raise ValueError(f"Missing interview source: {ref}")
        if status == "imagined" and refs:
            raise ValueError(f"Imagined answer cannot claim a source: {item_id}")
        records.append({"id": item_id, "group": module, "question": question,
                        "answer": answer, "options": options,
                        "answer_status": status, "source_refs": refs})
        questions[question] = options

    destination.mkdir(parents=True, exist_ok=True)
    questionnaire_path.parent.mkdir(parents=True, exist_ok=True)
    counts = dict(Counter(record["answer_status"] for record in records))
    meta = {
        "id": source_meta["id"], "agent_type": "synthetic_lsms_style_survey",
        "source_agent": str(source), "questionnaire": str(questionnaire_path),
        "reference_period": "time of the source interview",
        "provenance_note": "Imagined LSMS-inspired answers; not an administered LSMS or participant survey",
        "item_counts": counts,
    }
    (destination / "scratch.json").write_text(json.dumps(scratch, indent=2) + "\n")
    (destination / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (destination / "survey_responses.json").write_text(json.dumps(records, indent=2) + "\n")
    questionnaire_path.write_text(json.dumps(questions, indent=2) + "\n")

    memory_dir = destination / "memory_stream"
    memory_dir.mkdir(parents=True, exist_ok=True)
    # SurveyAgent uses survey_responses.json directly; no vector memory is needed.
    (memory_dir / "nodes.json").write_text("[]\n")
    (memory_dir / "embeddings.json").write_text("{}\n")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--questionnaire", type=Path, default=Path("environment/survey/lsms_like_questionnaire.json"))
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.destination, args.questionnaire)["item_counts"]))
