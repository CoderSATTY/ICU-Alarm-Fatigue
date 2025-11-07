from crewai import Agent, Task, Crew
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from json_map import search_alarms_by_name  # Assuming json_map.py exists
import json
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatAnthropic(model_name="claude-3-haiku-20240307", temperature=0, api_key=os.getenv("ANTHROPIC_API_KEY"))

# 1. Updated Pydantic Model
class AlarmSummary(BaseModel):
    urgency: str = Field(description="The urgency level of the alarm.")
    alarm_name: str = Field(description="The base name of the alarm.")
    combined_summary: str = Field(description="A single merged summary of all analysis, remedy, and comments, formatted with bullet points.")


def process_urgency_group(urgency: str, alarms: list) -> dict:
    agent = Agent(
        role="ICU Alarm Expert",
        goal="Combine multiple alarm entries into one structured summary.",
        backstory="Expert biomedical engineer specializing in ICU ventilator alarms.",
        llm=llm,
        verbose=True
        # output_pydantic removed for more reliable raw JSON parsing
    )

    # 2. Updated Task Description
    task = Task(
        description=f"""
        You are given a list of ICU alarms, all for the urgency '{urgency}'. 
        Combine them into a single, structured JSON output.

        Rules:
        1.  Create a JSON object with three keys: "urgency", "alarm_name", and "combined_summary".
        2.  Set 'urgency' to '{urgency}'.
        3.  Set 'alarm_name' to the alarm's base message (e.g., "AC POWER LOSS").
        4.  For 'combined_summary', create a single string. Inside this string, synthesize ALL information 
            from 'analysisMessage', 'remedyMessage', and 'comments' into clear markdown bullet points.
        5.  Group the bullet points logically. Start with "Analysis:", then "Remedy:", then "Comments:".
        6.  Use '\n' for newlines.

        Example for 'combined_summary':
        "Analysis:\n* Point 1 from analysis...\n* Point 2 from another analysis...\n\nRemedy:\n* Point 1 from remedy...\n\nComments:\n* Point 1 from comments..."

        Alarms List: {json.dumps(alarms, indent=2)}
        """,
        expected_output="""A single, raw, valid JSON object string ONLY. Do not add any other text, markdown, or commentary.
        Example:
        {
          "urgency": "low_urgency",
          "alarm_name": "AC POWER LOSS",
          "combined_summary": "Analysis:\n* Operating on battery.\n\nRemedy:\n* Prepare for power loss.\n\nComments:\n* Power switch on, AC power not available"
        }
        """,
        agent=agent 
    )

    crew = Crew(agents=[agent], tasks=[task])
    response = crew.kickoff()  # 'response' is a CrewOutput object
    
    # 3. Robust parsing from the 'raw' attribute
    try:
        return json.loads(response.raw)
    except json.JSONDecodeError:
        print(f"Error parsing JSON from agent output: {response.raw}")
        return {} 
    except AttributeError:

        try:
            return json.loads(response)
        except Exception as e:
            print(f"Unexpected crew.kickoff() response type: {e} | {response}")
            return {}

def generate_final_output(grouped_alarms: dict):
    results = []
    for urgency, alarms in grouped_alarms.items():
        if alarms:
            print(f"\n--- Processing {urgency} alarms... ---")
            results.append(process_urgency_group(urgency, alarms))
    return results

if __name__ == "__main__":
    alarm_name = input("Enter alarm name: ")
    urgency = input("Enter urgency (optional): ").strip() or None
    grouped_alarms = search_alarms_by_name(alarm_name, urgency)
    
    results = generate_final_output(grouped_alarms)
    
    print("\n--- FINAL SUMMARY ---")
    print(json.dumps(results, indent=2))