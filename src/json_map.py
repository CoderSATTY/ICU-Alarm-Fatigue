from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json
import os
from dotenv import load_dotenv
from pathlib import Path
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain.agents.structured_output import ToolStrategy

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "resources"
JSON_FILE_1 = DATA_DIR / "Puritan_Bennett_840_Ventilator_-_Technical_Reference_manual.json"
JSON_FILE_2 = DATA_DIR / "pb840quickguide.json"

load_dotenv()

class InputSchema(BaseModel):
    alarm_name: str = Field(
        description="Name of the alarm to search for (e.g., 'AC POWER LOSS', 'APNEA')."
    )
    urgency: Optional[str] = Field(
        default=None,
        description="Optional urgency level filter (e.g., 'Low', 'Medium', 'High')."
    )

class OutputSchema(BaseModel):
    alarm_name: str = Field(description="The canonical alarm name found in the database.")
    analysis_message: str = Field(description="A combined brief analysis of what this alarm means taken from all analysis_message, synthesizing all retrieved data.")
    remedy_message: str = Field(description="The combined recommended steps to resolve the alarm received from all remedy_message.")
    comments: str = Field(description="Any additional comments, technical notes, or context explained step by step.")

def open_json_file(file_path: Path) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def load_knowledge_base() -> List[Dict[str, Any]]:
    data_1 = open_json_file(JSON_FILE_1)
    alarms_1 = data_1["data"]["alarms"]

    data_2 = open_json_file(JSON_FILE_2)
    alarms_2 = data_2["data"]["alarms"]
    combined_alarms = alarms_1 + alarms_2

    return combined_alarms
def combine_data(alarms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # print(alarms)
    no_urgency_alarms,  low_alarms, medium_alarms, high_alarms = [], [], [], []

    for alarm in alarms:
        urgency = (alarm.get("urgency") or "").lower().strip()  
        
        if urgency == "":
            no_urgency_alarms.append(alarm)
        elif urgency == "low":
            low_alarms.append(alarm)
        elif urgency == "medium":
            medium_alarms.append(alarm)
        elif urgency == "high":
            high_alarms.append(alarm)
        else:
            no_urgency_alarms.append(alarm)  

    combined_alarms = {
        "low_urgency": low_alarms,
        "medium_urgency": medium_alarms,
        "high_urgency": high_alarms
    }

    if no_urgency_alarms:
        combined_alarms["no_urgency"] = no_urgency_alarms
    else:
        combined_alarms["no_urgency"] = []

    return combined_alarms

# def low_urgency(alarms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     return alarms['low_urgency']
# def medium_urgency(alarms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     return alarms['medium_urgency']
# def high_urgency(alarms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     return alarms['high_urgency']

def extract_summaries(results_list, urgency_levels):
    """
    Filters a list of alarm dictionaries for specific urgency levels
    and returns a single, formatted string of their combined_summaries.
    """
    if not results_list:
        return "Please search for an alarm first."
        
    summaries = []
    for item in results_list:
        if item.get("urgency") in urgency_levels:
            summaries.append(item.get("combined_summary", "No summary found."))
    
    if not summaries:
        return f"No alarms found for urgency level(s): {', '.join(urgency_levels)}"

    return "\n\n---\n\n".join(summaries)


def low_urgency(search_results):
    return extract_summaries(search_results, ["low_urgency"])

def medium_urgency(search_results):
    return extract_summaries(search_results, ["medium_urgency"])

def high_urgency(search_results):
    return extract_summaries(search_results, ["high_urgency"])

# @tool("map_alarms", args_schema=InputSchema, description="Search for ICU alarms by name and optional urgency.")
def search_alarms_by_name(alarm_name: str, urgency: Optional[str] = None) -> List[Dict[str, Any]]:
    all_alarms = load_knowledge_base()
    matching_alarms = []
    
    alarm_name_upper = alarm_name.upper().strip()
    
    for alarm in all_alarms:
        alarm_key = None
        if "baseMessage" in alarm:
            alarm_key = alarm["baseMessage"]
        elif "alarmMessage" in alarm:
            alarm_key = alarm["alarmMessage"]
        
        if alarm_key and alarm_key.upper().strip() == alarm_name_upper:
            if urgency:
                if "urgency" in alarm and alarm["urgency"].upper() == urgency.upper():
                    matching_alarms.append(alarm)
            else:
                matching_alarms.append(alarm)
    combined_alarm_data = combine_data(matching_alarms)
    return combined_alarm_data

if __name__ == "__main__":
    alarm_name_input = input("Enter alarm name: ")
    urgency_input = input("Enter urgency (optional, press Enter to skip): ").strip()
    
    urgency_filter = urgency_input if urgency_input else None
    
    results = search_alarms_by_name(alarm_name_input, urgency_filter)
    # print("Low_urgency:", results['low_urgency'])
    print(json.dumps(results, indent=2))
    print(f"\nFound {len(results.get("low_urgency")) + len(results.get("medium_urgency")) + len(results.get("high_urgency")) + len(results.get("no_urgency"))} matching alarm(s):\n")
