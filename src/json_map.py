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
    analysis_message: str = Field(description="A brief analysis of what this alarm means, synthesizing all retrieved data.")
    remedy_message: str = Field(description="The recommended steps to resolve the alarm.")
    comments: str = Field(description="Any additional comments, technical notes, or context.")

def open_json_file(file_path: Path) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def load_knowledge_base() -> List[Dict[str, Any]]:
    data_1 = open_json_file(JSON_FILE_1)
    data_2 = open_json_file(JSON_FILE_2)

    alarms_1 = data_1["data"]["alarms"]
    alarms_2 = data_2["data"]["alarms"]

    combined_alarms = alarms_1 + alarms_2

    return combined_alarms

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
    
    return matching_alarms
# model  = ChatGroq(
#     model="llama-3.3-70b-versatile",
#     temperature=0,
#     max_tokens=None,
#     reasoning_format="parsed",
#     timeout=None,
#     max_retries=2,
#     api_key=os.getenv("GROQ_API_KEY")
# )
if __name__ == "__main__":
    alarm_name_input = input("Enter alarm name: ")
    urgency_input = input("Enter urgency (optional, press Enter to skip): ").strip()
    
    urgency_filter = urgency_input if urgency_input else None
    
    results = search_alarms_by_name(alarm_name_input, urgency_filter)
    
    print(f"\nFound {len(results)} matching alarm(s):\n")
    print(json.dumps(results, indent=2))
    # agent = create_agent(model, tools=[search_alarms_by_name], response_format= ToolStrategy(OutputSchema)
    #                      )
    # result = agent.invoke({
    #     "messages": [{"role": "user", "content": "Alarm name: 'APNEA' and urgency: 'Low'. Provide the canonical alarm name, analysis message, remedy message, and comments."}]
    # })
    # print("Response:\n" ,result["structured_response"])