from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, TypedDict
import json
import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "resources"
JSON_FILE_1 = DATA_DIR / "Puritan_Bennett_840_Ventilator_-_Technical_Reference_manual.json"
JSON_FILE_2 = DATA_DIR / "pb840quickguide.json"

load_dotenv()

class UrgencyOutputSchema(BaseModel):
    urgency_level: str = Field(description="The urgency level (low_urgency, medium_urgency, high_urgency)")
    combined_analysis: str = Field(description="Combined analysis messages from all alarms in proper points")
    combined_remedy: str = Field(description="Combined remedy messages from all alarms in proper points")
    combined_comments: str = Field(description="Combined comments from all alarms in proper points")
    alarm_count: int = Field(description="Number of alarms processed")

class FinalOutputSchema(BaseModel):
    low_urgency: Optional[UrgencyOutputSchema] = Field(default=None, description="Combined output for low urgency alarms")
    medium_urgency: Optional[UrgencyOutputSchema] = Field(default=None, description="Combined output for medium urgency alarms")
    high_urgency: Optional[UrgencyOutputSchema] = Field(default=None, description="Combined output for high urgency alarms")
    no_urgency: List[Dict[str, Any]] = Field(default=[], description="Unprocessed no urgency alarms")

class GraphState(TypedDict):
    alarm_name: str
    raw_data: Dict[str, List[Dict[str, Any]]]
    current_urgency: Optional[str]
    processed_urgencies: List[str]
    final_output: Dict[str, Any]

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

def combine_data(alarms: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    no_urgency_alarms, low_alarms, medium_alarms, high_alarms = [], [], [], []

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

    return combined_alarms

def search_alarms_by_name(alarm_name: str, urgency: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
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

def initialize_state(state: GraphState) -> GraphState:
    alarm_name = state["alarm_name"]
    raw_data = search_alarms_by_name(alarm_name)
    
    return {
        "alarm_name": alarm_name,
        "raw_data": raw_data,
        "current_urgency": None,
        "processed_urgencies": [],
        "final_output": {}
    }

def select_next_urgency(state: GraphState) -> GraphState:
    urgencies_to_process = ["low_urgency", "medium_urgency", "high_urgency"]
    processed = state["processed_urgencies"]
    
    for urgency in urgencies_to_process:
        if urgency not in processed and urgency in state["raw_data"] and state["raw_data"][urgency]:
            return {
                **state,
                "current_urgency": urgency
            }
    
    return {
        **state,
        "current_urgency": None
    }

def process_urgency_node(state: GraphState) -> GraphState:
    model = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    current_urgency = state["current_urgency"]
    alarms_data = state["raw_data"][current_urgency]
    
    prompt = f"""You are processing {current_urgency} alarms. Here is the data:

{json.dumps(alarms_data, indent=2)}

Combine all the information from these alarms without losing any context:

1. For combined_analysis: Extract and combine all analysisMessage fields. Present as clear, organized points. Include all unique information.

2. For combined_remedy: Extract and combine all remedyMessage fields. Present as actionable steps in proper points. Include all unique remedies.

3. For combined_comments: Extract and combine all comments fields. Present as organized technical notes in proper points. Include all unique details.

Rules:
- Mix up the information but preserve ALL details
- Format as proper bullet points or numbered lists
- No information should be lost
- Remove only exact duplicates
- Keep technical accuracy
- Maintain context for each point

Return a UrgencyOutputSchema with:
- urgency_level: "{current_urgency}"
- combined_analysis: (all analysis messages as proper points)
- combined_remedy: (all remedy messages as proper points)
- combined_comments: (all comments as proper points)
- alarm_count: {len(alarms_data)}"""

    structured_llm = model.with_structured_output(UrgencyOutputSchema)
    processed = structured_llm.invoke([{"role": "user", "content": prompt}])
    
    new_final_output = state["final_output"].copy()
    new_final_output[current_urgency] = processed
    
    new_processed = state["processed_urgencies"].copy()
    new_processed.append(current_urgency)
    
    return {
        **state,
        "final_output": new_final_output,
        "processed_urgencies": new_processed
    }

def should_continue(state: GraphState) -> str:
    urgencies_to_process = ["low_urgency", "medium_urgency", "high_urgency"]
    processed = state["processed_urgencies"]
    
    remaining = [u for u in urgencies_to_process if u not in processed and u in state["raw_data"] and state["raw_data"][u]]
    
    if remaining:
        return "continue"
    else:
        return "finalize"

def finalize_output(state: GraphState) -> GraphState:
    final_output = state["final_output"].copy()
    
    if "no_urgency" in state["raw_data"]:
        final_output["no_urgency"] = state["raw_data"]["no_urgency"]
    else:
        final_output["no_urgency"] = []
    
    return {
        **state,
        "final_output": final_output
    }

def create_alarm_processing_graph():
    workflow = StateGraph(GraphState)
    
    workflow.add_node("initialize", initialize_state)
    workflow.add_node("select_urgency", select_next_urgency)
    workflow.add_node("process_urgency", process_urgency_node)
    workflow.add_node("finalize", finalize_output)
    
    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "select_urgency")
    workflow.add_edge("select_urgency", "process_urgency")
    
    workflow.add_conditional_edges(
        "process_urgency",
        should_continue,
        {
            "continue": "select_urgency",
            "finalize": "finalize"
        }
    )
    
    workflow.add_edge("finalize", END)
    
    return workflow.compile()

if __name__ == "__main__":
    alarm_name_input = input("Enter alarm name: ")
    
    graph = create_alarm_processing_graph()
    
    result = graph.invoke({
        "alarm_name": alarm_name_input
    })
    
    final_output = result["final_output"]
    
    print("\n" + "="*80)
    print("FINAL COMBINED OUTPUT")
    print("="*80 + "\n")
    
    for urgency_key in ["low_urgency", "medium_urgency", "high_urgency"]:
        if urgency_key in final_output and final_output[urgency_key]:
            urgency_data = final_output[urgency_key]
            print(f"\n{urgency_key.upper().replace('_', ' ')}")
            print("-"*80)
            print(f"Alarm Count: {urgency_data.alarm_count}\n")
            print(f"COMBINED ANALYSIS:\n{urgency_data.combined_analysis}\n")
            print(f"COMBINED REMEDY:\n{urgency_data.combined_remedy}\n")
            print(f"COMBINED COMMENTS:\n{urgency_data.combined_comments}\n")
    
    if "no_urgency" in final_output and final_output["no_urgency"]:
        print(f"\nNO URGENCY DATA (Unprocessed)")
        print("-"*80)
        print(json.dumps(final_output["no_urgency"], indent=2))
    
    print("\n" + "="*80)