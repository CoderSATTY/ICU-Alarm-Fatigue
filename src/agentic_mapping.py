from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict, List, Dict, Any, Optional
from json_map import search_alarms_by_name
from tqdm import tqdm
import time
import os
import json
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    api_key=os.getenv("OPENAI_API_KEY")
)

class AlarmState(TypedDict):
    alarm_name: str
    urgency: Optional[str]
    grouped_alarms: Dict[str, List[Dict[str, Any]]]
    results: List[Dict[str, Any]]
    current_urgency: str
    current_alarms: List[Dict[str, Any]]

def process_urgency_node(state: AlarmState) -> AlarmState:
    urgency = state["current_urgency"]
    alarms = state["current_alarms"]
    alarm_name = state["alarm_name"]
    
    if not alarms:
        return state
    
    system_prompt = """You are an ICU Clinical Educator. Output ONLY valid JSON with keys: urgency, alarm_name, combined_summary.

Format combined_summary EXACTLY like this (use \\n for newlines):

**Alarm Description**\\nThis alarm indicates [what exactly triggers this alarm, what physiological/equipment condition it represents, and why it matters clinically]. It activates when [specific threshold or condition].\\n\\n• **What Happened?**\\n1. Brief point\\n2. Brief point\\n\\n• **What To Do?**\\n1. Brief action\\n2. Brief action\\n\\n• **Dependent Alarms**\\n1. First related alarm\\n2. Second related alarm\\n3. (list ALL dependent/related alarms from the data)\\n\\n• **Priority Task**\\n1. Single immediate action

Rules:
- Alarm Description: Explain WHAT the alarm is, WHY it triggers, clinical significance (2-3 sentences)
- All numbered points: MAX 8 words each
- Dependent Alarms: List ALL alarms that may trigger together or as a result. There could be multiple - list every single one mentioned in the data
- Use bullet (•) for section headings only
- The 'comments' field contains critical alarm details - interpret and use ALL information from it
- Escape all newlines as \\n"""

    user_prompt = f"""Summarize these {urgency} ventilator alarms:

Alarm: {alarm_name}
Urgency: {urgency}
Data: {json.dumps(alarms, indent=2)}

Return ONLY valid JSON."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    try:
        response = llm.invoke(messages)
        response_text = response.content
        
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0].strip()
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0].strip()
        
        result = json.loads(response_text)
        current_results = state.get("results", [])
        current_results.append(result)
        return {"results": current_results}
    except Exception as e:
        print(f"Error processing {urgency}: {e}")
        return state

cached_results = {}

def generate_final_output(alarm_name: str, urgency: str = None) -> str:
    global cached_results
    start_time = time.time()
    grouped_alarms = search_alarms_by_name(alarm_name, urgency)
    results = []
    
    urgency_items = [(k, v) for k, v in grouped_alarms.items() if v]
    
    for urg_level, alarms in tqdm(urgency_items, desc="Processing alarms", unit="urgency"):
        state = AlarmState(
            alarm_name=alarm_name,
            urgency=urgency,
            grouped_alarms=grouped_alarms,
            results=[],
            current_urgency=urg_level,
            current_alarms=alarms
        )
        
        updated_state = process_urgency_node(state)
        if updated_state.get("results"):
            results.extend(updated_state["results"])
    
    elapsed = time.time() - start_time
    cached_results = {"results": results, "time": elapsed}
    
    if not results:
        return f"No alarms found. (Search took {elapsed:.2f}s)"
    
    return format_all_results(results, elapsed)

def format_all_results(results, elapsed):
    output_parts = []
    for result in results:
        urg = result.get("urgency", "").upper().replace("_", " ")
        name = result.get("alarm_name", "")
        summary = result.get("combined_summary", "").replace("\\n", "\n")
        output_parts.append(f"## 🔔 {urg}: {name}\n\n{summary}")
    
    output_parts.append(f"\n---\n*Processed in {elapsed:.2f} seconds*")
    return "\n\n---\n\n".join(output_parts)

def get_urgency_output(urgency_filter: str) -> str:
    global cached_results
    if not cached_results.get("results"):
        return "Please search for an alarm first."
    
    filtered = [r for r in cached_results["results"] if r.get("urgency") == urgency_filter]
    if not filtered:
        return f"No {urgency_filter.replace('_', ' ')} alarms found."
    
    return format_all_results(filtered, cached_results.get("time", 0))

def get_low_urgency() -> str:
    return get_urgency_output("low_urgency")

def get_medium_urgency() -> str:
    return get_urgency_output("medium_urgency")

def get_high_urgency() -> str:
    return get_urgency_output("high_urgency")

if __name__ == "__main__":
    alarm_name = input("Enter alarm name: ")
    urgency = input("Enter urgency (optional): ").strip() or None
    print(generate_final_output(alarm_name, urgency))