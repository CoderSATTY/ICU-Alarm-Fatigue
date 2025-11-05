from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json
from langchain_groq import ChatGroq
import getpass
import os
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "resources"
JSON_FILE_1 = DATA_DIR / "Puritan_Bennett_840_Ventilator_-_Technical_Reference_manual.json"
JSON_FILE_2 = DATA_DIR / "pb840quickguide.json"

# print(f"DATA directory: {DATA_DIR}")
# print(f"json_file_1: {JSON_FILE_1}")
# print(f"json_file_2: {JSON_FILE_2}")


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
    with open(file_path, "r", encoding = "utf-8") as f:
        data = json.load(f)

    return data

from typing import List, Dict, Any

def load_knowledge_base() -> List[Dict[str, Any]]:
    data_1 = open_json_file(JSON_FILE_1)
    data_2 = open_json_file(JSON_FILE_2)

    alarms_1 = data_1["data"]["alarms"]
    alarms_2 = data_2["data"]["alarms"]

    combined_alarms = alarms_1 + alarms_2

    print(f"Loaded {len(combined_alarms)} total alarms.")
    return combined_alarms


print(json.dumps(load_knowledge_base(), indent=2))