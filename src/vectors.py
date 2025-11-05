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

load_dotenv()

llm = ChatAnthropic(model_name="claude-3-haiku-20240307", temperature=0, api_key=os.getenv("CLAUDE_API_KEY"))

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

json_path_1 = "C:\\IITI\\AIML\\ICU_Alarm_Fatigue\\resources\\Puritan_Bennett_840_Ventilator_-_Technical_Reference_manual.json"
json_path_2 = "C:\\IITI\\AIML\\ICU_Alarm_Fatigue\\resources\\pb840quickguide.json"
FAISS_INDEX_PATH = "faiss_alarm_index"
FAISS_INDEX_FILE = os.path.join(FAISS_INDEX_PATH, "index.faiss")
FAISS_PICKLE_FILE = os.path.join(FAISS_INDEX_PATH, "index.pkl")

VECTOR_STORE = None

def _get_text_for_embedding(doc: Dict[str, Any]) -> str:
    if "alarmMessage" in doc:
        return f"Alarm: {doc.get('alarmMessage', '')}. Meaning: {doc.get('alarmMeaning', '')}. Action: {doc.get('recommendedAction', '')}"
    elif "baseMessage" in doc:
        return f"Alarm: {doc.get('baseMessage', '')}. Urgency: {doc.get('urgency', '')}. Analysis: {doc.get('analysisMessage', '')}. Remedy: {doc.get('remedyMessage', '')}. Note: {doc.get('comments', '')}"
    else:
        return json.dumps(doc)

def create_vector_store():
    global VECTOR_STORE
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(FAISS_PICKLE_FILE):
        VECTOR_STORE = FAISS.load_local(
            FAISS_INDEX_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        print(f"Vector store loaded from {FAISS_INDEX_PATH}.")
    else:
        print("Creating new vector store...")
        try:
            with open(json_path_1, 'r') as f:
                json_file_1 = json.load(f)
            
            with open(json_path_2, 'r') as f:
                json_file_2 = json.load(f)

        except FileNotFoundError as e:
            print(f"Error: Could not find file. {e}")
            print("Please make sure the files are in the correct directory.")
            return
        except json.JSONDecodeError as e:
            print(f"Error: Could not parse JSON file. {e}")
            return

        docs_list_1 = json_file_1.get("data", {}).get("alarms", [])
        docs_list_2 = json_file_2.get("data", {}).get("alarms", [])
        
        all_docs = docs_list_1 + docs_list_2
        
        if not all_docs:
            print("No alarm data found in JSON files.")
            return

        texts = [_get_text_for_embedding(doc) for doc in all_docs]
        metadatas = all_docs
        
        VECTOR_STORE = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
        VECTOR_STORE.save_local(FAISS_INDEX_PATH)
        print(f"Vector store created and saved to {FAISS_INDEX_PATH} with {len(all_docs)} documents.")

create_vector_store()

@tool(args_schema=InputSchema)
def search_alarm_database(alarm_name: str, urgency: Optional[str] = None) -> str:
    """
    Searches the alarm vector database for a given alarm name and optional urgency.
    Returns all relevant information from all document types as a JSON string.
    """
    if not VECTOR_STORE:
        return "Error: Vector store is not initialized. Please check file paths and JSON content."

    search_query = f"Alarm: {alarm_name}"
    if urgency:
        search_query += f" Urgency: {urgency}"
        
    results = VECTOR_STORE.similarity_search(search_query, k=4)
    
    retrieved_data = [doc.metadata for doc in results]
    
    if not retrieved_data:
        return "No information found for that alarm."
        
    return json.dumps(retrieved_data)

system_prompt = """
You are an expert alarm analysis assistant.
Your job is to help a user understand a medical ventilator alarm.
The user will provide an alarm name. You MUST use the `search_alarm_database` tool to find all relevant information.
The tool will return a JSON string of data retrieved from the database. This data may come from multiple different sources (alarm definitions, technical bulletins, etc.).
You must synthesize ALL the retrieved information to provide a clear and consolidated answer.
Your final output MUST be in the specified `OutputSchema` format.
"""

if VECTOR_STORE:
    agent = create_agent(
        model=llm,
        tools=[search_alarm_database],
        response_format=OutputSchema,
        system_prompt=system_prompt
    )
    print("Agent created successfully.")
else:
    print("Agent creation failed: Vector store not initialized.")
    agent = None

def run_query(query: str):
    if not agent:
        print("Cannot run query: Agent is not initialized.")
        return

    print(f"\n--- Querying Agent for: '{query}' ---")
    try:
        response = agent.invoke(
            {"messages": [("user", query)]}
        )
        
        if response.get('structured_response'):
            print(json.dumps(response['structured_response'].model_dump(), indent=2))
        else:
            print("Agent did not return a parsed output.")
            print(f"Full response: {response}")

    except Exception as e:
        print(f"An error occurred: {e}")

if agent:
    run_query("AC POWER LOSS with medium urgency")
 