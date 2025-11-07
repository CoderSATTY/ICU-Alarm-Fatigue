from crewai import Agent, Task, Crew
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from json_map import search_alarms_by_name
import json
import os
import re
from dotenv import load_dotenv

load_dotenv()

llm = ChatAnthropic(model_name="claude-3-haiku-20240307", temperature=0, api_key=os.getenv("ANTHROPIC_API_KEY"))

class AlarmSummary(BaseModel):
    urgency: str = Field(description="The urgency level of the alarm.")
    alarm_name: str = Field(description="The base name of the alarm.")
    combined_summary: str = Field(description="A natural language summary for nurses.")


def extract_json_from_response(response_text):
    """Extract and repair JSON from various response formats"""
    # Remove markdown code blocks
    if '```json' in response_text:
        response_text = response_text.split('```json')[1].split('```')[0].strip()
    elif '```' in response_text:
        response_text = response_text.split('```')[1].split('```')[0].strip()
    
    # Find JSON object using regex as fallback
    json_match = re.search(r'\{[\s\S]*\}', response_text)
    if json_match:
        response_text = json_match.group(0)
    
    return response_text


def repair_json_string(json_text):
    """Attempt to repair common JSON formatting issues"""
    try:
        # First, try to parse as-is
        return json.loads(json_text)
    except json.JSONDecodeError:
        # If it fails, try to fix common issues
        
        # Method 1: Replace literal newlines with escaped newlines within string values
        # This regex finds strings in JSON and replaces newlines within them
        def fix_newlines_in_strings(match):
            string_content = match.group(0)
            # Replace actual newlines with \n
            fixed = string_content.replace('\n', '\\n').replace('\r', '\\r')
            return fixed
        
        # Find all string values in JSON (content between quotes)
        # This pattern looks for quoted strings
        pattern = r'"combined_summary"\s*:\s*"([^"]*(?:"[^"]*)*)"'
        
        # Try to extract just the combined_summary value and fix it
        summary_match = re.search(pattern, json_text, re.DOTALL)
        if summary_match:
            original_summary = summary_match.group(0)
            # Get the content between the quotes
            content_match = re.search(r'"combined_summary"\s*:\s*"(.*)"', json_text, re.DOTALL)
            if content_match:
                content = content_match.group(1)
                # Fix the content
                fixed_content = content.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                # Replace in original JSON
                json_text = json_text.replace(content, fixed_content)
        
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            # If still failing, return None
            return None


def process_urgency_group(urgency: str, alarms: list) -> dict:
    agent = Agent(
        role="ICU Clinical Educator",
        goal="Translate technical alarm data into clear, actionable guidance for nurses.",
        backstory="Experienced ICU nurse educator who specializes in making complex medical equipment information accessible to bedside clinicians.",
        llm=llm,
    )
    
    task = Task(
        description=f"""
        You are reviewing {urgency} level DEVICE ALERT alarms for a ventilator. Your audience is ICU nurses who need to understand what's happening and what to do.

        CRITICAL JSON FORMATTING REQUIREMENT:
        You MUST output VALID JSON. This means:
        - Newlines inside strings MUST be escaped as \\n (two characters: backslash followed by n)
        - Do NOT use actual line breaks inside the JSON string values
        - All quotes must be properly escaped
        - The output must be parseable by json.loads() in Python

        CONTENT RULES:
        1. Write in clear, natural language that nurses can immediately understand
        2. Output ONLY a valid JSON object with three keys: "urgency", "alarm_name", "combined_summary"
        3. Set 'urgency' to '{urgency}'
        4. Set 'alarm_name' to "DEVICE ALERT"
        5. For 'combined_summary', write a natural language summary organized in THREE sections:
           - **What's Happening:** Explain the situation in plain language based on analysisMessage
           - **What To Do:** Provide clear action steps based on remedyMessage
           - **Additional Information:** Include relevant technical details from comments
        
        6. Use bullet points (•) for each point within sections
        7. Make sure each point is specific and actionable
        8. Combine similar information across multiple alarms into coherent points
        9. Use medical terminology appropriately but explain technical jargon
        10. Focus on patient safety and clinical relevance

        FORMATTING RULES:
        - Use bullet point symbol (•) not asterisks
        - Use \\n\\n (escaped) for blank lines between sections
        - Use \\n (escaped) for new bullet points
        - Keep each bullet point concise but complete
        - Write naturally as if explaining to a colleague

        Alarms to summarize:
        {json.dumps(alarms, indent=2)}
        """,
        expected_output="""A single valid JSON object ONLY on a single line or with proper JSON formatting. No markdown blocks. CRITICAL: All newlines in the combined_summary string MUST be escaped as \\n. Example:
        {"urgency": "low_urgency", "alarm_name": "DEVICE ALERT", "combined_summary": "**What's Happening:**\\n• The ventilator's background diagnostic checks have identified an internal problem\\n• Breath delivery to the patient is continuing normally\\n\\n**What To Do:**\\n• Continue monitoring the patient closely\\n• Arrange for biomedical engineering service\\n\\n**Additional Information:**\\n• This is a maintenance issue, not an immediate patient safety concern"}
        """,
        agent=agent
    )

    crew = Crew(agents=[agent], tasks=[task])
    
    try:
        response = crew.kickoff()
        
        # Get response text
        if hasattr(response, 'raw'):
            response_text = response.raw
        elif hasattr(response, 'output'):
            response_text = response.output
        else:
            response_text = str(response)
        
        # Extract and parse JSON
        json_text = extract_json_from_response(response_text)
        result = json.loads(json_text)
        
        return result
        
    except json.JSONDecodeError as e:
        print(f"✗ JSON parsing error: {e}")
        print(f"Response text: {response_text[:300]}...")
        return {}
    except Exception as e:
        print(f"✗ Unexpected error: {type(e).__name__}: {e}")
        return {}


def generate_final_output(alarm_name: str,urgency: str =None) -> list:
    grouped_alarms = search_alarms_by_name(alarm_name, urgency)
    results = []
    for urgency, alarms in grouped_alarms.items():
        if alarms:
            print(f"\n{'='*60}")
            print(f"Processing {urgency.upper().replace('_', ' ')} alarms...")
            print(f"{'='*60}")
            result = process_urgency_group(urgency, alarms)
            if result:
                results.append(result)
                print(f"✓ Successfully processed {urgency}")
            else:
                print(f"✗ Failed to process {urgency}")
    return results


def print_formatted_summary(results):
    """Print the summaries in a readable format"""
    if not results:
        print("\n⚠ No results to display. Check error messages above.")
        return
    
    print("\n" + "="*80)
    print("VENTILATOR ALARM SUMMARY FOR CLINICAL STAFF".center(80))
    print("="*80)
    
    for result in results:
        if result:
            urgency_display = result.get('urgency', 'Unknown').upper().replace('_', ' ')
            print(f"\n{'─'*80}")
            print(f"URGENCY LEVEL: {urgency_display}")
            print(f"ALARM: {result.get('alarm_name', 'Unknown')}")
            print(f"{'─'*80}\n")
            
            # Handle both escaped and unescaped newlines
            summary = result.get('combined_summary', 'No summary available')
            summary = summary.replace('\\n', '\n')
            print(summary)
            print()


if __name__ == "__main__":
    print("="*80)
    print("ICU VENTILATOR ALARM SUMMARY GENERATOR".center(80))
    print("="*80 + "\n")
    
    alarm_name = input("Enter alarm name: ")
    urgency = input("Enter urgency (optional, or press Enter to skip): ").strip() or None
    
    # print(f"\nFound alarms in {len([u for u, a in grouped_alarms.items() if a])} urgency level(s)")
    results = generate_final_output(alarm_name, urgency)
    
    # Print formatted output
    print_formatted_summary(results)
    
    # Also save JSON for reference
    if results:
        print("\n" + "="*80)
        print("JSON OUTPUT (for system integration)".center(80))
        print("="*80)
        print(json.dumps(results, indent=2))