import json
import os
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

API_KEY = os.getenv('API_KEY')
client = genai.Client(api_key=API_KEY)

def run_command(command):
    result = os.system(command=command)
    return result



def get_weather(city: str):
    # TODO!: Do an actual API Call
    print("🔨 Tool Called: get_weather", city)
    
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return f"The weather in {city} is {response.text}."
    return "Something went wrong"


available_tools = {
    "get_weather": {
        "fn": get_weather,
        "description": "Takes a city name as an input and returns the current weather for the city"
    },
    "run_command": {
        "fn": run_command,
        "description": "Takes a command as input to execute on system and returns output"
    }
}

tool_lines = '\n'.join(
    f"- {text['fn']}: {text['description']}"
    for text in available_tools.values()
)


system_prompt=f"""
    You are an helpful AI Assistant who is specialized in resolving user query.
    You work on start, plan, action, observe mode.
    For the given user query and available tools, plan the step by step execution, based on the planning,
    select the relevant tool from the available tool. and based on the tool selection you perform an action to call the tool.
    Wait for the observation and based on the observation from the tool call resolve the user query.

    Rules:
    - Follow the Output JSON Format.
    - Always perform one step at a time and wait for next input
    - Carefully analyse the user query

    Output JSON Format:
    {{
        "step": "string",
        "content": "string",
        "function": "The name of function if the step is action",
        "input": "The input parameter for the function",
    }}

    Available Tools:
    {tool_lines}      
    
    
    Example:
    User Query: What is the weather of new york?
    Output: {{ "step": "plan", "content": "The user is interested in weather data of new york" }}
    Output: {{ "step": "plan", "content": "From the available tools I should call get_weather" }}
    Output: {{ "step": "action", "function": "get_weather", "input": "new york" }}
    Output: {{ "step": "observe", "output": "12 Degree Cel" }}
    Output: {{ "step": "output", "content": "The weather for new york seems to be 12 degrees." }}
"""

# - get_weather: Takes a city name as an input and returns the current weather for the city
    # - run_command: Takes a command as input to execute on system and returns output

while True:
    query = input("> ")
    contents = [
        types.Content(
            role='user',
            parts=[types.Part.from_text(text=query)]
        )
    ]
    if(query == "quit"):
        print(f"🤖:: GoodBye...")
        break

    while True:

        response = client.models.generate_content(
            model='gemini-2.0-flash-001',
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
                system_instruction=system_prompt,
            ),
            contents=contents
        )
        parsed_response = json.loads(response.text)
        contents.append(types.ModelContent(
            parts=[
                types.Part.from_text(text=json.dumps(parsed_response))
            ]
        ))

        if parsed_response.get("step") != "output":
            print(f"🧠:: {parsed_response.get("step")} :: {parsed_response.get("content")}")
            continue

        if parsed_response.get("step") == "action":
            tool_name = parsed_response.get("function")
            tool_input = parsed_response.get("input")

            if available_tools.get(tool_name, False) != False:
                output = available_tools[tool_name].get("fn")(tool_input)
                contents.append(types.ModelContent(
                    parts=[
                    types.Part.from_function_call(
                        name=tool_name,
                        args=tool_input,
                    ),
                    ]
                ))
                continue
        
        print(f"🤖:: {parsed_response.get("step")} :: {parsed_response.get("content")}")
        break


