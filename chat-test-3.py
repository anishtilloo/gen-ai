import json
import os
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

API_KEY = os.getenv('API_KEY')
client = genai.Client(api_key=API_KEY)

import subprocess

def run_command(command):
    try:
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.stderr:
            return f"❌ Error: {result.stderr.strip()}"
        return result.stdout.strip()
    except Exception as e:
        return f"❌ Exception: {str(e)}"
    
def create_file(params):
    try:
        data = json.loads(params) if isinstance(params, str) else params
        file_name = data.get("filename")
        content = data.get("content", "")  # optional

        if not file_name:
            return "❌ Missing 'filename' in input."

        with open(file_name, "w") as f:
            f.write(content)
        return f"✅ File '{file_name}' created successfully."

    except Exception as e:
        return f"❌ Error creating file: {str(e)}"




def get_weather(city: str):
    # TODO!: Do an actual API Call
    print("🔨 Tool Called: get_weather", city)
    
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return f"The weather in {city} is {response.text}."
    return "Something went wrong"


available_tools = {
    get_weather: {
        "name": "get_weather",
        "description": "Takes a city name as an input and returns the current weather for the city",
        "parameters": "City Names"
    },
    run_command: {
        "name": "run_command",
        "description": "Takes a command as input to execute on system and returns output",
        "parameters": {
            "type": "string",
            "description": "Command to execute on the terminal",
        }

    },
    create_file: {
        "name": "create_file",
        "description": "Creates a file. Input should be JSON with 'filename' and optionally 'content'.",
        "parameters": {
            "type": "string",
            "description": "File name to create a new file",
        }
    },
}

tool_lines = '\n'.join(
    f"- {text['name']}: {text['description']}"
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
    Output: {{ "step": "observe", "content": "12 Degree Cel" }}
    Output: {{ "step": "output", "content": "The weather for new york seems to be 12 degrees." }}

    Example 2:
    User Query: Create a file named magic.txt?
    Output: {{ "step": "plan", "content": "The user is interested in creating a new file" }}
    Output: {{ "step": "plan", "content": "From the available tools, I should call the run_command" }}
    Output: {{ "step": "plan", "content": "The command for creating new file will be echo > 'magic.txt'" }}
    Output: {{ "step": "action", "function": "run_command", "input": "echo > 'magic.txt'" }}
    Output: {{ "step": "observe", "content": "The file is created" }}
    Output: {{ "step": "output", "content": "The file has been created in the current directory" }}
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
                tools=[types.Tool(function_declarations=[available_tools[tool]["name"] for tool in available_tools.keys()])],
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
                output = available_tools[tool_name].get("name")(tool_input)
                print("output", output)

                if output:
                    contents.append(types.CreateFileConfig(
                        filename=tool_input,
                        content=output,
                    ))
                    continue
                else:
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

# run_command("echo > Hello-World.txt")
# run_command("ls -la")
# run_command("cat Hello-World.txt")
