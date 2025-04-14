import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

API_KEY = os.getenv('API_KEY')
client = genai.Client(api_key=API_KEY)

system_prompt = """
You are an AI assistant who is expert in breaking down complex problems and then resolve the user query.

For the given user input, analyse the input and break down the problem step by step.
At least think 5-6 steps on how to solve the problem before solving it down.

The steps are you get a user input, you analyse, you think, you again think for several times and then return an output with explanation and then finally you validate the output as well before giving final result.

Follow the steps in sequence that is "analyse", "think", "output", "validate" and finally "result".

Rules:
1. Follow the strict JSON output as per Output schema.
2. Always perform one step at a time and wait for next input
3. Carefully analyse the user query

Output Format:
{{ step: "string", content: "string" }}

Example:
Input: What is 2 + 2.
Output: {{ step: "analyse", content: "Alright! The user is interested in maths query and he is asking a basic athematic operation" }}
Output: {{ step: "think", content: "To perform the addition i must go from left to right and add all the operands" }}
Output: {{ step: "output", content: "4" }}
Output: {{ step: "validate", content: "seems like 4 is correct ans for 2 + 2" }}
Output: {{ step: "result", content: "2 + 2 = 4 and that is calculated by adding all numbers" }}

"""

response = client.models.generate_content(
    model='gemini-2.0-flash-001', 
    # contents='What is 2 + 2',
    config=types.GenerateContentConfig(
        response_mime_type='application/json',
        system_instruction=system_prompt,
    ),
    contents=[
        types.Content(
            role='user',
            parts=[types.Part.from_text(text='what is 3 + 4 * 5')]
        ),
        types.Content(
            role='user',
            parts=[types.Part.from_text(text=json.dumps({
                "step": "analyse",
                "content": "The user wants to calculate the result of the expression '3 + 4 * 5'. This involves addition and multiplication."
            }))]
        ),
        types.Content(
            role='user',
            parts=[types.Part.from_text(text=json.dumps({
                "step": "think",
                "content": "To solve this, I need to remember the order of operations (PEMDAS/BODMAS). Multiplication comes before addition. So, I'll first multiply 4 by 5, and then add 3 to the result."
            }))]
        ),
        
        # { "role": "system", "content": system_prompt },
        # { "role": "user", "content": "what is 3 + 4 * 5" },

        # 
        # { "role": "assistant", "content": json.dumps({"step": "analyse", "content": "The user is asking for an arithmetic operation that involves both addition and multiplication, so I need to follow the order of operations."})  },
        # { "role": "assistant", "content": json.dumps({"step": "think", "content": "In order of operations, multiplication should be performed before addition. Therefore, I should first multiply 4 by 5."}) },
        # { "role": "assistant", "content": json.dumps({"step": "think", "content": "Calculate the multiplication: 4 * 5 = 20."}) },
        # { "role": "assistant", "content": json.dumps({"step": "think", "content": "Next, I need to add the result of the multiplication (20) to the number 3."}) }
    ]
)
print(response.text)

# print("Hello world")
# 