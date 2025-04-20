import os
import json
from pathlib import Path 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from openai import OpenAI

from langchain_qdrant import QdrantVectorStore

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./gen-lang-client-0749002261-05b6a39a97f3.json"

api_key = os.getenv("GEMINI_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Indexing Pipeline for RAG
def load_documents(file_path: str):
    loader = PyPDFLoader(file_path=file_path)
    return loader.load()

# pdf_path = Path(__file__).parent / "nodejs.pdf"

def text_splitter(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return text_splitter.split_documents(documents=docs)

def embedding_model():
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )
# embeddings = embedding_model()

def store_documents_in_qdrant(split_docs):
    return QdrantVectorStore.from_documents(
        documents=split_docs,
        url="http://localhost:6333",
        collection_name="learning_langchain",
        embedding=embeddings
    )


# embedder = OpenAIEmbeddings(
#     model="text-embedding-3-large",
#     api_key=""
# )


# vector_store = QdrantVectorStore.from_documents(
#     documents=[],
#     url="http://localhost:6333",
#     collection_name="learning_langchain",
#     embedding=embeddings
# )

# vector_store.add_documents(documents=split_docs)
# print("Injection Done")

def data_retriver(db_url: str, collection_name: str, embeddings):
    return QdrantVectorStore.from_existing_collection(
        url=db_url,
        collection_name=collection_name,
        embedding=embeddings
    )

# retriver = QdrantVectorStore.from_existing_collection(
#     url="http://localhost:6333",
#     collection_name="learning_langchain",
#     embedding=embeddings
# )

# print("Relevant Chunks", search_result)


system_prompt = f"""
    You are an helpful AI Assistant who is specialized in resolving user query.
    You work on start, plan, action, observe mode.
    For the given user query and available tools, plan the step by step execution, based on the planning,
    select the relevant tool from the available tool. and based on the tool selection you perform an action to call the tool
    Wait for the observation and based on the observation from the tool call resolve the user query.

    Rules:
    - Follow the Output JSON Format.
    - Always perform one step at a time and wait for next intput 
    - Carefully analyse the user query

    Output JSON Format:
    {{
        "step": "string",
        "content": "string",
        "function": "The name of function if the step is action",
        "input": "The input parameter for the function",
    }}

    Available Tools:
    - get_weather: Takes a city name as an input and returns the current weather of that city.
    - run_command: Runs a shell command and returns the output.

    Example:
    User Query:  What is the weather of new york?
    Output: {{ "step": "plan", "content": "The user is interested in weather data of new york" }}
    Output: {{ "step": "plan", "content": "From the available tools I should call get_weather" }}
    Output: {{ "step": "action", "function": "get_weather", "input": "new york" }}
    Output: {{ "step": "observe", "output": "12 Degree Celcius" }}
    Output: {{ "step": "output", "content": "The weather for new york seems to be 12 degrees." }}
"""

messages = [
    { 'role': 'system', 'content': system_prompt },
]

while True:
    docs = load_documents("nodejs.pdf")
    split_docs = text_splitter(docs)
    embeddings = embedding_model()
    vector_store = store_documents_in_qdrant(split_docs)
    retriver = data_retriver("http://localhost:6333", "learning_langchain", embeddings)
    user_query = input('> ')

    search_result = retriver.similarity_search(
        query="What is FS Module?"
    )

    messages.append({ 'role': 'user', 'content': user_query })

    while True:
        response = client.chat.completions.create(
            model='gemini-2.0-flash',
            response_format={"type": "json_object"},
            messages=messages,
        )

        parsed_output = json.loads(response.choices[0].message.content)
        messages.append({ 'role': 'assistant', 'content': json.dumps(parsed_output) })

        if parsed_output['step'] == 'plan':
            print(f"🧠: {parsed_output.get('content')}")
            continue
        
        if parsed_output['step'] == 'action':
            tool_name = parsed_output.get('function')
            tool_input = parsed_output.get('input')

            if available_tools.get(tool_name, False) != False:
                output = available_tools[tool_name].get('fn')(tool_input)
                messages.append({ 'role': 'assistant', 'content': json.dumps({ 'step': 'observe', 'output': output }) })
                continue

        if parsed_output['step'] == 'output':
            print(f"🤖: {parsed_output.get('content')}")
            break