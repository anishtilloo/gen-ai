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

api_key = os.getenv("GEMINI_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

pdf_path = Path(__file__).parent / "nodejs.pdf"
# Indexing Pipeline for RAG
def load_documents(file_path: str):
    loader = PyPDFLoader(file_path=file_path)
    return loader.load()

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

def store_documents_in_qdrant(split_docs):
    return QdrantVectorStore.from_documents(
        documents=split_docs,
        url="http://localhost:6333",
        collection_name="learning_langchain",
        embedding=embeddings
    )


def data_retriever(db_url: str, collection_name: str, embeddings):
    return QdrantVectorStore.from_existing_collection(
        url=db_url,
        collection_name=collection_name,
        embedding=embeddings
    )

def similarity_search(retriever, query: str):
    return retriever.similarity_search(
        query=query
    )


embeddings = embedding_model()
while True:
    retriever = data_retriever("http://localhost:6333", "learning_langchain", embeddings)
    user_query = input('> ')

    if user_query.lower() == "exit":
        print("Exiting... Goodbye!")
        break
        
    similarity_search_result = similarity_search(retriever, user_query)
    context = similarity_search_result[0].page_content
    prompt_template = f"""
    You are an helpful AI Assistant who is specialized in resolving user query based on the context provided.
    You work on start, plan, action, observe mode.

    Relevant Context: {context}

    If the user query is relevant to the context provided, then only answer the user query, if it is not relevant to the context then say the question is out of context.

    Rules:
    - Follow the Output JSON Format.
    - Always perform one step at a time and wait for next input 
    - Carefully analyse the user query

    Output JSON Format:
    {{
        "step": "string",
        "content": "string",
        "input": "The input parameter for the function",
    }}

    Example:
    User Query:  Explain fs module in detail?
    Output: {{ "step": "plan", "content": "The user is interested in understanding the fs module" }}
    Output: {{ "step": "plan", "content": "Based on my context the user is asking fs module from node js" }}
    Output: {{ "step": "observe", "content": "The fs module in Node.js provides file I/O operations. It offers simple wrappers around standard POSIX functions, allowing you to interact with the file system. To use the fs module, you need to require it using require('fs'). All the methods in the fs module have both asynchronous and synchronous forms, giving you flexibility in how you handle file operations." }}
    Output: {{ "step": "output", "content": "The fs module is one of the core module of Node.js, which provided functions to perform operations on files." }}
    """
    messages = [{ 'role': 'system', 'content': prompt_template },
]

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

        if parsed_output['step'] == 'output':
            print(f"🤖: {parsed_output.get('content')}")
            break
    

