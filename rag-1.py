import os
from pathlib import Path 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

from langchain_qdrant import QdrantVectorStore

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./gen-lang-client-0749002261-05b6a39a97f3.json"


def load_documents(file_path: str):
    loader = PyPDFLoader(file_path=file_path)
    docs = loader.load()
    return docs

# pdf_path = Path(__file__).parent / "nodejs.pdf"

def text_splitter(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    split_docs = text_splitter.split_documents(documents=docs)
    return split_docs

def embedding_model():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )
    return embeddings
embeddings = embedding_model()
# print("text_splitter", text_splitter)


# print("Split Docs", split_docs)

# embedder = OpenAIEmbeddings(
#     model="text-embedding-3-large",
#     api_key=""
# )

def store_documents_in_qdrant(split_docs):
    vector_store = QdrantVectorStore.from_documents(
        documents=split_docs,
        url="http://localhost:6333",
        collection_name="learning_langchain",
        embedding=embeddings
    )
    return vector_store

# vector_store = QdrantVectorStore.from_documents(
#     documents=[],
#     url="http://localhost:6333",
#     collection_name="learning_langchain",
#     embedding=embeddings
# )

# vector_store.add_documents(documents=split_docs)
# print("Injection Done")

retriver = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_langchain",
    embedding=embeddings
)

search_result = retriver.similarity_search(
    query="What is FS Module?"
)

print("Relevant Chunks", search_result)