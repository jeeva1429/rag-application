# rag_app.py
import os
import json
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
# from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from google import genai
from openai import OpenAI



# Load environment variables from .env file
load_dotenv()
OPEN_AI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in .env file")



# Path to your PDF
FILE_PATH = "C:\\Users\\parkk\\Downloads\\sample_pdf.pdf"
PERSIST_DIR = "./chroma_langchain_db"
COLLECTION_NAME = "pdf_docs"


print("Loading and splitting PDF...")

# Load the PDF file
loader = PyPDFLoader(FILE_PATH)
documents = loader.load()

# Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100, add_start_index=True
)
splits = text_splitter.split_documents(documents)


print("Setting up Chroma vector store...")
# Create or load the Chroma vector store
# embeddings = ''
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001",api_key=os.getenv("GOOGLE_API_KEY"))

# Check if the persistent directory exists
if not os.path.exists(PERSIST_DIR):
    print("Creating new embeddings (first run)...")
    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )
    print("Embeddings created and saved locally.")
else:
    print("Loading existing vector store...")
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

# Function to get relevant passages
def get_relevant_passage(query, k=3):
    """Retrieve top-k relevant passages from the vector store"""
    results = vector_store.similarity_search(query, k=k)
    # print(results[0].metadata)
    return " ".join([r.page_content for r in results])

# Function to create RAG prompt
def make_rag_prompt(query, relevant_passage):
    """Constructs a contextual, friendly RAG prompt"""
    formatted_passage = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = f"""
            You are a helpful and informative assistant that answers questions using the passage below.
            The one who asks the question is not very technical, so please keep your answers clear and simple. 
            Respond in a clear and concise manner and make it at least two sentences. 
            If the passage does not contain the answer, say you don’t know.

            QUESTION: '{query}'
            PASSAGE: '{formatted_passage}'

            ANSWER:
            """
    return prompt.strip()
query = "which is this git in this document?"

# RAG process
relevant_passage = get_relevant_passage(query, k=3)
prompt = make_rag_prompt(query, relevant_passage)

# print(" Query:", query)
# print(" Retrieved Passage:\n", relevant_passage[:400], "...")


# Generate answer using Gemini LLM
def generate_respose(prompt):
    client = genai.Client()
    # client = OpenAI()
    try:
        # response = client.responses.create(model="gpt-5-nano", input=prompt)
        # return response.output_text
        response  = client.models.generate_content(model="gemini-2.0-flash",contents=prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

# print("RAG process completed.")
