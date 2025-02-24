


import chromadb
import google.generativeai as genai
from chromadb.utils import embedding_functions
from datetime import datetime

# Configure Gemini API
API_KEY = ""  # Insert your API key here
genai.configure(api_key=API_KEY)

# Use GoogleGenerativeAiEmbeddingFunction for embedding
google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=API_KEY)

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")  # Persistent storage

# Create or get a collection
collection = client.get_or_create_collection(
    name="my_collection",
    embedding_function=google_ef,
    metadata={"description": "Chat memory storage", "created": str(datetime.now())}
)

# Sample documents
documents = [
    "Machine learning is fascinating.",
    "Deep learning is a subset of machine learning.",
    "Artificial intelligence is transforming the world.",
    "The stock market is unpredictable.",
    "Investing in AI stocks can be profitable."
]

# Add documents to ChromaDB
# Let the embedding_function handle the embedding process internally
collection.add(
    documents=documents,
    ids=[str(i) for i in range(len(documents))]
)

# Query ChromaDB
query = "What is stocks prediction?"

# Retrieve top 2 most relevant documents
results = collection.query(
    query_texts=[query],  # Use query_texts parameter
    n_results=2
)

print("Top Matches:", results["documents"])