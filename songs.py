import streamlit as st
import chromadb
import google.generativeai as genai
from chromadb.utils import embedding_functions
from datetime import datetime
import csv

# Configure Gemini API
API_KEY = ""  # Replace with your actual API key
genai.configure(api_key=API_KEY)

# Use GoogleGenerativeAiEmbeddingFunction for embedding
google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=API_KEY)

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")  # Persistent storage

# Create or get a collection
collection = client.get_or_create_collection(
    name="songs_collection",
    embedding_function=google_ef,
    metadata={"description": "Song lyrics storage", "created": str(datetime.now())}
)

# Load data from CSV file
documents = []
with open("songs.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file, fieldnames=["song", "album", "artist", "lyrics"])
    for row in reader:
        documents.append(row)

# Extract lyrics for embedding
lyrics_list = [doc["lyrics"] for doc in documents]

# Add documents to ChromaDB (embed only the lyrics)
collection.add(
    documents=lyrics_list,  # Use only lyrics for embedding
    metadatas=documents,    # Store full metadata for retrieval
    ids=[str(i) for i in range(len(documents))]
)

# Streamlit App
st.title("Song Lyrics Search Engine 🎵")
st.write("Enter a query to find relevant songs based on their lyrics.")

# Input query from user
query = st.text_input("Enter your search query:")

if query:
    # Retrieve top 3 most relevant documents
    results = collection.query(
        query_texts=[query],  # Use query_texts parameter
        n_results=3
    )

    st.subheader("Top Matches:")
    for i, match in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i]  # Retrieve metadata for the match
        st.write(f"**{i+1}. Song:** {metadata['song']}")
        st.write(f"   **Artist:** {metadata['artist']}")
        st.write(f"   **Lyrics Snippet:** {match[:200]}...")  # Show a snippet of the lyrics
        st.write("---")