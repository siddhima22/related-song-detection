import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import google.generativeai as genai

# Function to load CSV files
def load_csv(file):
    return pd.read_csv(file)

# Function to calculate cosine similarity between rows
def compute_cosine_similarity(df1, df2):
    # Ensure both dataframes are aligned
    df1, df2 = df1.align(df2, axis=1, join='inner')
    
    # Convert the data to text for similarity calculation (if necessary)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix1 = vectorizer.fit_transform(df1.astype(str).apply(' '.join, axis=1))
    tfidf_matrix2 = vectorizer.transform(df2.astype(str).apply(' '.join, axis=1))

    # Compute cosine similarity
    similarity = cosine_similarity(tfidf_matrix1, tfidf_matrix2)
    return similarity

# Function to generate a financial narrative using Google's Generative AI (Gemini model)
def generate_financial_narrative(data):
    prompt = f"Analyze the following financial data and provide a narrative summary of the company's financial health:\n\n{data}"
    
    # Set the Google API key (replace with your own API key or set environment variable)
    genai.configure(api_key="")
    
    # Call the Google Generative AI model (Gemini)
    response = genai.generate_text(
        model="gemini-flash-1.5",  # Specify the model version you want to use
        prompt=prompt,
        temperature=0.7,
        max_output_tokens=500
    )
    
    return response['text'].strip()

# Streamlit UI
st.title('Financial Data Comparison and Narrative Generator')

# File upload for CSV 1
uploaded_file1 = st.file_uploader("Upload the first CSV file", type="csv")
if uploaded_file1:
    df1 = load_csv(uploaded_file1)
    st.write("First CSV data:", df1.head())

# File upload for CSV 2
uploaded_file2 = st.file_uploader("Upload the second CSV file", type="csv")
if uploaded_file2:
    df2 = load_csv(uploaded_file2)
    st.write("Second CSV data:", df2.head())

# Check if both files are uploaded
if uploaded_file1 and uploaded_file2:
    # Compare the data using cosine similarity
    similarity_matrix = compute_cosine_similarity(df1, df2)
    
    st.write("Cosine Similarity between corresponding rows in both CSVs:")
    st.write(similarity_matrix)
    
    # Generate financial narrative
    data_summary = df1.head(5).to_string(index=False)  # Just an example of data for the narrative
    financial_narrative = generate_financial_narrative(data_summary)
    
    st.write("Generated Financial Narrative:")
    st.write(financial_narrative)
