import requests
from bs4 import BeautifulSoup
import streamlit as st
import faiss
import numpy as np
import sys
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.setup import initialize_llm, initialize_embeddings, setup_page

# Setup page configuration first
title="AI Web Scraper with FAISS",
description="Enter a website URL and ask questions about its content!"
st.set_page_config(
    page_title=title,
    page_icon="🤖",
    layout="wide"
)
st.title(title)
if description:
    st.write(description) 
# Initialize components
llm = initialize_llm()
embeddings = initialize_embeddings()

# Initialize FAISS Vector Database
index = faiss.IndexFlatL2(384)  # Vector dimension for MiniLM
vector_store = {}

def scrape_website(url: str) -> str:
    """
    Scrape content from a website.
    
    Args:
        url (str): The URL to scrape
        
    Returns:
        str: The scraped content or error message
    """
    try:
        st.write(f"Scraping website: {url}")
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return f"Failed to fetch {url} (Status code: {response.status_code})"
        
        # Extract text content
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])

        if not text:
            return "No text content found on the page."
            
        return text[:2000]
    except requests.exceptions.Timeout:
        return "Error: Request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return f"Error: Failed to fetch the website - {str(e)}"
    except Exception as e:
        return f"Error: An unexpected error occurred - {str(e)}"

def store_in_faiss(text: str, url: str) -> str:
    """
    Store text content in FAISS vector database.
    
    Args:
        text (str): The text content to store
        url (str): The source URL
        
    Returns:
        str: Success or error message
    """
    try:
        st.write("Storing data in FAISS...")

        # Split text into chunks
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        texts = splitter.split_text(text)

        if not texts:
            return "Error: No text chunks generated."

        # Convert text into embeddings
        vectors = embeddings.embed_documents(texts)
        vectors = np.array(vectors, dtype=np.float32)

        # Store in FAISS
        index.add(vectors)
        vector_store[len(vector_store)] = (url, texts)

        return f"Successfully stored {len(texts)} chunks from {url}"
    except Exception as e:
        return f"Error storing data: {str(e)}"

def retrieve_and_answer(query: str) -> str:
    """
    Retrieve relevant content and generate an answer.
    
    Args:
        query (str): The user's question
        
    Returns:
        str: The AI-generated answer
    """
    try:
        if not vector_store:
            return "No data has been stored yet. Please scrape a website first."

        # Convert query into embedding
        query_vector = np.array(embeddings.embed_query(query), dtype=np.float32).reshape(1, -1)

        # Search FAISS
        D, I = index.search(query_vector, k=2)

        context = ""
        for idx in I[0]:
            if idx in vector_store:
                context += " ".join(vector_store[idx][1]) + "\n\n"

        if not context:
            return "No relevant data found for your question."

        # Generate answer
        prompt = f"""Based on the following context, answer the question.
        If the context doesn't contain enough information to answer the question,
        say so clearly.

        Context:
        {context}

        Question: {query}

        Answer:"""
        
        return llm.invoke(prompt)
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def main():
    """Main application logic."""
    # URL input and scraping
    url = st.text_input("Enter website URL:")
    if url:
        with st.spinner("Processing..."):
            content = scrape_website(url)

            if content.startswith("Error") or content.startswith("Failed"):
                st.error(content)
            else:
                store_message = store_in_faiss(content, url)
                st.write(store_message)

    # Q&A interface
    st.divider()
    query = st.text_input("Ask a question based on stored content:")
    if query:
        with st.spinner("Generating answer..."):
            answer = retrieve_and_answer(query)
            st.subheader("AI Answer:")
            st.write(answer)

if __name__ == "__main__":
    main()



