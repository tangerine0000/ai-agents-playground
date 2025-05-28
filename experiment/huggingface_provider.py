import requests
from bs4 import BeautifulSoup
import streamlit as st
import faiss
import numpy as np
import os # Keep os for potential future use, though not strictly needed for this self-contained version

# --- Placeholder/Mock implementations for utils/setup.py ---
# In a real deployment, you would ensure these functions correctly
# initialize actual LLM and embedding models, potentially using API keys
# stored securely in Streamlit Secrets.

class MockLLM:
    """A mock LLM to simulate Langchain's LLM behavior."""
    def invoke(self, prompt: str) -> str:
        # In a real scenario, this would call an actual LLM API (e.g., Gemini, OpenAI)
        # For demonstration, we'll return a generic response.
        # If the prompt contains a question, try to give a relevant-ish answer.
        if "question:" in prompt.lower():
            return "This is a mock AI answer based on the provided context."
        return "Mock LLM response: I received a prompt."

class MockEmbeddings:
    """A mock Embeddings class to simulate Langchain's embeddings behavior."""
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # In a real scenario, this would generate actual embeddings.
        # For demonstration, we'll return dummy vectors of the correct dimension (384 for MiniLM).
        return [np.random.rand(384).tolist() for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        # For demonstration, we'll return a dummy vector.
        return np.random.rand(384).tolist()

def initialize_llm():
    """Initializes a mock LLM."""
    # In a real app, you'd load your LLM here, potentially using st.secrets for API keys.
    # Example: return ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=st.secrets["GOOGLE_API_KEY"])
    return MockLLM()

def initialize_embeddings():
    """Initializes mock embeddings."""
    # In a real app, you'd load your embedding model here.
    # Example: return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return MockEmbeddings()

def setup_page(title: str, description: str):
    """Sets up the Streamlit page configuration."""
    st.set_page_config(page_title=title, layout="wide")
    st.title(title)
    st.markdown(f"<p style='font-size: 1.1em;'>{description}</p>", unsafe_allow_html=True)

# --- End of Placeholder/Mock implementations ---


# Initialize components
llm = initialize_llm()
embeddings = initialize_embeddings()
setup_page(
    title="AI Web Scraper with FAISS",
    description="Enter a website URL and ask questions about its content!"
)

# Initialize FAISS Vector Database
# Ensure the dimension matches your actual embedding model (MiniLM-L6-v2 is 384)
index = faiss.IndexFlatL2(384)
vector_store = {} # Stores (url, texts) tuples, indexed by integer ID

def scrape_website(url: str) -> str:
    """
    Scrape content from a website.

    Args:
        url (str): The URL to scrape

    Returns:
        str: The scraped content or error message
    """
    try:
        st.info(f"Attempting to scrape: {url}")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        # Add a more robust timeout and error handling for requests
        response = requests.get(url, headers=headers, timeout=15) 
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        st.success(f"Successfully fetched {url} (Status code: {response.status_code})")

        # Extract text content
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Prioritize main content areas, or fall back to paragraphs
        # This is a basic attempt; real-world scraping might need more specific selectors
        main_content = soup.find("main") or soup.find("article") or soup.find("body")
        if main_content:
            paragraphs = main_content.find_all("p")
        else:
            paragraphs = soup.find_all("p")

        text = " ".join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])

        if not text:
            st.warning("No significant text content found on the page (after stripping).")
            # Fallback: try to get all text from the body
            text = soup.get_text(separator=' ', strip=True)
            if not text:
                return "No text content found on the page."
            
        # Limit the text length to avoid overwhelming the embedding model or LLM
        # Adjust this limit based on your use case and model capabilities
        return text[:5000] # Increased limit for potentially more content
    except requests.exceptions.Timeout:
        st.error("Error: Request timed out. The website took too long to respond.")
        return "Error: Request timed out. Please try again with a stable connection or a different URL."
    except requests.exceptions.HTTPError as e:
        st.error(f"Error: HTTP request failed for {url} - Status code: {e.response.status_code}")
        return f"Error: Failed to fetch the website (HTTP Status {e.response.status_code}). This might be due to the website blocking automated requests or an issue with the URL."
    except requests.exceptions.ConnectionError as e:
        st.error(f"Error: Connection to {url} failed. This often means the website is blocking the connection or is unreachable.")
        return f"Error: Failed to establish a connection to the website. Details: {str(e)}. This could be due to network issues or the website blocking access."
    except requests.exceptions.RequestException as e:
        st.error(f"Error: An unexpected request error occurred while fetching {url} - {str(e)}")
        return f"Error: An unexpected error occurred during the request - {str(e)}"
    except Exception as e:
        st.error(f"Error: An unexpected error occurred during scraping - {str(e)}")
        return f"Error: An unexpected error occurred during scraping - {str(e)}"

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
        st.info("Splitting text and generating embeddings...")

        # Split text into chunks
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        texts = splitter.split_text(text)

        if not texts:
            st.warning("No text chunks generated after splitting.")
            return "Error: No text chunks generated. The content might be too short or un-splittable."

        # Convert text into embeddings
        # Ensure embeddings.embed_documents returns a list of lists of floats
        vectors = embeddings.embed_documents(texts)
        vectors_np = np.array(vectors, dtype=np.float32)

        # Check if vectors_np has the correct shape before adding to FAISS
        if vectors_np.shape[1] != index.d:
            st.error(f"Embedding dimension mismatch! Expected {index.d}, got {vectors_np.shape[1]}.")
            return "Error: Embedding dimension mismatch. Check your embedding model and FAISS index dimension."

        # Store in FAISS
        index.add(vectors_np)
        
        # Store metadata (URL and original texts) corresponding to each chunk
        # We need to ensure that the index in vector_store maps to the correct chunk in FAISS
        # A simple way for a single URL is to store all chunks under one entry, or
        # map each chunk's FAISS index to its text and URL.
        # For simplicity here, we'll just store the URL and all texts associated with it.
        # If you want to retrieve specific chunks, you'd need a more complex mapping.
        
        # For this example, let's map FAISS index to original text and URL
        current_faiss_size = index.ntotal
        for i, chunk_text in enumerate(texts):
            # Map the FAISS index to the chunk and its URL
            # This assumes FAISS adds vectors sequentially
            vector_store[current_faiss_size - len(texts) + i] = (url, chunk_text)


        st.success(f"Successfully stored {len(texts)} chunks from {url} in FAISS.")
        return f"Successfully processed and stored content from {url}."
    except Exception as e:
        st.error(f"Error storing data in FAISS: {str(e)}")
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
        if index.ntotal == 0: # Check if any vectors are in the FAISS index
            return "No data has been stored yet. Please scrape a website first."

        st.info("Searching FAISS for relevant content...")
        # Convert query into embedding
        query_vector = np.array(embeddings.embed_query(query), dtype=np.float32).reshape(1, -1)

        # Search FAISS
        # k=2 means retrieve the 2 most similar chunks
        D, I = index.search(query_vector, k=min(2, index.ntotal)) # Ensure k doesn't exceed total vectors

        context = ""
        retrieved_urls = set()
        for idx in I[0]:
            if idx in vector_store:
                url, chunk_text = vector_store[idx]
                context += chunk_text + "\n\n"
                retrieved_urls.add(url)

        if not context:
            st.warning("No relevant data found for your question in the stored content.")
            return "No relevant data found for your question."

        st.info("Generating answer using the LLM...")
        # Generate answer
        prompt = f"""Based on the following context, answer the question.
        If the context doesn't contain enough information to answer the question,
        say so clearly.

        Context:
        {context}

        Question: {query}

        Answer:"""
        
        answer = llm.invoke(prompt)
        
        if retrieved_urls:
            answer += "\n\n**Source(s):**\n" + "\n".join(retrieved_urls)

        return answer
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        return f"Error generating answer: {str(e)}"

def main():
    """Main application logic."""
    # URL input and scraping
    url = st.text_input("Enter website URL (e.g., https://www.example.com):")
    if url:
        # Basic URL validation
        if not (url.startswith("http://") or url.startswith("https://")):
            st.error("Please enter a valid URL starting with http:// or https://")
            return

        with st.spinner("Processing website... This might take a moment."):
            content = scrape_website(url)

            if content.startswith("Error"):
                # Error message already handled by scrape_website
                pass 
            else:
                store_message = store_in_faiss(content, url)
                st.write(store_message)

    # Q&A interface
    st.divider()
    query = st.text_input("Ask a question based on the stored content:")
    if query:
        with st.spinner("Generating answer..."):
            answer = retrieve_and_answer(query)
            st.subheader("AI Answer:")
            st.write(answer)

if __name__ == "__main__":
    main()
