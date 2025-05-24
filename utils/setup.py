import os
import streamlit as st
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
import torch

# Force PyTorch to use CPU
torch.set_num_threads(1)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# Configure environment
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Disable warnings
import warnings
warnings.filterwarnings('ignore')

def initialize_llm():
    """Initialize the LLM with Ollama."""
    try:
        # Get Ollama API base URL from environment variable
        ollama_base = os.getenv("OLLAMA_API_BASE")
        if not ollama_base:
            st.error("""
            Error: OLLAMA_API_BASE environment variable is not set.
            
            For local development:
            1. Make sure Ollama is running locally
            2. Set OLLAMA_API_BASE=http://localhost:11434
            
            For Streamlit Cloud deployment:
            1. Set OLLAMA_API_BASE to your public Ollama endpoint
            2. Make sure the endpoint is accessible from Streamlit Cloud
            """)
            return None

        # Initialize Ollama
        llm = Ollama(
            base_url=ollama_base,
            model="llama2"
        )
        
        # Test the connection
        try:
            llm.invoke("test")
        except Exception as e:
            st.error(f"""
            Error connecting to Ollama at {ollama_base}:
            {str(e)}
            
            Please check:
            1. The Ollama server is running
            2. The URL is correct
            3. The server is accessible from this environment
            """)
            return None
            
        return llm
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None

def initialize_embeddings():
    """Initialize the embeddings model."""
    try:
        # Set device to CPU explicitly
        device = "cpu"
        
        # Initialize embeddings with specific model parameters
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Force model to CPU
        if hasattr(embeddings, 'client'):
            embeddings.client.to(device)
            
        return embeddings
    except Exception as e:
        st.error(f"Error initializing embeddings: {str(e)}")
        return None

def setup_page(title="AI Web Scraper", description=None):
    """Setup the Streamlit page configuration.
    
    Args:
        title (str): The title of the page
        description (str, optional): A description to display below the title
    """
    st.set_page_config(
        page_title=title,
        page_icon="🤖",
        layout="wide"
    )
    st.title(title)
    if description:
        st.write(description) 