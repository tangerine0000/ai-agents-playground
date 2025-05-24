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
        llm = Ollama(
            base_url=os.getenv("OLLAMA_API_BASE", "http://localhost:11434"),
            model="llama2"
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None

def initialize_embeddings():
    """Initialize the embeddings model."""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        return embeddings
    except Exception as e:
        st.error(f"Error initializing embeddings: {str(e)}")
        return None

def setup_page():
    """Setup the Streamlit page configuration."""
    st.set_page_config(
        page_title="AI Web Scraper",
        page_icon="🤖",
        layout="wide"
    )
    st.title("AI Web Scraper") 