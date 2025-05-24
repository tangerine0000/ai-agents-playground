import os
import streamlit as st
from langchain_ollama import OllamaLLM
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
    """Initialize the LLM model."""
    try:
        return OllamaLLM(model="gemma3:1b")
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        st.error("Please make sure Ollama is running and the model is installed")
        st.stop()

def initialize_embeddings():
    """Initialize the embeddings model."""
    try:
        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception as e:
        st.error(f"Error loading embeddings model: {str(e)}")
        st.error("Please make sure you have installed the required packages: pip install sentence-transformers")
        st.stop()

def setup_page(title, description):
    """Setup the Streamlit page with title and description."""
    try:
        st.set_page_config(
            page_title=title,
            layout="wide",
            initial_sidebar_state="collapsed"
        )
        st.title(title)
        st.write(description)
    except Exception as e:
        st.error(f"Error setting up page: {str(e)}")
        st.stop() 