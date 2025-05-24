# AI Agents Collection

A collection of AI-powered applications built with Streamlit.

## Applications

1. **Basic AI Chat** (`basic_agents/basic_ai_agents.py`)
   - Chat interface with memory
   - Remembers conversation history
   - Simple Q&A functionality

2. **Basic Web Scraper** (`web_scraper_agents/ai_web_scraper.py`)
   - URL input
   - Website content extraction
   - AI-generated summary

3. **FAISS Web Scraper** (`web_scraper_agents/ai_web_scraper_faiss.py`)
   - URL input
   - Content storage in vector database
   - Q&A based on stored content
   - Semantic search capabilities

## Deployment Instructions

1. Fork this repository to your GitHub account
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Sign in with your GitHub account
4. Click "New app"
5. Select your forked repository
6. For each app, create a new deployment with these settings:
   - Main file path: `basic_agents/basic_ai_agents.py` (or the respective file)
   - Python version: 3.10
   - Requirements file: `requirements.txt`

## Environment Variables

Set these environment variables in Streamlit Cloud:
- `OLLAMA_API_BASE`: Your Ollama API endpoint
- `TOKENIZERS_PARALLELISM`: "false"
- `CUDA_VISIBLE_DEVICES`: ""

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run locally:
```bash
streamlit run basic_agents/basic_ai_agents.py
streamlit run web_scraper_agents/ai_web_scraper.py
streamlit run web_scraper_agents/ai_web_scraper_faiss.py
``` 