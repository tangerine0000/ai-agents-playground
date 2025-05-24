# AI Agents Playground

A collection of AI-powered applications built with Streamlit, LangChain, and Ollama.

## Applications

### 1. AI Chatbot with Memory
A conversational AI chatbot that remembers the conversation history.
- **Local Run**: `streamlit run basic_agents/basic_ai_agents.py`
- **Deployment**: [AI Chatbot with Memory](https://tan-chatbot-agents.streamlit.app/)

### 2. AI-Powered Web Scraper
A web scraper that can extract and summarize content from any website.
- **Local Run**: `streamlit run web_scraper_agents/ai_web_scraper.py`
- **Deployment**: [AI Web Scraper](https://tan-summarizer-web-agents.streamlit.app/)

### 3. AI Web Scraper with FAISS
An advanced web scraper that stores content in a FAISS vector database for semantic search and Q&A.
- **Local Run**: `streamlit run web_scraper_agents/ai_web_scraper_faiss.py`
- **Deployment**: [AI Web Scraper with FAISS](https://tan-chatbot-web-scraper-agents.streamlit.app/)

## Features

- **AI Chatbot**: Conversational AI with memory using LangChain and Ollama
- **Web Scraping**: Content extraction and summarization
- **Vector Storage**: FAISS-based semantic search and Q&A
- **Streamlit UI**: User-friendly interface for all applications

## Requirements

- Python 3.10+
- Streamlit
- LangChain
- Ollama
- FAISS
- BeautifulSoup4
- Other dependencies listed in `requirements.txt`

## Local Development

1. Clone the repository:
```bash
git clone https://github.com/tangerine0000/ai-agents-playground.git
cd ai-agents-playground
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run any of the applications:
```bash
streamlit run basic_agents/basic_ai_agents.py
# or
streamlit run web_scraper_agents/ai_web_scraper.py
# or
streamlit run web_scraper_agents/ai_web_scraper_faiss.py
```

## Environment Variables

Set these environment variables for local development:
```
OLLAMA_API_BASE = "http://localhost:11434"  # Change this to your Ollama endpoint
TOKENIZERS_PARALLELISM = "false"
CUDA_VISIBLE_DEVICES = ""
```

## Deployment

The applications are deployed on Streamlit Cloud:
1. [AI Chatbot with Memory](https://tan-chatbot-agents.streamlit.app/)
2. [AI Web Scraper](https://tan-summarizer-web-agents.streamlit.app/)
3. [AI Web Scraper with FAISS](https://tan-chatbot-web-scraper-agents.streamlit.app/)

## License

MIT License 