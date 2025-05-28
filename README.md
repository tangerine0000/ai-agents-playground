# AI Agents Playground

A collection of AI-powered applications built with Streamlit, LangChain, and Ollama.

## Applications

### 1. AI Chatbot with Memory
A conversational AI chatbot that remembers the conversation history.
- **File**: `basic_agents/basic_ai_agents.py`
- **Agents**: 
  - Conversation Agent: Handles chat interactions with memory
  - Memory Agent: Manages conversation history
  - Response Agent: Generates contextual responses
- **Deployment**: [AI Chatbot with Memory](https://tan-chatbot-agents.streamlit.app/)

### 2. AI-Powered Web Scraper
A web scraper that can extract and summarize content from any website.
- **File**: `web_scraper_agents/ai_web_scraper.py`
- **Agents**:
  - Scraping Agent: Extracts content from websites
  - Summarization Agent: Generates concise summaries
  - Content Analysis Agent: Analyzes and processes web content
- **Deployment**: [AI Web Scraper](https://tan-summarizer-web-agents.streamlit.app/)

### 3. AI Web Scraper with FAISS
An advanced web scraper that stores content in a FAISS vector database for semantic search and Q&A.
- **File**: `web_scraper_agents/ai_web_scraper_faiss.py`
- **Agents**:
  - Scraping Agent: Extracts content from websites
  - Vector Storage Agent: Manages FAISS vector database
  - Semantic Search Agent: Performs similarity searches
  - Q&A Agent: Answers questions based on stored content
- **Deployment**: [AI Web Scraper with FAISS](https://tan-chatbot-web-scraper-agents.streamlit.app/)

## Features

- **AI Chatbot**: Conversational AI with memory using LangChain and Ollama
- **Web Scraping**: Content extraction and summarization
- **Vector Storage**: FAISS-based semantic search and Q&A
- **Streamlit UI**: User-friendly interface for all applications

## Requirements

- Python 3.10
- Streamlit
- LangChain
- Ollama
- FAISS
- BeautifulSoup4
- Other dependencies listed in `requirements.txt`

### Ollama Setup (Linux)

1. Install Ollama:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

2. Install a model (example using Gemma):
```bash
ollama pull gemma:1b
```

3. Start Ollama server:
```bash
ollama serve
```

Keep the Ollama server running in a separate terminal while using the applications.

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
# Grant execute permission to scripts
chmod +x scripts/*.sh

# Run applications using scripts (with default model gemma3:1b)
./scripts/run_chat_agents.sh
# or
./scripts/run_web_scraper_basic.sh
# or
./scripts/run_web_scraper_faiss.sh

# Run with a custom model
./scripts/run_chat_agents.sh -m <your_model_name>
# or
./scripts/run_web_scraper_basic.sh -m <your_model_name>
# or
./scripts/run_web_scraper_faiss.sh -m <your_model_name>

# Alternatively, you can run directly with streamlit
streamlit run basic_agents/chat_agents.py -- -m <your_model_name>
# or
streamlit run web_scraper_agents/ai_web_scraper.py -- -m <your_model_name>
# or
streamlit run web_scraper_agents/ai_web_scraper_faiss.py -- -m <your_model_name>
```

Note: All applications use "gemma3:1b" as the default model. You can specify a different model using the `-m` or `--model` option. Make sure the model is installed in Ollama before using it.

## Deployment

The applications are deployed on Streamlit Cloud:
1. [AI Chatbot with Memory](https://tan-chatbot-agents.streamlit.app/)
2. [AI Web Scraper](https://tan-summarizer-web-agents.streamlit.app/)
3. [AI Web Scraper with FAISS](https://tan-chatbot-web-scraper-agents.streamlit.app/)

## License

MIT License