import requests
from bs4 import BeautifulSoup
import streamlit as st
import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.setup import initialize_llm, setup_page

# Initialize components
llm = initialize_llm()
setup_page(
    title="AI-Powered Web Scraper",
    description="Enter a website URL below and get a summarized version!"
)

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

def summarize_content(content: str) -> str:
    """
    Summarize content using AI.
    
    Args:
        content (str): The content to summarize
        
    Returns:
        str: The AI-generated summary
    """
    try:
        st.write("Summarizing content...")
        prompt = f"Summarize the following content in a clear and concise way:\n\n{content[:1000]}"
        return llm.invoke(prompt)
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def main():
    """Main application logic."""
    url = st.text_input("Enter website URL:")
    if url:
        with st.spinner("Processing..."):
            content = scrape_website(url)

            if content.startswith("Error") or content.startswith("Failed"):
                st.error(content)
            else:
                summary = summarize_content(content)
                st.subheader("Website Summary")
                st.write(summary)

if __name__ == "__main__":
    main()