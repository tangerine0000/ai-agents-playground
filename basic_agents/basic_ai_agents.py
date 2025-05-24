import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import initialize_llm, setup_page

# Initialize components
llm = initialize_llm()
setup_page(title="AI Chatbot with Memory", description="Ask me anything! I'll remember our conversation.")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

# Define AI Chat Prompt
CHAT_PROMPT = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""Previous conversation:
{chat_history}

User: {question}

AI: Let me help you with that. """
)

def format_chat_history() -> str:
    """Format the chat history for the prompt."""
    return "\n".join([
        f"{msg.type.capitalize()}: {msg.content}"
        for msg in st.session_state.chat_history.messages
    ])

def run_chain(question: str) -> str:
    """
    Process the user's question and generate a response.
    
    Args:
        question (str): The user's question
        
    Returns:
        str: The AI's response
    """
    try:
        # Get formatted chat history
        chat_history_text = format_chat_history()

        # Generate response
        response = llm.invoke(
            CHAT_PROMPT.format(
                chat_history=chat_history_text,
                question=question
            )
        )

        # Update chat history
        st.session_state.chat_history.add_user_message(question)
        st.session_state.chat_history.add_ai_message(response)

        return response
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        st.error(error_msg)
        return error_msg

def display_chat_history():
    """Display the chat history in the UI."""
    st.subheader("Chat History")
    for msg in st.session_state.chat_history.messages:
        with st.chat_message(msg.type):
            st.write(msg.content)

def main():
    """Main application logic."""
    # Chat input
    user_input = st.chat_input("Type your message here...")
    if user_input:
        with st.spinner("Thinking..."):
            response = run_chain(user_input)
            
        # Display the latest exchange
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("ai"):
            st.write(response)

    # Display chat history
    display_chat_history()

if __name__ == "__main__":
    main()
