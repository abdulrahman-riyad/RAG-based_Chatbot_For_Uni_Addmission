import streamlit as st

CHAT_UI_STYLES = """
    <style>
    .stApp {
        /* Add global styles for the app if needed */
    }
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding-bottom: 100px; /* Increased space for the fixed input box */
    }
    .chat-message {
        padding: 10px;
        margin: 8px 0;
        border-radius: 8px;
        width: fit-content;
        max-width: 70%;
        border: 1px solid transparent; /* Base border */
    }
    .user-message {
        background-color: #DCF8C6;
        align-self: flex-end;
        margin-left: auto; /* Aligns to the right */
        border-color: #c5e0b4; /* Slightly darker border for user */
    }
    .assistant-message {
        background-color: #F1F0F0;
        align-self: flex-start;
        margin-right: auto; /* Aligns to the left */
        border-color: #dcdcdc; /* Slightly darker border for assistant */

    }
    .chat-box {
        display: flex;
        flex-direction: column-reverse; /* Newest messages at the bottom */
        height: calc(100vh - 250px); /* Adjust height dynamically, considering header and input */
        min-height: 300px; /* Minimum height */
        overflow-y: auto; /* Scroll for older messages */
        border: 1px solid #ddd;
        padding: 10px;
        border-radius: 8px;
        background-color: #ffffff;
    }
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        padding: 15px 10px; /* Increased padding */
        background-color: #f8f9fa; /* Light background for input area */
        border-top: 1px solid #dee2e6; /* Clearer border */
        box-shadow: 0 -2px 5px rgba(0,0,0,0.05); /* Subtle shadow */
        z-index: 1000;
    }
    .input-area {
        display: flex;
        gap: 10px; /* Space between text input and button */
        align-items: center; /* Vertically align items */
        max-width: 800px; /* Match chat container width */
        margin: 0 auto; /* Center the input area */
    }
    .input-area > div[data-testid="stTextInput"] { /* Target Streamlit's text input parent div */
        flex-grow: 1;
    }
    .stButton > button { /* Style Streamlit buttons */
        width: auto; /* Adjust width based on content */
        padding: 0.5em 1em;
    }
    </style>
"""


def apply_ui_styles():
    """Applies the CSS styles to the Streamlit app."""
    st.markdown(CHAT_UI_STYLES, unsafe_allow_html=True)


def display_chat_messages():
    """
    Displays chat messages in a scrollable, bottom-aligned chat box.
    Assumes st.session_state.chat_history exists.
    """
    st.markdown('<div class="chat-box">', unsafe_allow_html=True)
    for msg in st.session_state.get("chat_history", []):
        content = msg["content"].replace("<", "<").replace(">", ">")
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-message user-message">{content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message assistant-message">{content}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)