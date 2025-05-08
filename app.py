# app.py

import streamlit as st
import logging
import os
import nltk

try:
    nltk.download("punkt", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
except Exception as e:
    st.warning(
        f"NLTK download failed: {e}. Some functionalities might be affected if NLTK is strictly required by a component.")

# Import project modules
import config
from config import MODEL_NAME, EMBEDDING_MODEL, DOC_PATH
from utils import setup_logging, validate_model_available
from llm_services import get_llm, get_rag_chain
from vector_db_manager import load_or_create_vector_db
from ui_components import apply_ui_styles, display_chat_messages

# --- Initial Setup ---
setup_logging()  # Configure logging
apply_ui_styles()  # Apply CSS styles


# --- Main Application Logic ---
def main():
    st.title("📖 Beta E-JUST RAG Chatbot (Modular)")

    # Initial model and document validation
    models_validated = validate_model_available(MODEL_NAME) and \
                       validate_model_available(EMBEDDING_MODEL)

    doc_exists = os.path.exists(DOC_PATH)
    if not doc_exists:
        st.error(
            f"CRITICAL: Document PDF file not found at '{DOC_PATH}'. The application cannot create a new vector database without it.")
        logging.error(f"CRITICAL: Document PDF file not found at '{DOC_PATH}'.")

    if not models_validated:
        st.warning(
            "One or more Ollama models are not available or Ollama service is down. Please check your Ollama setup.")

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Load core components (cached)
    rag_chain = None
    if models_validated:
        try:
            llm = get_llm()
            vector_db = load_or_create_vector_db()

            if vector_db is None:
                st.error("Failed to load or create the vector database. Chat functionality will be limited.")
            elif llm is None:
                st.error("LLM failed to initialize.")
            else:
                from llm_services import get_retriever
                retriever = get_retriever(vector_db, llm)
                if retriever:
                    rag_chain = get_rag_chain(retriever, llm)
                else:
                    st.error("Failed to create the document retriever. Chat functionality will be impaired.")

        except Exception as e:
            logging.error(f"Error initializing core RAG components: {e}", exc_info=True)
            st.error(f"An critical error occurred during RAG component initialization: {e}")

    # --- UI Layout ---

    # Chat display area
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    display_chat_messages()
    st.markdown('</div>', unsafe_allow_html=True)

    # Input area fixed at the bottom
    st.markdown('<div class="input-container"><div class="input-area">', unsafe_allow_html=True)

    # Use a form for the input and button to prevent rerun on text input key press
    with st.form(key="chat_input_form", clear_on_submit=True):
        user_input = st.text_input(
            "Your message:",
            key="user_input_text",
            label_visibility="collapsed",
            placeholder="Type your message here..."
        )
        send_button = st.form_submit_button(label="Send")

    clear_button = st.button("Clear Chat")

    st.markdown('</div></div>', unsafe_allow_html=True)  # Close input-area and input-container

    # --- Event Handling ---
    if clear_button:
        st.session_state.chat_history = []
        logging.info("Chat history cleared.")
        st.rerun()

    if send_button and user_input:
        if not models_validated:
            st.warning("Models are not validated. Cannot process message.")
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.chat_history.append({"role": "assistant",
                                                  "content": "I cannot process your request as the required AI models are not available. Please check the setup."})
            st.rerun()
            return

        if rag_chain is None:
            st.error("The RAG chain is not initialized. Cannot process message.")
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.chat_history.append({"role": "assistant",
                                                  "content": "I'm sorry, but I'm not properly initialized to answer questions. Please check the application logs."})
            st.rerun()
            return

        st.session_state.chat_history.append({"role": "user", "content": user_input})
        logging.info(f"User input: {user_input}")

        with st.spinner("Generating response..."):
            try:
                response = rag_chain.invoke(user_input)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                logging.info(f"Assistant response: {response}")
            except Exception as e:
                logging.error(f"Error during RAG chain invocation: {e}", exc_info=True)
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                st.error(error_msg)
        st.rerun()
    elif send_button and not user_input:
        st.toast("Please enter a message.", icon="📝")


if __name__ == "__main__":
    if not os.path.exists(DOC_PATH) and not os.path.exists(config.PERSIST_DIRECTORY):
        st.error(
            f"CRITICAL: Document PDF ('{DOC_PATH}') not found and no existing vector database. Please add the PDF to create a new database.")
        logging.critical(f"Document PDF ('{DOC_PATH}') not found and no existing vector database ({config.PERSIST_DIRECTORY}).")

    main()