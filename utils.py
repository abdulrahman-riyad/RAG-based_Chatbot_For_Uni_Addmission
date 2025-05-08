import logging
import ollama
import streamlit as st
from config import LOG_LEVEL, LOG_FORMAT

def setup_logging():
    """Configures the application-wide logging."""
    logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO), format=LOG_FORMAT)

def validate_model_available(model_name: str) -> bool:
    """
    Validates if the specified Ollama model is available.
    Displays an error in Streamlit if not available.
    """
    try:
        ollama.show(model_name)
        logging.info(f"Model '{model_name}' is available.")
        return True
    except Exception as e:
        error_message = f"Error: Ollama model '{model_name}' not found or Ollama service is not running. Please ensure Ollama is active and the model is pulled (e.g., `ollama pull {model_name}`). Details: {e}"
        logging.error(error_message)
        st.error(error_message)
        return False