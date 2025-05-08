import os
import logging
import streamlit as st
from langchain_community.vectorstores import Chroma
import ollama

from config import PERSIST_DIRECTORY, VECTOR_STORE_NAME, DOC_PATH, EMBEDDING_MODEL
from data_handler import ingest_pdf, split_documents
from llm_services import get_embedding_model


@st.cache_resource
def load_or_create_vector_db():
    """
    Loads an existing vector database or creates a new one if not found.
    Uses a cached embedding function.
    Returns the Chroma vector database instance or None on failure.
    """
    embedding_function = get_embedding_model()
    if embedding_function is None:
        st.error("Embedding model failed to initialize. Cannot proceed with Vector DB.")
        return None

    if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
        logging.info(f"Loading existing vector database from '{PERSIST_DIRECTORY}'.")
        try:
            vector_db = Chroma(
                embedding_function=embedding_function,
                collection_name=VECTOR_STORE_NAME,
                persist_directory=PERSIST_DIRECTORY,
            )
            logging.info("Successfully loaded vector database.")
            return vector_db
        except Exception as e:
            logging.error(f"Error loading existing vector database: {e}. Will attempt to recreate.")

    logging.info(f"Creating new vector database in '{PERSIST_DIRECTORY}'.")

    # Ensure document exists before attempting to create DB
    if not os.path.exists(DOC_PATH):
        st.error(f"Document not found at {DOC_PATH}. Cannot create vector DB.")
        logging.error(f"Document not found at {DOC_PATH}. Vector DB creation aborted.")
        return None

    # Attempt to pull the embedding model if creating DB
    try:
        logging.info(f"Ensuring embedding model '{EMBEDDING_MODEL}' is pulled for new DB creation...")
        ollama.pull(EMBEDDING_MODEL)
    except Exception as e:
        logging.warning(f"Could not explicitly pull '{EMBEDDING_MODEL}': {e}. Proceeding, assuming it's available.")

    documents = ingest_pdf(DOC_PATH)
    if not documents:
        st.error("Failed to ingest PDF. Cannot create vector database.")
        return None

    chunks = split_documents(documents)
    if not chunks:
        st.error("Failed to split documents into chunks. Cannot create vector database.")
        logging.error("No chunks were created from the document. Cannot create vector database.")
        return None

    try:
        os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_function,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        logging.info("Vector database created and persisted successfully.")
        return vector_db
    except Exception as e:
        logging.error(f"Failed to create vector database: {e}")
        st.error(f"An error occurred while creating the vector database: {e}")
        return None