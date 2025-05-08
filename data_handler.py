import os
import logging
import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


def ingest_pdf(doc_path: str):
    """
    Loads the PDF document using UnstructuredPDFLoader.
    Returns documents or None if an error occurs.
    """
    if not os.path.exists(doc_path):
        logging.error(f"PDF file not found at path: {doc_path}")
        st.error(f"PDF file not found at '{doc_path}'. Please ensure it exists.")
        return None
    try:
        loader = UnstructuredPDFLoader(file_path=doc_path)
        data = loader.load()
        logging.info(f"PDF '{doc_path}' loaded successfully. {len(data)} document(s) found.")
        return data
    except Exception as e:
        logging.error(f"Failed to load PDF '{doc_path}': {e}")
        st.error(f"Failed to load PDF: {e}")
        return None


def split_documents(documents):
    """
    Splits the documents into chunks for vector storage.
    Returns a list of chunks or None if input is invalid.
    """
    if not documents:
        logging.warning("No documents provided to split.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Documents split into {len(chunks)} chunks.")
    if not chunks:
        logging.warning("Splitting documents resulted in no chunks.")
        return None
    return chunks