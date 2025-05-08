import logging
import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

from config import MODEL_NAME, EMBEDDING_MODEL


@st.cache_resource
def get_llm():
    """Initializes and returns the ChatOllama LLM instance."""
    logging.info(f"Initializing LLM: {MODEL_NAME}")
    return ChatOllama(model=MODEL_NAME)


@st.cache_resource
def get_embedding_model():
    """Initializes and returns the OllamaEmbeddings instance."""
    logging.info(f"Initializing Embedding Model: {EMBEDDING_MODEL}")
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


@st.cache_resource
def get_retriever(_vector_db, _llm):  # Underscore to signify not to re-run if object ID changes
    """
    Creates a MultiQueryRetriever.
    _vector_db: The cached vector database instance.
    _llm: The cached LLM instance.
    """
    if _vector_db is None or _llm is None:
        logging.error("Vector DB or LLM is None, cannot create retriever.")
        st.error("Failed to initialize components needed for the retriever.")
        return None

    logging.info("Creating MultiQueryRetriever.")
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents from
a vector database. By generating multiple perspectives on the user question, your
goal is to help the user overcome some of the limitations of the distance-based
similarity search. Provide these alternative questions separated by newlines.
Original question: {question}""",
    )
    retriever = MultiQueryRetriever.from_llm(
        _vector_db.as_retriever(), _llm, prompt=QUERY_PROMPT
    )
    return retriever


@st.cache_resource
def get_rag_chain(_retriever, _llm):
    """
    Creates the RAG chain.
    _retriever: The cached retriever instance.
    _llm: The cached LLM instance.
    """
    if _retriever is None or _llm is None:
        logging.error("Retriever or LLM is None, cannot create RAG chain.")
        st.error("Failed to initialize components needed for the RAG chain.")
        return None

    logging.info("Creating RAG chain.")
    template = """Answer the question based ONLY on the following context. If no context is provided or the context does not contain relevant information to answer the question, say that you couldn't find any relevant information in the document.

Context:
{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
            {"context": _retriever, "question": RunnablePassthrough()}
            | prompt
            | _llm
            | StrOutputParser()
    )
    return chain