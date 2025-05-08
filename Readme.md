# RAG-based Chatbot for University Admissions (Beta)

This project implements a Retrieval Augmented Generation (RAG) chatbot designed to answer questions based on a university admissions PDF document. It leverages locally run Ollama models (Qwen2.5-7B for generation, BGE-M3 for embeddings) and features a user-friendly interface built with Streamlit.

## Key Features

*   **PDF Document Ingestion:** Processes a local PDF file to build a knowledge base.
*   **Local LLM & Embeddings:** Uses Ollama to serve `qwen2.5:7b` (LLM) and `bge-m3` (embeddings) locally, ensuring data privacy and offline capability.
*   **Retrieval Augmented Generation:** Employs a RAG pipeline to retrieve relevant context from the document before generating answers.
*   **Vector Store:** Utilizes ChromaDB for efficient similarity search over document chunks.
*   **Multi-Query Retriever:** Enhances retrieval by generating multiple perspectives of the user's question.
*   **Streamlit UI:** Provides an interactive chat interface for users.
*   **Modular Codebase:** Organized into separate Python modules for better readability and maintainability.

## Technologies Used

*   **Python 3.12**
*   **Langchain:** For orchestrating the RAG pipeline.
*   **Ollama:** For serving local LLMs and embedding models.
    *   LLM: `qwen2.5:7b`
    *   Embedding Model: `bge-m3`
*   **Streamlit:** For building the web application UI.
*   **ChromaDB:** As the vector database.
*   **Unstructured:** For PDF document loading.
*   **NLTK:** For text processing tasks.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/abdulrahman-riyad/RAG-based_Chatbot_For_Uni_Addmission.git
    cd RAG-based_Chatbot_For_Uni_Addmission
    ```

2.  **Set up a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install Ollama:**
    Follow the instructions at [https://ollama.com/](https://ollama.com/) to install Ollama on your system.

5.  **Pull the required Ollama models:**
    ```bash
    ollama pull qwen2.5:7b
    ollama pull bge-m3
    ```
    Ensure Ollama is running before starting the application.

6.  **Place your PDF:**
    Rename your university admissions PDF to `input.pdf` and place it in the root directory of the project.

## How to Run

1.  Ensure your Ollama application is running and the models are downloaded.
2.  Navigate to the project directory in your terminal.
3.  Activate your virtual environment.
4.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
    The application should open in your web browser. The first time you run it, it will process `input.pdf` and create the vector database (`chroma_db_beta_rag/` directory).

## Project Structure

The project is organized into several modules:
*   `app.py`: Main Streamlit application logic.
*   `config.py`: Configuration constants.
*   `data_handler.py`: PDF ingestion and document splitting.
*   `llm_services.py`: LLM, embeddings, retriever, and RAG chain setup.
*   `vector_db_manager.py`: Vector database management (ChromaDB).
*   `ui_components.py`: Streamlit UI elements and styling.
*   `utils.py`: Utility functions (e.g., model validation).

## Future Improvements (Optional To-Do)

*   [ ] Add support for more document types (e.g., .txt, .docx).strategies.
*   [ ] Allow users to upload PDFs through the UI.
*   [ ] Add conversation history management (saving/loading chats).