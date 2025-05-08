# --- Document and Paths ---
DOC_PATH = "./input.pdf"
PERSIST_DIRECTORY = "./chroma_db_beta_rag" # Directory to store the Chroma vector database

# --- Model Names ---
MODEL_NAME = "qwen2.5:7b"
EMBEDDING_MODEL = "bge-m3"

# --- Vector Store ---
VECTOR_STORE_NAME = "beta-rag"

# --- Text Splitting ---
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 300

# --- Logging ---
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(module)s - %(message)s'