import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()


@dataclass
class Config:
    """Simple configuration object reading environment variables."""

    # Ollama settings
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model_name: str = os.getenv("MODEL_NAME", "llama3.2:3b")
    model_temperature: float = float(os.getenv("MODEL_TEMPERATURE", "0.0"))
    ollama_timeout: float = float(os.getenv("OLLAMA_TIMEOUT", "180"))

    # RAG settings
    chroma_db_dir: str = os.getenv(
        "CHROMA_DB_DIR",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "chroma")),
    )
    rag_collection_name: str = os.getenv("RAG_COLLECTION_NAME", "oracle_rules")
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "1.2"))


# Instantiate global configuration object
config = Config()
