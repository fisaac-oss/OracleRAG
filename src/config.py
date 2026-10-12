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
    model_name: str = os.getenv("MODEL_NAME", "qwen3.5:4b")
    model_temperature: float = float(os.getenv("MODEL_TEMPERATURE", "0.7"))


# Instantiate global configuration object
config = Config()
