import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    PAGE_TITLE = "RAG Evaluation Studio"
    PAGE_ICON = "🤖"
    
    # OpenAI Settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LLM_MODEL = "gpt-4o"  # Using a capable model for RAG
    EMBEDDING_MODEL = "text-embedding-3-small"
    
    # Qdrant Settings
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    COLLECTION_NAME = "rag_evaluation_demo"
    
    # Text Splitting Settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    @staticmethod
    def validate():
        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is missing in .env")
        if not Config.QDRANT_URL:
            raise ValueError("QDRANT_URL is missing in .env")
        if not Config.QDRANT_API_KEY:
            raise ValueError("QDRANT_API_KEY is missing in .env")
