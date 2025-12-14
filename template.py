"""
RAG Q&A System - Project Structure Generator

Run this script to create the complete folder structure for the RAG Q&A project.

Usage:
    python template.py

This will create the following structure:
    rag-qa-project/
    ├── app/
    │   ├── api/routes/
    │   ├── core/
    │   └── utils/
    ├── tests/
    ├── docs/
    ├── sample_data/
    └── .github/workflows/
"""

import os
from pathlib import Path

# Project root directory name
PROJECT_NAME = "rag-qa-project"

# Directory structure
DIRECTORIES = [
    "app",
    "app/api",
    "app/api/routes",
    "app/core",
    "app/utils",
    "tests",
    "docs",
    "sample_data",
    ".github",
    ".github/workflows",
]

# Files to create with their content
FILES = {
    # ===========================================
    # Root Configuration Files
    # ===========================================
    ".env.example": """# ===========================================
# RAG Q&A System - Environment Configuration
# ===========================================

# OpenAI Configuration (Required)
OPENAI_API_KEY=sk-proj-your-key-here

# Qdrant Cloud Configuration (Required)
QDRANT_URL=https://your-cluster-id.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key-here

# Collection Settings
COLLECTION_NAME=rag_documents

# Document Processing Settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Model Configuration
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0

# Retrieval Settings
RETRIEVAL_K=4

# Logging
LOG_LEVEL=INFO

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
""",

    ".gitignore": """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
dist/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.env
.venv
env/
venv/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Testing
.pytest_cache/
.coverage
htmlcov/

# Jupyter
.ipynb_checkpoints/

# Logs
logs/
*.log

# OS
.DS_Store
""",

    "requirements.txt": """# Core Framework
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
python-multipart>=0.0.6

# LangChain & AI
langchain>=0.3.0
langchain-openai>=0.2.0
langchain-qdrant>=0.2.0
langchain-community>=0.3.0
langchain-text-splitters>=0.3.0

# Vector Database
qdrant-client>=1.12.0

# Document Processing
pypdf>=4.0.0

# Configuration & Validation
pydantic>=2.5.0
pydantic-settings>=2.1.0
python-dotenv>=1.0.0

# HTTP Client
httpx>=0.26.0
""",

    "requirements-dev.txt": """-r requirements.txt

# Testing
pytest>=8.0.0
pytest-asyncio>=0.23.0
pytest-cov>=4.1.0

# Linting & Formatting
ruff>=0.2.0
black>=24.1.0
""",

    "pytest.ini": """[pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
asyncio_mode = auto
addopts = -v --tb=short
""",

    "Dockerfile": """FROM python:3.12-slim as builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim as production
WORKDIR /app
RUN groupadd -r appgroup && useradd -r -g appgroup appuser
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY app/ ./app/
RUN chown -R appuser:appgroup /app
USER appuser
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 CMD python -c "import httpx; httpx.get('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
""",

    "docker-compose.yml": """version: "3.8"

services:
  rag-api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: rag-qa-api
    ports:
      - "8000:8000"
    env_file:
      - .env
    restart: unless-stopped
""",

    # ===========================================
    # App Package Files
    # ===========================================
    "app/__init__.py": '''"""RAG Q&A System - Main Application Package."""

__version__ = "0.1.0"
''',

    "app/config.py": '''"""Application configuration using pydantic-settings."""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # OpenAI Configuration
    openai_api_key: str

    # Qdrant Cloud Configuration
    qdrant_url: str
    qdrant_api_key: str

    # Collection Settings
    collection_name: str = "rag_documents"

    # Document Processing Settings
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Model Configuration
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0

    # Retrieval Settings
    retrieval_k: int = 4

    # Logging
    log_level: str = "INFO"

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Application Info
    app_name: str = "RAG Q&A System"
    app_version: str = "0.1.0"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
''',

    "app/main.py": '''"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.routes import health, documents, query
from app.config import get_settings
from app.utils.logger import get_logger, setup_logging

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    setup_logging(settings.log_level)
    logger = get_logger(__name__)
    logger.info(f"Starting {settings.app_name} v{__version__}")
    yield
    logger.info("Shutting down application")


app = FastAPI(
    title=settings.app_name,
    description="RAG Q&A System API",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(query.router)


@app.get("/", tags=["Root"])
async def root():
    return {"message": f"Welcome to {settings.app_name}", "version": __version__}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.api_host, port=settings.api_port, reload=True)
''',

    # ===========================================
    # Utils Package
    # ===========================================
    "app/utils/__init__.py": '''"""Utility modules package."""

from app.utils.logger import get_logger

__all__ = ["get_logger"]
''',

    "app/utils/logger.py": '''"""Logging configuration."""

import logging
import sys
from functools import lru_cache


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for the application."""
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Reduce noise from third-party libraries
    for lib in ["httpx", "httpcore", "openai", "qdrant_client", "urllib3"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


@lru_cache
def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)
''',

    # ===========================================
    # API Package
    # ===========================================
    "app/api/__init__.py": '''"""API layer package."""
''',

    "app/api/schemas.py": '''"""Pydantic schemas for API request/response models."""

from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str


class DocumentUploadResponse(BaseModel):
    message: str
    filename: str
    chunks_created: int
    document_ids: list[str]


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    include_sources: bool = True


class SourceDocument(BaseModel):
    content: str
    metadata: dict[str, Any]


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceDocument] | None = None
    processing_time_ms: float
''',

    "app/api/routes/__init__.py": '''"""API routes package."""
''',

    "app/api/routes/health.py": '''"""Health check endpoints."""

from datetime import datetime
from fastapi import APIRouter
from app import __version__
from app.api.schemas import HealthResponse

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(status="healthy", timestamp=datetime.utcnow(), version=__version__)
''',

    "app/api/routes/documents.py": '''"""Document management endpoints."""

from fastapi import APIRouter, File, HTTPException, UploadFile
from app.api.schemas import DocumentUploadResponse
from app.core.document_processor import DocumentProcessor
from app.core.vector_store import VectorStoreService
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    """Upload and process a document."""
    logger.info(f"Received document upload: {file.filename}")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    try:
        processor = DocumentProcessor()
        chunks = processor.process_upload(file.file, file.filename)

        if not chunks:
            raise HTTPException(status_code=400, detail="No content extracted")

        vector_store = VectorStoreService()
        document_ids = vector_store.add_documents(chunks)

        return DocumentUploadResponse(
            message="Document uploaded successfully",
            filename=file.filename,
            chunks_created=len(chunks),
            document_ids=document_ids,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info")
async def get_collection_info():
    """Get collection information."""
    vector_store = VectorStoreService()
    return vector_store.get_collection_info()
''',

    "app/api/routes/query.py": '''"""Query endpoints for RAG Q&A."""

import time
from fastapi import APIRouter, HTTPException
from app.api.schemas import QueryRequest, QueryResponse, SourceDocument
from app.core.rag_chain import RAGChain
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/query", tags=["Query"])


@router.post("", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Process a RAG query."""
    logger.info(f"Query received: {request.question[:100]}...")
    start_time = time.time()

    try:
        rag_chain = RAGChain()

        if request.include_sources:
            result = await rag_chain.aquery_with_sources(request.question)
            sources = [
                SourceDocument(content=s["content"], metadata=s["metadata"])
                for s in result["sources"]
            ]
            answer = result["answer"]
        else:
            answer = await rag_chain.aquery(request.question)
            sources = None

        processing_time = (time.time() - start_time) * 1000

        return QueryResponse(
            question=request.question,
            answer=answer,
            sources=sources,
            processing_time_ms=round(processing_time, 2),
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))
''',

    # ===========================================
    # Core Package
    # ===========================================
    "app/core/__init__.py": '''"""Core business logic package."""
''',

    "app/core/document_processor.py": '''"""Document processing module."""

import tempfile
from pathlib import Path
from typing import BinaryIO

from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class DocumentProcessor:
    """Process documents for RAG pipeline."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".csv"}

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\\n\\n", "\\n", ". ", " ", ""],
        )

    def load_file(self, file_path: str | Path) -> list[Document]:
        """Load a file based on its extension."""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported extension: {extension}")

        loaders = {
            ".pdf": lambda p: PyPDFLoader(str(p)).load(),
            ".txt": lambda p: TextLoader(str(p), encoding="utf-8").load(),
            ".csv": lambda p: CSVLoader(str(p), encoding="utf-8").load(),
        }
        return loaders[extension](file_path)

    def process_upload(self, file: BinaryIO, filename: str) -> list[Document]:
        """Process uploaded file."""
        extension = Path(filename).suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported extension: {extension}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        try:
            documents = self.load_file(tmp_path)
            for doc in documents:
                doc.metadata["source"] = filename
            return self.text_splitter.split_documents(documents)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
''',

    "app/core/embeddings.py": '''"""Embedding generation module."""

from functools import lru_cache
from langchain_openai import OpenAIEmbeddings
from app.config import get_settings

settings = get_settings()


@lru_cache
def get_embeddings() -> OpenAIEmbeddings:
    """Get cached OpenAI embeddings instance."""
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )
''',

    "app/core/vector_store.py": '''"""Vector store module for Qdrant operations."""

from functools import lru_cache
from uuid import uuid4

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Distance, VectorParams

from app.config import get_settings
from app.core.embeddings import get_embeddings
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()
EMBEDDING_DIMENSION = 1536


@lru_cache
def get_qdrant_client() -> QdrantClient:
    """Get cached Qdrant client instance."""
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


class VectorStoreService:
    """Service for managing vector store operations."""

    def __init__(self, collection_name: str | None = None):
        self.collection_name = collection_name or settings.collection_name
        self.client = get_qdrant_client()
        self.embeddings = get_embeddings()
        self._ensure_collection()
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

    def _ensure_collection(self) -> None:
        """Ensure collection exists."""
        try:
            self.client.get_collection(self.collection_name)
        except UnexpectedResponse:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE),
            )

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to vector store."""
        if not documents:
            return []
        ids = [str(uuid4()) for _ in documents]
        self.vector_store.add_documents(documents, ids=ids)
        return ids

    def get_retriever(self, k: int | None = None):
        """Get retriever for the vector store."""
        return self.vector_store.as_retriever(search_kwargs={"k": k or settings.retrieval_k})

    def get_collection_info(self) -> dict:
        """Get collection information."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {"name": self.collection_name, "points_count": info.points_count, "status": info.status.value}
        except UnexpectedResponse:
            return {"name": self.collection_name, "points_count": 0, "status": "not_found"}
''',

    "app/core/rag_chain.py": '''"""RAG chain module using LangChain LCEL."""

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.core.vector_store import VectorStoreService
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

RAG_PROMPT = """Answer the question based on the context below.
If you cannot answer based on the context, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:"""


def format_docs(docs: list[Document]) -> str:
    return "\\n\\n---\\n\\n".join(doc.page_content for doc in docs)


class RAGChain:
    """RAG chain for question answering."""

    def __init__(self, vector_store_service: VectorStoreService | None = None):
        self.vector_store = vector_store_service or VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            openai_api_key=settings.openai_api_key,
        )
        self.prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
        self.chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    async def aquery(self, question: str) -> str:
        """Execute async RAG query."""
        return await self.chain.ainvoke(question)

    async def aquery_with_sources(self, question: str) -> dict:
        """Execute async RAG query with sources."""
        answer = await self.chain.ainvoke(question)
        source_docs = self.retriever.invoke(question)
        sources = [
            {"content": doc.page_content[:500], "metadata": doc.metadata}
            for doc in source_docs
        ]
        return {"answer": answer, "sources": sources}
''',

    # ===========================================
    # Tests Package
    # ===========================================
    "tests/__init__.py": '''"""Test suite package."""
''',

    "tests/conftest.py": '''"""Pytest configuration and fixtures."""

import os
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["QDRANT_URL"] = "http://localhost:6333"
os.environ["QDRANT_API_KEY"] = "test-key"


@pytest.fixture
def mock_vector_store():
    with patch("app.core.vector_store.VectorStoreService") as mock:
        service = MagicMock()
        service.health_check.return_value = True
        service.get_collection_info.return_value = {"name": "test", "points_count": 0, "status": "green"}
        service.add_documents.return_value = ["id1"]
        mock.return_value = service
        yield service


@pytest.fixture
def mock_rag_chain():
    with patch("app.core.rag_chain.RAGChain") as mock:
        chain = MagicMock()
        chain.aquery.return_value = "Test answer"
        chain.aquery_with_sources.return_value = {"answer": "Test answer", "sources": []}
        mock.return_value = chain
        yield chain


@pytest.fixture
def client(mock_vector_store, mock_rag_chain):
    from app.main import app
    with TestClient(app) as c:
        yield c
''',

    "tests/test_health.py": '''"""Tests for health endpoints."""


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
''',

    "tests/test_query.py": '''"""Tests for query endpoints."""


def test_query_endpoint(client, mock_rag_chain):
    response = client.post("/query", json={"question": "What is RAG?", "include_sources": True})
    assert response.status_code == 200
    assert "answer" in response.json()


def test_query_empty_question(client):
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422
''',

    # ===========================================
    # GitHub Actions
    # ===========================================
    ".github/workflows/ci.yml": """name: CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install ruff black
      - run: ruff check app/ tests/
      - run: black --check app/ tests/

  test:
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements-dev.txt
      - run: pytest --cov=app
        env:
          OPENAI_API_KEY: test-key
          QDRANT_URL: http://localhost:6333
          QDRANT_API_KEY: test-key

  build:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - run: docker build -t rag-qa-system .
""",

    ".github/workflows/deploy.yml": """name: CD - Deploy to AWS

on:
  push:
    branches: [main]
  workflow_dispatch:

env:
  AWS_REGION: us-east-1
  ECR_REPOSITORY: rag-qa-system
  APP_RUNNER_SERVICE: rag-qa-service

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    outputs:
      image_uri: ${{ steps.build.outputs.image_uri }}
    steps:
      - uses: actions/checkout@v4
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}
      - uses: aws-actions/amazon-ecr-login@v2
        id: login-ecr
      - name: Build and push
        id: build
        run: |
          docker build -t ${{ steps.login-ecr.outputs.registry }}/$ECR_REPOSITORY:${{ github.sha }} .
          docker push ${{ steps.login-ecr.outputs.registry }}/$ECR_REPOSITORY:${{ github.sha }}
          echo "image_uri=${{ steps.login-ecr.outputs.registry }}/$ECR_REPOSITORY:${{ github.sha }}" >> $GITHUB_OUTPUT

  deploy:
    runs-on: ubuntu-latest
    needs: build-and-push
    steps:
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}
      - uses: awslabs/amazon-app-runner-deploy@main
        with:
          service: ${{ env.APP_RUNNER_SERVICE }}
          image: ${{ needs.build-and-push.outputs.image_uri }}
          access-role-arn: ${{ secrets.APP_RUNNER_ECR_ACCESS_ROLE_ARN }}
          region: ${{ env.AWS_REGION }}
          cpu: 1
          memory: 2
          port: 8000
          runtime-environment-variables: |
            OPENAI_API_KEY=${{ secrets.OPENAI_API_KEY }}
            QDRANT_URL=${{ secrets.QDRANT_URL }}
            QDRANT_API_KEY=${{ secrets.QDRANT_API_KEY }}
""",

    # ===========================================
    # Documentation
    # ===========================================
    "README.md": """# RAG Q&A System

A production-ready RAG question-answering system with FastAPI, LangChain, and Qdrant.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 3. Run the application
uvicorn app.main:app --reload

# 4. Open Swagger UI
# http://localhost:8000/docs
```

## Docker

```bash
docker-compose up --build
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/documents/upload` | POST | Upload document |
| `/documents/info` | GET | Collection info |
| `/query` | POST | Ask a question |

## Tech Stack

- FastAPI
- LangChain
- Qdrant Cloud
- OpenAI (gpt-4o-mini)
- Docker
- GitHub Actions
""",

    "sample_data/.gitkeep": "",
}


def create_project_structure():
    """Create the complete project structure."""
    # Get the directory where the script is run
    base_path = Path.cwd()

    # Check if we should create in current directory or new directory
    if base_path.name == PROJECT_NAME:
        project_path = base_path
    else:
        project_path = base_path / PROJECT_NAME

    print(f"Creating project structure in: {project_path}")
    print("=" * 50)

    # Create directories
    for directory in DIRECTORIES:
        dir_path = project_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}/")

    print()

    # Create files
    for file_path, content in FILES.items():
        full_path = project_path / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
        print(f"Created file: {file_path}")

    print()
    print("=" * 50)
    print("Project structure created successfully!")
    print()
    print("Next steps:")
    print(f"  1. cd {PROJECT_NAME}")
    print("  2. python -m venv venv")
    print("  3. source venv/bin/activate  # or venv\\Scripts\\activate on Windows")
    print("  4. pip install -r requirements.txt")
    print("  5. cp .env.example .env")
    print("  6. Edit .env with your API keys")
    print("  7. uvicorn app.main:app --reload")
    print()
    print("Open http://localhost:8000/docs for Swagger UI")


if __name__ == "__main__":
    create_project_structure()
