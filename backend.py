from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
# FIXED IMPORT:
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import Config
import tempfile
import os
import pandas as pd
import hashlib

# RAGAS 0.3.x Imports
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import SingleTurnSample
from ragas.metrics import Faithfulness, ResponseRelevancy

class RAGBackend:
    def __init__(self):
        # Initialize Clients
        self.embeddings = OpenAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            openai_api_key=Config.OPENAI_API_KEY
        )
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            openai_api_key=Config.OPENAI_API_KEY
        )
        self.qdrant_client = QdrantClient(
            url=Config.QDRANT_URL,
            api_key=Config.QDRANT_API_KEY
        )
        self._ensure_collection()

    def _ensure_collection(self):
        """Check if collection exists, else create it."""
        collections = self.qdrant_client.get_collections()
        exists = any(c.name == Config.COLLECTION_NAME for c in collections.collections)
        
        if not exists:
            self.qdrant_client.create_collection(
                collection_name=Config.COLLECTION_NAME,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )

    def ingest_file(self, uploaded_file):
        """Process a PDF upload and index it."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP
            )
            chunks = splitter.split_documents(docs)

            # --- DEDUPLICATION LOGIC ---
            # Generate deterministic IDs based on content hash
            ids = [hashlib.md5(d.page_content.encode("utf-8")).hexdigest() for d in chunks]

            vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=Config.COLLECTION_NAME,
                embedding=self.embeddings
            )
            vector_store.add_documents(chunks, ids=ids)
            return len(chunks)
        finally:
            os.remove(tmp_path)

    def clear_database(self):
        self.qdrant_client.delete_collection(Config.COLLECTION_NAME)
        self._ensure_collection()

    def generate_test_data(self, file_path, num_questions=5):
        """Generate synthetic test data using Ragas 0.3.x"""
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        generator_llm = LangchainLLMWrapper(self.llm)
        generator_embeddings = LangchainEmbeddingsWrapper(self.embeddings)

        generator = TestsetGenerator(
            llm=generator_llm,
            embedding_model=generator_embeddings
        )

        dataset = generator.generate_with_langchain_docs(
            documents,
            testset_size=num_questions
        )
        return dataset.to_pandas()

    def query_and_evaluate(self, question: str):
        """Retrieve, Generate, and Evaluate (Single Turn)"""
        # 1. Retrieval
        vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=Config.COLLECTION_NAME,
            embedding=self.embeddings
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(question)
        retrieved_contexts = [d.page_content for d in docs]
        context_text = "\n\n".join(retrieved_contexts)

        # 2. Generation
        template = """Answer the question based ONLY on the following context:
        {context}
        
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context_text, "question": question})

        # 3. Evaluation
        eval_llm = LangchainLLMWrapper(self.llm)
        eval_embeddings = LangchainEmbeddingsWrapper(self.embeddings)
        
        faith_metric = Faithfulness(llm=eval_llm)
        relevancy_metric = ResponseRelevancy(llm=eval_llm, embeddings=eval_embeddings)
        
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=retrieved_contexts
        )
        
        # Async methods are preferred but .single_turn_score() is the sync wrapper in 0.3.x
        faith_score = faith_metric.single_turn_score(sample)
        rel_score = relevancy_metric.single_turn_score(sample)

        return {
            "answer": answer,
            "contexts": retrieved_contexts,
            "metrics": {
                "faithfulness": faith_score,
                "answer_relevancy": rel_score
            }
        }
