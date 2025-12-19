import os
import tempfile
import hashlib
import pandas as pd
from datasets import Dataset

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Qdrant Imports
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Ragas Imports (v0.3.x)
from ragas import evaluate, SingleTurnSample
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    ContextPrecision,
    ContextRecall,
    ContextEntityRecall,
    NoiseSensitivity
)

# Configuration
from config import Config


class RAGBackend:
    """
    Backend handler for RAG operations, including ingestion, retrieval, generation,
    and evaluation using Ragas.
    """

    def __init__(self):
        """
        Initialize the RAG Backend.
        Sets up the OpenAI Embeddings, LLM, and Qdrant Client based on configuration.
        Ensures the vector collection exists on startup.
        """
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
        """
        Check if the configured Qdrant collection exists.
        If it does not exist, create it with the correct vector configuration (Cosine Distance).
        """
        collections = self.qdrant_client.get_collections()
        exists = any(c.name == Config.COLLECTION_NAME for c in collections.collections)
        
        if not exists:
            self.qdrant_client.create_collection(
                collection_name=Config.COLLECTION_NAME,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )

    def ingest_file(self, uploaded_file):
        """
        Process an uploaded PDF file and index it into the Vector Database.
        
        Steps:
        1. Save uploaded file to a temporary location.
        2. Load text using PyPDFLoader.
        3. Split text into chunks using RecursiveCharacterTextSplitter.
        4. Generate deterministic IDs (MD5 hash) for deduplication.
        5. Add chunks to Qdrant.
        
        Args:
            uploaded_file: Streamlit UploadedFile object.
            
        Returns:
            int: The number of chunks successfully indexed.
        """
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
        """
        Delete the entire vector collection and recreate it empty.
        Useful for resetting the knowledge base.
        """
        self.qdrant_client.delete_collection(Config.COLLECTION_NAME)
        self._ensure_collection()

    def generate_test_data(self, file_path, num_questions=5):
        """
        Generate synthetic 'Golden Data' (Questions & Ground Truths) from a PDF.
        Uses Ragas TestsetGenerator to create diverse questions (reasoning, multi-hop, etc.).
        
        Args:
            file_path (str): Path to the source PDF file.
            num_questions (int): Number of test cases to generate.
            
        Returns:
            pd.DataFrame: A DataFrame containing the generated test set.
        """
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
        """
        Perform a single-turn RAG operation (Retrieve -> Generate) and immediately
        evaluate it using Ragas metrics (Faithfulness & Answer Relevancy).
        
        Args:
            question (str): The user's input question.
            
        Returns:
            dict: Contains 'answer', 'contexts', and 'metrics' (faithfulness/relevancy scores).
        """
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
        
        # Instantiate Metrics (Use Uppercase Class Names)
        faith_metric = Faithfulness(llm=eval_llm)
        relevancy_metric = ResponseRelevancy(llm=eval_llm, embeddings=eval_embeddings)
        
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=retrieved_contexts
        )
        
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

    def run_batch_evaluation(self, test_df: pd.DataFrame):
        """
        Run a full RAGAS evaluation (6 KPIs) on a provided test dataset.
        
        The dataset must contain 'question' and 'ground_truth' columns.
        The system will run the RAG pipeline for each question to generate 'answer' and 'contexts',
        then compare them against the ground truth.
        
        Args:
            test_df (pd.DataFrame): DataFrame with test questions and ground truths.
            
        Returns:
            EvaluationResult: The Ragas evaluation result object containing all scores.
        """
        # 1. Data Normalization
        df = test_df.copy()
        if 'user_input' in df.columns and 'question' not in df.columns:
            df = df.rename(columns={'user_input': 'question'})
        if 'reference' in df.columns and 'ground_truth' not in df.columns:
            df = df.rename(columns={'reference': 'ground_truth'})
            
        if 'question' not in df.columns:
            raise ValueError(f"Dataset missing 'question' column. Found: {df.columns}")
        if 'ground_truth' not in df.columns:
            raise ValueError(f"Dataset missing 'ground_truth' column. Found: {df.columns}")

        # 2. Run Pipeline (Retrieval + Generation)
        questions = df['question'].tolist()
        ground_truths = df['ground_truth'].tolist()
        answers = []
        contexts = []

        vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=Config.COLLECTION_NAME,
            embedding=self.embeddings
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        template = "Answer the question based ONLY on the following context:\n{context}\n\nQuestion: {question}"
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()

        for q in questions:
            docs = retriever.invoke(q)
            retrieved_txts = [d.page_content for d in docs]
            context_str = "\n\n".join(retrieved_txts)
            
            ans = chain.invoke({"context": context_str, "question": q})
            
            answers.append(ans)
            contexts.append(retrieved_txts)

        # 3. Create Dataset for Ragas
        data = {
            "user_input": questions,
            "response": answers,
            "retrieved_contexts": contexts,
            "reference": ground_truths
        }
        dataset = Dataset.from_dict(data)

        # 4. Define Metrics (Classes Instantiated)
        eval_llm = LangchainLLMWrapper(self.llm)
        eval_embeddings = LangchainEmbeddingsWrapper(self.embeddings)
        
        metrics = [
            Faithfulness(llm=eval_llm),
            ResponseRelevancy(llm=eval_llm, embeddings=eval_embeddings),
            ContextPrecision(llm=eval_llm),
            ContextRecall(llm=eval_llm),
            ContextEntityRecall(llm=eval_llm),
            NoiseSensitivity(llm=eval_llm)
        ]

        # 5. Execute Evaluation
        results = evaluate(
            dataset=dataset,
            metrics=metrics
        )
        
        return results
