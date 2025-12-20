# 🤖 RAG Evaluation Studio

A comprehensive RAG observability platform built to demonstrate **Real-Time Metrics** and **Deep Evaluation** workflows. This tool allows users to interact with the US Constitution in a chat interface while also running full-scale regression testing using Ragas/DeepEval.

## 🚀 Features

### 1. 💬 Chat & Live Metrics
*   **Real-Time Scoring**: Every chat response is scored on-the-fly for **Faithfulness** and **Answer Relevancy**.
*   **Transparent Retrieval**: "Reasoning" expander shows exactly which text chunks were retrieved.
*   **Live Gauges**: Visual feedback on the quality of every single turn.

### 2. 📊 Deep Evaluation Engine
*   **Golden Data Generation**: Automatically generate synthetic test sets (Questions + Ground Truth) from your source documents.
*   **Full Regression Testing**: Run a comprehensive batch evaluation across 6 key KPIs:
    *   **Context Precision & Recall**: Is the retrieval accurate?
    *   **Faithfulness**: Is the answer hallucination-free?
    *   **Answer Relevancy**: Does it address the user's intent?
    *   **Entity Recall**: Did it capture all specific entities (e.g., list of Rights)?
    *   **Noise Sensitivity**: Does the system get distracted by irrelevant info?
*   **Report Card**: A detailed summary view of your system's overall health (Shopper Metrics, Chef Metrics, Robustness).

---

## 🛠️ Tech Stack

This project leverages a modern Python AI stack designed for production-grade evaluation:

*   **Frontend**: [Streamlit](https://streamlit.io/) (Interactive Dashboard)
*   **Orchestration**: [LangChain](https://www.langchain.com/)
*   **Evaluation**: [Ragas](https://docs.ragas.io/) & [DeepEval](https://confident-ai.com) (Metrics & Synthetic Data)
*   **LLM**: OpenAI `gpt-4o` (High-fidelity generation)
*   **Embeddings**: OpenAI `text-embedding-3-small`
*   **Vector Database**: [Qdrant](https://qdrant.tech/) (Managed vector search)
*   **Tracing**: [LangSmith](https://smith.langchain.com/) (Pipeline observability)

---

## 📂 Project Structure

```
rag-qa-ragas/
├── app.py                 # 🖥️ Main Streamlit application (UI & Navigation)
├── backend.py             # ⚙️ Core RAG logic, Metrics Calculation, & Golden Data Gen
├── config.py              # 🔧 Configuration (Qdrant, OpenAI, Chunk settings)
├── constitution_usa.pdf   # 📄 Source Data: Cleaned "Pocket Edition" PDF
├── golden_dataset.csv     # 📊 Ground Truth dataset for offline evaluation
├── questions.txt          # ❓ List of sample/test questions for the demo
├── .env.example           # 🔐 Environment variables (API Keys)
├── requirements.txt       # 📦 Production dependencies
```

---

## 🏃‍♂️ Usage Guide

### Mode 1: RAG Chat & Metrics
1.  Select **"RAG Chat & Metrics"** from the sidebar.
2.  Type a question (e.g., *"What are the requirements to be President?"*).
3.  View the answer and the **Live Metric Gauges** on the right panel.

### Mode 2: Deep Evaluation Engine
1.  Select **"Deep Evaluation (6 KPIs)"** from the sidebar.
2.  **Step 1: Generate Test Set**:
    *   Choose the number of questions (e.g., 3).
    *   Click **"⚡ Generate Golden Data"**. The system will create synthetic questions and ground truths from the Constitution PDF.
    *   *Optional*: Preview the generated data in the expandable table.
3.  **Step 2: Run Deep Evaluation**:
    *   Click **"🚀 Run Full Evaluation"**.
    *   Wait for the progress bar to complete (calculates all 6 metrics).
4.  **View Report Card**:
    *   Analyze the scores broken down by category (**Shopper Metrics** vs. **Chef Metrics**).
    *   Use this baseline to tune your chunk size or retriever settings in `config.py`.

---

## ⚡ Quick Start

### 1. Environment Setup
Clone the repository and create a virtual environment:

```
git clone <repo-url>
cd rag-qa-ragas
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```
pip install -r requirements.txt
```

### 3. Configure Credentials
Create a `.env` file in the root directory. You will need a **Qdrant** instance (local or Cloud) and **OpenAI** keys.

```
# Core Keys
OPENAI_API_KEY=sk-...

# Vector Database (Qdrant)
QDRANT_URL=https://your-cluster-url.qdrant.tech
QDRANT_API_KEY=your-qdrant-key

# Tracing (Optional but recommended)
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2-...
```

### 4. Run the Application
```
streamlit run app.py
```

---

## 📝 Configuration Settings
You can fine-tune the pipeline in `config.py`:
*   `LLM_MODEL`: Defaults to `"gpt-4o"`
*   `COLLECTION_NAME`: Defaults to `"rag_evaluation_demo"`
*   `CHUNK_SIZE`: Default `1000`
*   `CHUNK_OVERLAP`: Default `200`