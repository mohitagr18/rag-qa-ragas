import streamlit as st
import tempfile
import os
import pandas as pd
from config import Config
from backend import RAGBackend

# Setup Page
st.set_page_config(page_title=Config.PAGE_TITLE, page_icon=Config.PAGE_ICON, layout="wide")

# Helper function for HTML table rendering
def render_wrapped_table(df):
    """Renders a dataframe with wrapping and fixed widths using HTML."""
    html = df.to_html(index=False, classes="table")
    style = """<style>
    table {width: 100%; border-collapse: collapse; font-size: 14px;} 
    th, td {border: 1px solid #ddd; padding: 10px; text-align: left; vertical-align: top;}
    
    /* Column Width Control */
    td:nth-child(1) {width: 25%;} /* Question */
    td:nth-child(2) {width: 15%;} /* Type */
    td:nth-child(3) {width: 30%;} /* Answer */
    td:nth-child(4) {width: 30%;} /* Context */
    </style>"""
    st.markdown(style + html, unsafe_allow_html=True)

# Initialize Backend
if "backend" not in st.session_state:
    try:
        Config.validate()
        st.session_state.backend = RAGBackend()
    except Exception as e:
        st.error(f"Configuration Error: {e}")
        st.stop()

# Load Persistent Data
if "test_data" not in st.session_state:
    if os.path.exists("golden_dataset.csv"):
        try:
            st.session_state.test_data = pd.read_csv("golden_dataset.csv")
        except Exception:
            pass

# Sidebar Navigation
page = st.sidebar.radio("Navigation", ["1. RAG Chat & Metrics", "2. Deep Evaluation (6 KPIs)"])

# ==========================================
# PAGE 1: CHAT & INGESTION (Consolidated)
# ==========================================
if page == "1. RAG Chat & Metrics":
    st.title("💬 RAG Chat & Live Metrics")
    st.markdown("Upload a document, then chat with it. The system evaluates every answer in real-time.")

    with st.expander("📚 Document Ingestion (Upload PDF)", expanded=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
        with col2:
            if st.button("Clear DB"):
                st.session_state.backend.clear_database()
                st.toast("Database Cleared!")

        if uploaded_file:
            # Save for Generation logic later
            if "temp_file_path" not in st.session_state:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    st.session_state.temp_file_path = tmp_file.name

            if st.button("🚀 Ingest & Ready to Chat"):
                with st.spinner("Indexing..."):
                    try:
                        uploaded_file.seek(0)
                        num = st.session_state.backend.ingest_file(uploaded_file)
                        st.success(f"Indexed {num} chunks! You can now chat below.")
                    except Exception as e:
                        st.error(f"Error: {e}")

    st.divider()

    # Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "metrics" in message:
                cols = st.columns(2)
                cols[0].metric("Faithfulness", f"{message['metrics']['faithfulness']:.2f}")
                cols[1].metric("Relevancy", f"{message['metrics']['answer_relevancy']:.2f}")

    if prompt := st.chat_input("Ask a question about your PDF..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.backend.query_and_evaluate(prompt)
                    answer = result["answer"]
                    metrics = result["metrics"]
                    
                    st.markdown(answer)
                    
                    c1, c2 = st.columns(2)
                    c1.metric("Faithfulness", f"{metrics['faithfulness']:.2f}", help="Is the answer derived from context?")
                    c2.metric("Relevancy", f"{metrics['answer_relevancy']:.2f}", help="Did it answer the user's question?")
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "metrics": metrics
                    })
                    
                    with st.expander("🔍 View Retrieved Context"):
                        for i, ctx in enumerate(result["contexts"]):
                            st.caption(f"**Chunk {i+1}:**")
                            st.text(ctx[:300] + "...")
                except Exception as e:
                    st.error(f"Error: {e}")

# ==========================================
# PAGE 2: DEEP EVALUATION
# ==========================================
elif page == "2. Deep Evaluation (6 KPIs)":
    st.title("📊 Deep Evaluation Engine")
    st.markdown("Generate a 'Golden Dataset' (Exam) and run a full regression test.")

    # --- STEP 1: GENERATION ---
    st.header("Step 1: Generate Test Set")
    
    if "temp_file_path" not in st.session_state or not os.path.exists(st.session_state.temp_file_path):
        st.warning("⚠️ Please upload a PDF in the 'Chat' page first.")
    else:
        col1, col2 = st.columns([1, 3])
        with col1:
            num_questions = st.number_input("Questions", 1, 20, 3)
            st.write("")
            if st.button("⚡ Generate Golden Data", use_container_width=True):
                with st.spinner("Generating..."):
                    try:
                        df = st.session_state.backend.generate_test_data(
                            st.session_state.temp_file_path, num_questions
                        )
                        st.session_state.test_data = df
                        df.to_csv("golden_dataset.csv", index=False)
                        st.success("✅ Saved!")
                    except Exception as e:
                        st.error(f"Error: {e}")
            st.caption("⚠️ Takes 3-5 mins.")

    # Show Preview
    if "test_data" in st.session_state:
        with st.expander("👀 Preview Golden Data", expanded=False):
            cols = ["user_input", "synthesizer_name", "reference", "reference_contexts"]
            valid_cols = [c for c in cols if c in st.session_state.test_data.columns]
            df_show = st.session_state.test_data[valid_cols].copy()
            
            name_map = {
                "user_input": "Question",
                "synthesizer_name": "Type",
                "reference": "Ground Truth",
                "reference_contexts": "Contexts"
            }
            df_show = df_show.rename(columns=name_map)
            if "Type" in df_show.columns:
                df_show["Type"] = df_show["Type"].str.replace("_", " ").str.title()
            
            render_wrapped_table(df_show.head())

    st.divider()

    # --- STEP 2: EVALUATION ---
    st.header("Step 2: Run Deep Evaluation")
    
    if "test_data" not in st.session_state:
        st.info("Generate data in Step 1 first.")
    else:
        if st.button("🚀 Run Full Evaluation (Calculate 6 KPIs)"):
            with st.spinner("Running Pipeline..."):
                try:
                    results = st.session_state.backend.run_batch_evaluation(st.session_state.test_data)
                    st.session_state.eval_results = results
                    st.success("✅ Complete!")
                except Exception as e:
                    st.error(f"Error: {e}")

    # --- DISPLAY RESULTS ---
    if "eval_results" in st.session_state:
        st.divider()
        st.header("🏆 RAG Evaluation Report Card")
        
        res = st.session_state.eval_results
        try: results_df = res.to_pandas()
        except: results_df = pd.DataFrame()

        def get_mean(col):
            if not results_df.empty and col in results_df.columns:
                return results_df[col].mean()
            return 0.0

        st.markdown("### 🛍️ Shopper Metrics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Context Precision", f"{get_mean('context_precision'):.2f}", 
                  help="Signal-to-Noise Ratio: Of the chunks we retrieved, what percentage were actually relevant to the question? (Higher is better)")
        c2.metric("Context Recall", f"{get_mean('context_recall'):.2f}", 
        help="Completeness: Did we manage to find ALL the relevant information needed to answer the question? (Higher is better)")
        c3.metric("Entity Recall", f"{get_mean('context_entity_recall'):.2f}", 
        help="Specifics: Did we retrieve the specific proper nouns, dates, and IDs mentioned in the ground truth? (Higher is better)")
        
        st.write(""); st.write("")

        st.markdown("### 👨‍🍳 Chef Metrics")
        c4, c5 = st.columns(2)
        c4.metric("Faithfulness", f"{get_mean('faithfulness'):.2f}", 
                 help="Hallucination Check: Is every claim in the answer supported by the retrieved context? 1.0 means no hallucinations. (Higher is better)")
        c5.metric("Answer Relevancy", f"{get_mean('answer_relevancy'):.2f}", 
                 help="Focus: Did the system answer the specific question asked, or did it pivot to a generic response? (Higher is better)")
        
        st.write(""); st.write("")

        st.markdown("### 🛡️ Robustness")
        c6, c7 = st.columns(2)
        c6.metric("Noise Sensitivity (Relevant)", f"{get_mean('noise_sensitivity_relevant'):.2f}", 
                 help="Distraction by Truth: How often does the system make a mistake because it saw a TRUE but IRRELEVANT fact? (LOWER is better)")
        c7.metric("Noise Sensitivity (Irrelevant)", f"{get_mean('noise_sensitivity_irrelevant'):.2f}", 
                 help="Distraction by Noise: How often does the system make a mistake because it saw GARBAGE/FALSE data in the context? (LOWER is better)")

        st.divider()
        st.subheader("🔍 Detailed Breakdown")
        if not results_df.empty:
            st.dataframe(results_df)
