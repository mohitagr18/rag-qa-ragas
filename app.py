import streamlit as st
import tempfile
import os
import pandas as pd
from config import Config
from backend import RAGBackend

# Setup Page
st.set_page_config(page_title=Config.PAGE_TITLE, page_icon=Config.PAGE_ICON, layout="wide")

# Initialize Backend
if "backend" not in st.session_state:
    try:
        Config.validate()
        st.session_state.backend = RAGBackend()
    except Exception as e:
        st.error(f"Configuration Error: {e}")
        st.stop()

# Sidebar Navigation
page = st.sidebar.radio("Navigation", ["1. Ingestion & Data", "2. Chat & Live Metrics", "3. Full Evaluation"])

if page == "1. Ingestion & Data":
    st.title("📚 Ingestion & Data Generation")
    st.markdown("""
    This section handles the preparation of your RAG system.
    1. **Ingest**: Upload PDFs to populate the Vector Database (The Shopper).
    2. **Generate**: Create synthetic 'Ground Truth' data for evaluation.
    """)
    
    st.divider()

    # --- SECTION 1: INGESTION ---
    st.header("Step 1: Ingest Documents")
    uploaded_file = st.file_uploader("Upload a PDF Document", type=["pdf"])
    
    if uploaded_file:
        # Save file to temp location to allow both Ingestion and Generation to access it
        # We persist the temp path in session state so it survives re-runs
        if "temp_file_path" not in st.session_state:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                st.session_state.temp_file_path = tmp_file.name

        if st.button("🚀 Ingest into Vector DB"):
            with st.spinner("Processing document..."):
                try:
                    # Reset file pointer for ingestion
                    uploaded_file.seek(0)
                    num_chunks = st.session_state.backend.ingest_file(uploaded_file)
                    st.success(f"✅ Successfully ingested **{num_chunks} chunks** into Qdrant!")
                except Exception as e:
                    st.error(f"Error during ingestion: {e}")

    # Database Management
    with st.expander("⚠️ Database Management"):
        if st.button("Clear Vector Database"):
            st.session_state.backend.clear_database()
            st.info("Database cleared.")

    st.divider()

    # --- SECTION 2: SYNTHETIC DATA ---
    st.header("Step 2: Generate Golden Dataset")
    st.info("Uses LLM to generate Questions and Ground Truth Answers from the document. This is your 'Exam' for the RAG system.")

    if "temp_file_path" in st.session_state:
        col1, col2 = st.columns([1, 2])
        with col1:
            num_questions = st.number_input("Number of Questions", min_value=1, max_value=20, value=3)
        
        with col2:
            st.write("") # Spacing
            st.write("") 
            if st.button("⚡ Generate Synthetic Test Data"):
                with st.spinner("Generating Synthetic Data... (This takes 30-60s per question)"):
                    try:
                        # Call the backend generation method
                        df = st.session_state.backend.generate_test_data(
                            st.session_state.temp_file_path, 
                            num_questions
                        )
                        st.session_state.test_data = df # Save to session
                        st.success("✅ Generation Complete!")
                    except Exception as e:
                        st.error(f"Error during generation: {e}")

        # Display & Download
        if "test_data" in st.session_state:
            st.subheader("Generated Test Data")
            
            # Create a clean view for display
            display_df = st.session_state.test_data.copy()
            
            # Rename columns for better readability
            # Note: Ragas column names can vary slightly, so we use a safe rename
            column_mapping = {
                "user_input": "Question",
                "question": "Question", # Fallback
                "reference": "Ground Truth Answer",
                "ground_truth": "Ground Truth Answer", # Fallback
                "reference_contexts": "Source Context",
                "contexts": "Source Context" # Fallback
            }
            display_df = display_df.rename(columns=column_mapping)
            
            # Select only relevant columns to show (hide internal metadata if any)
            cols_to_show = ["Question", "Ground Truth Answer", "Source Context"]
            # Filter to exist columns only
            final_cols = [c for c in cols_to_show if c in display_df.columns]
            
            st.dataframe(display_df[final_cols])
            
            # Convert the ORIGINAL dataframe to CSV for download (keep raw data for logic)
            # OR convert the display one if you prefer the user to download the clean names
            # Let's download the Clean names as it's for the user.
            csv = display_df[final_cols].to_csv(index=False).encode('utf-8')
            
            st.download_button(
                "📥 Download CSV",
                csv,
                "golden_dataset.csv",
                "text/csv",
                key='download-csv'
            )
    else:
        st.markdown("*Please upload a PDF above to enable data generation.*")

elif page == "2. Chat & Live Metrics":
    st.title("💬 Chat & Live Metrics")
    st.markdown("Ask a question to your PDF. The **Chef** answers, and the **Evaluator** grades it instantly.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "metrics" in message:
                # Display metrics for assistant messages
                cols = st.columns(2)
                cols[0].metric("Faithfulness", f"{message['metrics']['faithfulness']:.2f}", help="Is the answer derived from the context?")
                cols[1].metric("Relevancy", f"{message['metrics']['answer_relevancy']:.2f}", help="Does the answer actually address the question?")

    # Accept user input
    if prompt := st.chat_input("Ask a question about your document..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking & Evaluating..."):
                try:
                    result = st.session_state.backend.query_and_evaluate(prompt)
                    
                    answer = result["answer"]
                    metrics = result["metrics"]
                    
                    st.markdown(answer)
                    
                    # Show Metrics immediately
                    cols = st.columns(2)
                    cols[0].metric("Faithfulness", f"{metrics['faithfulness']:.2f}")
                    cols[1].metric("Relevancy", f"{metrics['answer_relevancy']:.2f}")
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "metrics": metrics
                    })
                    
                    # Show Sources in Expander
                    with st.expander("🔍 View Retrieved Context (The Shopper's Basket)"):
                        for i, ctx in enumerate(result["contexts"]):
                            st.caption(f"**Chunk {i+1}:**")
                            st.text(ctx[:300] + "...")

                except Exception as e:
                    st.error(f"Error: {e}")


elif page == "3. Full Evaluation":
    st.title("📊 Deep Evaluation")
    st.info("🚧 Coming in Phase 4...")
