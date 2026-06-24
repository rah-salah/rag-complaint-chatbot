import streamlit as st
import os
import sys
sys.path.insert(0, ".")
from sentence_transformers import SentenceTransformer
from src.rag_pipeline import load_vector_store, run_rag_pipeline
from groq import Groq

st.title("CrediTrust Financial - Complaint Analyst")

@st.cache_resource
def load_resources():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index, metadata = load_vector_store()
    client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
    return model, index, metadata, client

model, index, metadata, client = load_resources()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "sources" not in st.session_state:
    st.session_state.sources = ""

col1, col2 = st.columns([2, 1])

with col1:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    question = st.chat_input("Ask about complaints...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("Searching..."):
            result = run_rag_pipeline(question, index, metadata, model, client)
        answer = result["answer"]
        st.session_state.messages.append({"role": "assistant", "content": answer})
        parts = []
        for i, s in enumerate(result["sources"], 1):
            parts.append("Source " + str(i) + " | " + s.get("complaint_id","") + " | " + s.get("product_category","") + chr(10) + s.get("text","")[:300])
        st.session_state.sources = chr(10) + "---" + chr(10).join(parts)
        st.rerun()
    if st.button("Clear"):
        st.session_state.messages = []
        st.session_state.sources = ""
        st.rerun()

with col2:
    st.markdown("### Sources")
    if st.session_state.sources:
        st.markdown(st.session_state.sources)
    else:
        st.info("Sources appear here.")