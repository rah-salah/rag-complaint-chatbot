import faiss
import json
import os
import logging
from sentence_transformers import SentenceTransformer
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_STORE_DIR = "vector_store"
TOP_K = 5

PROMPT_TEMPLATE = """You are a financial analyst assistant for CrediTrust Financial.
Your role is to help internal teams understand customer complaints.

Use ONLY the context below to answer the question.
If the context does not contain enough information, say: "I don't have enough information in the complaints data to answer this question."
Always cite the complaint IDs you used in your answer.

Context:
{context}

Question: {question}

Answer:"""


def load_vector_store():
    """Load FAISS index and metadata from disk."""
    index_path = os.path.join(VECTOR_STORE_DIR, "complaints.index")
    metadata_path = os.path.join(VECTOR_STORE_DIR, "metadata.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path}.")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found at {metadata_path}.")
    index = faiss.read_index(index_path)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    logger.info(f"Loaded FAISS index with {index.ntotal} vectors.")
    return index, metadata


def retrieve(question: str, index, metadata: list, model: SentenceTransformer, k: int = TOP_K) -> list:
    """Embed question and retrieve top-k most relevant chunks."""
    query_vector = model.encode([question], show_progress_bar=False)
    query_vector = np.array(query_vector, dtype="float32")
    distances, indices = index.search(query_vector, k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(metadata):
            chunk = metadata[idx].copy()
            chunk["score"] = float(dist)
            results.append(chunk)
    logger.info(f"Retrieved {len(results)} chunks for query.")
    return results


def build_context(chunks: list) -> str:
    """Format retrieved chunks into a context string."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[{i}] Complaint ID: {chunk.get('complaint_id', 'N/A')} | "
            f"Product: {chunk.get('product_category', 'N/A')}\n"
            f"{chunk.get('text', '')}"
        )
    return "\n\n".join(parts)


def generate(question: str, context: str, client) -> str:
    """Send prompt to LLM and return the generated answer."""
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return f"Error generating response: {str(e)}"


def run_rag_pipeline(question: str, index, metadata: list, model: SentenceTransformer, client) -> dict:
    """End-to-end RAG: retrieve + generate."""
    chunks = retrieve(question, index, metadata, model)
    context = build_context(chunks)
    answer = generate(question, context, client)
    return {
        "question": question,
        "answer": answer,
        "sources": chunks
    }
