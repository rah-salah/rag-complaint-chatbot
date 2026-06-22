import polars as pl
import faiss
import numpy as np
import json
import os
import logging
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SAMPLE_SIZE = 10000
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_STORE_DIR = "vector_store"


def stratified_sample(df: pl.DataFrame, total: int) -> pl.DataFrame:
    """Sample complaints proportionally across product categories."""
    counts = df["product_category"].value_counts()
    total_rows = df.shape[0]
    samples = []
    for row in counts.iter_rows(named=True):
        product = row["product_category"]
        proportion = row["count"] / total_rows
        n = max(1, int(proportion * total))
        subset = df.filter(pl.col("product_category") == product).sample(n=n, seed=42)
        samples.append(subset)
    result = pl.concat(samples)
    logger.info(f"Stratified sample: {result.shape[0]} complaints")
    return result


def chunk_complaints(df: pl.DataFrame) -> list:
    """Split narratives into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = []
    for row in df.iter_rows(named=True):
        text = row.get("cleaned_narrative", "") or ""
        if not text.strip():
            continue
        splits = splitter.split_text(text)
        for i, chunk in enumerate(splits):
            chunks.append({
                "complaint_id": str(row.get("complaint_id", "")),
                "product_category": row["product_category"],
                "chunk_index": i,
                "text": chunk,
            })
    logger.info(f"Total chunks: {len(chunks)}")
    return chunks


def embed_and_index(chunks):
    """Generate embeddings and save FAISS index + metadata."""
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    logger.info(f"Loading model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["text"] for c in chunks]
    logger.info("Generating embeddings...")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, os.path.join(VECTOR_STORE_DIR, "complaints.index"))
    metadata = [{k: v for k, v in c.items() if k != "text"} for c in chunks]
    with open(os.path.join(VECTOR_STORE_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f)
    logger.info(f"Saved FAISS index + metadata. Vectors: {index.ntotal}")


def main():
    if not os.path.exists("data/filtered_complaints.csv"):
        raise FileNotFoundError("data/filtered_complaints.csv not found. Run preprocessing first.")
    logger.info("Loading filtered complaints...")
    df = pl.read_csv("data/filtered_complaints.csv")
    logger.info(f"Loaded: {df.shape}")
    sample = stratified_sample(df, SAMPLE_SIZE)
    chunks = chunk_complaints(sample)
    embed_and_index(chunks)
    logger.info("Task 2 complete!")


if __name__ == "__main__":
    main()
