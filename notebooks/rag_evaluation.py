import sys
import os
sys.path.insert(0, ".")

from sentence_transformers import SentenceTransformer
from src.rag_pipeline import load_vector_store, run_rag_pipeline
from groq import Groq

model = SentenceTransformer("all-MiniLM-L6-v2")
index, metadata = load_vector_store()
client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

test_questions = [
    "What are the most common Credit Card complaints?",
    "Why do customers complain about Money Transfer delays?",
    "What issues do customers report with Personal Loans?",
    "What are common Savings Account problems?",
    "How do customers describe billing disputes on Credit Cards?",
]

print("=" * 70)
for q in test_questions:
    print(f"\nQ: {q}")
    result = run_rag_pipeline(q, index, metadata, model, client)
    print(f"A: {result['answer'][:400]}")
    print(f"Sources: {[s['complaint_id'] for s in result['sources']]}")
    print("-" * 70)
