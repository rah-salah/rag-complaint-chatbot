# RAG Complaint Chatbot — CrediTrust Financial

An intelligent complaint analysis system using Retrieval-Augmented Generation (RAG) to help CrediTrust Financial understand and respond to customer complaints.

## Project Structure
rag-complaint-chatbot/

├── data/

│   ├── raw/                  # Raw CFPB dataset (not tracked in git)

│   ├── processed/            # EDA summary stats and charts

│   └── filtered_complaints.csv  # Filtered dataset (not tracked in git)

├── vector_store/             # FAISS index and metadata

├── notebooks/

│   └── eda-preprocessing.ipynb  # EDA notebook

├── src/

│   ├── preprocessing.py      # Data cleaning and filtering pipeline

│   └── chunking_embedding.py # Chunking, embedding, and vector store

├── tests/

│   └── test_preprocessing.py # Unit tests

├── .github/workflows/        # CI pipeline

└── requirements.txt

## Setup

1. Clone the repo:
```bash
git clone https://github.com/rah-salah/rag-complaint-chatbot.git
cd rag-complaint-chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the CFPB dataset from https://www.consumerfinance.gov/data-research/consumer-complaints/ and place it at `data/raw/complaints.csv`

## Running the Pipeline

**Task 1 — Preprocessing:**
```bash
PYTHONPATH=. python src/preprocessing.py
```
Filters 9.6M CFPB complaints to 478,818 records across 4 target products.

**Task 2 — Embedding:**
```bash
PYTHONPATH=. python src/chunking_embedding.py
```
Stratified sample of 10,000 complaints → 27,591 chunks → FAISS vector store.

**Tests:**
```bash
PYTHONPATH=. pytest tests/ -v
```

## Data Source

Consumer Financial Protection Bureau (CFPB) Consumer Complaint Database.  
Target products: Credit Card, Personal Loan, Savings Account, Money Transfer.

## Key Findings (EDA)

- Full CFPB dataset: 9,609,797 complaints
- After filtering to 4 target products + removing empty narratives: 478,818 complaints
- Median narrative length: 132 words (mean: 196, max: 6,239)
- Chunk size: 500 characters with 50 character overlap
- Embedding model: sentence-transformers/all-MiniLM-L6-v2
