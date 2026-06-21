# RAG Complaint Chatbot

## Overview
An internal AI tool for CrediTrust Financial that turns raw customer complaint data into actionable insights. Built using Retrieval-Augmented Generation (RAG), this chatbot allows Product Managers, Support, and Compliance teams to ask plain-English questions about customer complaints and receive evidence-backed answers.

## Business Problem
CrediTrust Financial receives thousands of complaints per month across four products: Credit Cards, Personal Loans, Savings Accounts, and Money Transfers. This tool reduces the time to identify complaint trends from days to minutes.

## Data Source
Consumer Financial Protection Bureau (CFPB) complaint dataset.

## Project Structure
- `data/raw/` - Original CFPB dataset
- `data/processed/` - Cleaned and filtered complaints
- `vector_store/` - Persisted vector database (ChromaDB/FAISS)
- `notebooks/` - EDA and exploration notebooks
- `src/` - Source code (preprocessing, chunking, RAG pipeline)
- `tests/` - Unit tests
- `app.py` - Gradio/Streamlit chat interface

## Setup
```bash
pip install -r requirements.txt
```

## Usage
```bash
python app.py
```
