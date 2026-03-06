# RAG Document Search

Production-style Retrieval-Augmented Generation (RAG) system for semantic document search using LLM embeddings and vector databases.

## Features

- semantic search over documents
- embedding generation
- vector similarity search
- LLM-based answer generation

## Tech Stack

- Python
- OpenAI API
- FAISS / Chroma
- LangChain

## Architecture

User Query
   ↓
Embedding Model
   ↓
Vector Database
   ↓
Relevant Documents
   ↓
LLM Response

## Installation

pip install -r requirements.txt

## Run

python main.py

## Example

Query:
"What is vector search?"

Answer:
Vector search retrieves documents based on semantic similarity rather than keyword matching.
