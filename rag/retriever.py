"""
Retriever: queries ChromaDB for the most relevant chunks given a user question.

Retrieval strategy:
    Semantic search (top-k) over the embedded chunks. When a branch name
    is detected in the query, we don't hard-filter by it in the database
    query itself (Chroma's 'where' filter needs an exact match on the
    'branches' metadata string, which is unreliable since a chunk can
    list multiple branches). Instead we retrieve a slightly larger top-k
    and let the LLM reason over the branch info embedded in each chunk's
    text. This keeps retrieval simple while still being branch-aware.
"""

import os
import chromadb
from chromadb.utils import embedding_functions

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        _collection = _client.get_collection(
            name="restaurant_knowledge",
            embedding_function=embedding_fn
        )
    return _collection


def retrieve(query: str, n_results: int = 5):
    """
    Returns a list of relevant text chunks for the given query.
    """
    collection = _get_collection()
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    documents = results.get("documents", [[]])[0]
    return documents


def retrieve_as_context(query: str, n_results: int = 5) -> str:
    """
    Returns the retrieved chunks joined into a single context string,
    ready to be inserted into an LLM prompt.
    """
    chunks = retrieve(query, n_results=n_results)
    if not chunks:
        return "No relevant information found in the knowledge base."
    return "\n\n---\n\n".join(chunks)