"""
Ingestion script: reads the raw text documents (menu, policies, about us),
splits them into chunks, embeds them, and stores them in ChromaDB.

Run this once before starting the app:
    python rag/ingest.py

Chunking strategy:
    Each document is split by its natural "record" boundaries (blank lines
    between dish entries, or section headers "===...==="). This keeps each
    chunk semantically self-contained - a chunk is either one full dish
    entry or one full policy paragraph - instead of splitting mid-sentence
    with a fixed character length. This avoids cutting a dish's price away
    from its name or its vegetarian flag, which would hurt retrieval
    accuracy and could cause hallucinated or incomplete answers.

Embedding model:
    all-MiniLM-L6-v2 (via sentence-transformers). Chosen because it is
    small, fast, runs fully locally with no API cost, and performs well
    on short semantic-similarity tasks like ours (dish descriptions,
    short policy paragraphs).
"""

import os
import re
import chromadb
from chromadb.utils import embedding_functions

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")


def split_menu_into_chunks(text, source_file):
    """
    Menu file uses blank-line-separated entries and '=== Section ===' headers.
    Each dish entry becomes one chunk, tagged with its section and branch info.
    """
    chunks = []
    current_section = "General"
    blocks = text.split("\n\n")

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        section_match = re.match(r"=== (.+) ===", block)
        if section_match:
            current_section = section_match.group(1)
            continue

        # Extract branch info from the block text for metadata filtering
        branch_match = re.search(r"Available in branches?: (.+)", block)
        branches_text = branch_match.group(1) if branch_match else "All branches"

        chunks.append({
            "text": block,
            "metadata": {
                "source": source_file,
                "section": current_section,
                "branches": branches_text
            }
        })

    return chunks


def split_generic_into_chunks(text, source_file):
    """
    For policies.txt and about_us.txt: split by section headers,
    each section becomes one chunk (keeps related sentences together).
    """
    chunks = []
    sections = re.split(r"=== (.+?) ===", text)

    # re.split with a capturing group returns: [before, header1, content1, header2, content2, ...]
    for i in range(1, len(sections), 2):
        header = sections[i].strip()
        content = sections[i + 1].strip() if i + 1 < len(sections) else ""
        if content:
            chunks.append({
                "text": f"{header}: {content}",
                "metadata": {
                    "source": source_file,
                    "section": header
                }
            })

    return chunks


def load_and_chunk_all_documents():
    all_chunks = []

    menu_path = os.path.join(DATA_DIR, "menu.txt")
    with open(menu_path, "r", encoding="utf-8") as f:
        all_chunks.extend(split_menu_into_chunks(f.read(), "menu.txt"))

    for filename in ["policies.txt", "about_us.txt"]:
        path = os.path.join(DATA_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            all_chunks.extend(split_generic_into_chunks(f.read(), filename))

    return all_chunks


def run_ingestion():
    print("Loading and chunking documents...")
    chunks = load_and_chunk_all_documents()
    print(f"Created {len(chunks)} chunks.")

    print("Setting up ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Reset the collection each time we ingest, so re-running this script
    # doesn't duplicate entries.
    try:
        client.delete_collection("restaurant_knowledge")
    except Exception:
        pass

    collection = client.create_collection(
        name="restaurant_knowledge",
        embedding_function=embedding_fn
    )

    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"Ingestion complete. {len(chunks)} chunks stored in ChromaDB at '{CHROMA_DIR}'.")


if __name__ == "__main__":
    run_ingestion()