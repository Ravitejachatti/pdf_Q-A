# vectordb.py
import os
import chromadb
from chromadb.config import Settings

CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")

_client = None
_collection = None


def get_client():
    """Create or reuse a persistent Chroma client."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=CHROMA_DB_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def get_collection():
    """Create or reuse a collection for PDF chunks."""
    global _collection
    if _collection is None:
        client = get_client()
        _collection = client.get_or_create_collection(name="pdf_qa_chunks")
    return _collection


def index_chunks(pdf_id: str, pdf_name: str, chunks, embeddings):
    """
    Store chunks + embeddings in Chroma, tagged with pdf_id.
    embeddings is a numpy array of shape (N, dim).
    """
    collection = get_collection()

    ids = [f"{pdf_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "pdf_id": pdf_id,
            "pdf_name": pdf_name,
            "chunk_index": i,
        }
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist(),  # Chroma expects Python lists
        metadatas=metadatas,
    )


def query_chunks(pdf_id: str, query_embedding, top_k: int = 4):
    """
    Query top_k chunks for this pdf_id given a query embedding.
    Returns (documents, metadatas).
    """
    collection = get_collection()

    result = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        where={"pdf_id": pdf_id},
    )

    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    return docs, metas