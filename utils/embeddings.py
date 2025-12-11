# embeddings.py
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# Load the sentence-transformer model from env
sentence_transformer_model = os.getenv("SENTENCE_TRANSFORMER", "all-MiniLM-L6-v2")
device = os.getenv("OLLAMA_DEVICE", "cpu")  # cpu, gpu, mps

@st.cache_resource
def get_embedder() -> SentenceTransformer:
    """Load and cache the sentence-transformer model."""
    return SentenceTransformer(sentence_transformer_model, device=device)


def embed_texts(texts, embedder: SentenceTransformer = None, normalize: bool = True):
    """
    Get embeddings for a list of texts.
    If normalize=True, cosine similarity = dot product.
    """
    if embedder is None:
        embedder = get_embedder()

    if isinstance(texts, str):
        texts = [texts]

    embs = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=normalize)
    return embs.astype("float32")