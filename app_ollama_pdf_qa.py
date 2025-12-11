import io
import requests
import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

# ==========================
#  MODEL + EMBEDDER
# ==========================

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "mistral:latest"  # you already have this model


@st.cache_resource
def get_embedder():
    """Load and cache the sentence-transformer model."""
    return SentenceTransformer("all-MiniLM-L6-v2")


# ==========================
#  PDF & TEXT HELPERS
# ==========================

def extract_text_from_pdf(uploaded_file) -> str:
    """Read all pages from the uploaded PDF and return plain text."""
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"
    return text


def split_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    """
    Split text into overlapping chunks so each chunk is ~chunk_size chars.
    Overlap keeps context between chunks.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


# ==========================
#  EMBEDDINGS + RETRIEVAL
# ==========================

def get_embeddings(texts, embedder: SentenceTransformer):
    """
    Get normalized embeddings for a list of texts.
    Using normalized embeddings makes cosine similarity = dot product.
    """
    if isinstance(texts, str):
        texts = [texts]

    embs = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embs.astype("float32")


def retrieve_relevant_chunks(question, chunks, chunk_embeddings, embedder, top_k=4):
    """Return top_k chunks most similar to the question."""
    if not chunks:
        return []

    q_emb = get_embeddings(question, embedder)[0]  # shape: (dim,)
    # cosine similarity = dot product because we normalized embeddings
    similarities = chunk_embeddings @ q_emb
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [chunks[i] for i in top_indices]


# ==========================
#  OLLAMA LLM CALL
# ==========================

def ollama_generate(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """
    Call local Ollama /api/generate (non-streaming).
    Make sure Ollama is running at http://localhost:11434.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        return f"[Error calling Ollama: {e}]"

    data = resp.json()
    # /api/generate returns a single JSON with 'response' when stream=False
    return data.get("response", "").strip()


def answer_question(question, context_chunks, model: str = OLLAMA_MODEL) -> str:
    """
    Use Ollama LLM to answer using only the given context.
    """
    context_text = "\n\n---\n\n".join(context_chunks)

    prompt = (
        "You are a helpful assistant answering questions about a PDF document.\n"
        "You must use ONLY the context provided below. If the answer is not "
        "present in the context, say you don't know.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n"
        "Answer in a clear, concise way.\n"
        "Answer:"
    )

    raw = ollama_generate(prompt, model=model)

    # Optional: try to trim to part after 'Answer:'
    if "Answer:" in raw:
        return raw.split("Answer:", 1)[-1].strip()
    return raw.strip()


# ==========================
#  STREAMLIT UI
# ==========================

st.set_page_config(page_title="PDF Q&A (Ollama + Open-Source)", page_icon="📄")
st.title("📄 PDF Q&A with Ollama (mistral) + Sentence-Transformers")
st.write("Upload a PDF, build an index, and ask questions based on its content.")

# Session state for chunks and embeddings
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "chunk_embeddings" not in st.session_state:
    st.session_state.chunk_embeddings = None

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Extracting text from PDF..."):
        # Read bytes and wrap in BytesIO for PdfReader
        pdf_bytes = uploaded_file.read()
        pdf_io = io.BytesIO(pdf_bytes)
        text = extract_text_from_pdf(pdf_io)

    if not text.strip():
        st.error("Could not extract any text from this PDF.")
    else:
        st.success("Text extracted from PDF ✅")
        st.write("**Preview of extracted text (first 1000 characters):**")
        st.code(text[:1000] + ("..." if len(text) > 1000 else ""), language="text")

        if st.button("Build Q&A Index (Create Embeddings)"):
            with st.spinner("Loading embedder & creating embeddings..."):
                embedder = get_embedder()
                chunks = split_text(text, chunk_size=1000, overlap=200)
                embeddings = get_embeddings(chunks, embedder)

                st.session_state.chunks = chunks
                st.session_state.chunk_embeddings = embeddings

            st.success(f"Index built ✅  ({len(chunks)} chunks)")


# Q&A section
if st.session_state.chunks is not None and st.session_state.chunk_embeddings is not None:
    st.subheader("Ask questions about your PDF")

    question = st.text_input("Your question")

    if question:
        if st.button("Get Answer"):
            with st.spinner("Querying Ollama (mistral:latest)..."):
                embedder = get_embedder()

                top_chunks = retrieve_relevant_chunks(
                    question,
                    st.session_state.chunks,
                    st.session_state.chunk_embeddings,
                    embedder,
                    top_k=4,
                )
                answer = answer_question(question, top_chunks)

            st.markdown("### Answer")
            st.write(answer)

            with st.expander("Show context used"):
                for i, ch in enumerate(top_chunks, start=1):
                    st.markdown(f"**Chunk {i}:**")
                    st.code(ch, language="text")
else:
    st.info("👆 Upload a PDF and click **Build Q&A Index** to start.")