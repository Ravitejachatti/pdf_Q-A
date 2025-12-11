# app.py
import io
import time
import uuid
import os

import numpy as np
import streamlit as st

from dotenv import load_dotenv
from utils.pdf_utils import extract_text_from_pdf, split_text
from utils.embeddings import get_embedder, embed_texts
from utils.vectordb import index_chunks, query_chunks
from utils.llm import answer_question


CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "4"))


st.set_page_config(page_title="PDF Q&A (Ollama + Vector DB)", page_icon="📄")
st.title("📄 PDF Q&A with Ollama (mistral) + Vector DB (Chroma)")
st.write(
    "Upload a PDF, index it into a vector database, and then ask questions.\n"
    "Behind the scenes: PDF → chunks → embeddings → Chroma → Ollama (Mistral)."
)

# -------------------------
# Session state
# -------------------------
if "current_pdf_id" not in st.session_state:
    st.session_state.current_pdf_id = None
if "current_pdf_name" not in st.session_state:
    st.session_state.current_pdf_name = None

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

# -------------------------
# Indexing (one-time per PDF)
# -------------------------
if uploaded_file is not None:
    if st.button("Build Q&A Index (Vector DB)"):
        with st.spinner("Indexing PDF into vector DB (extract → chunk → embed → store)..."):
            total_start = time.time()

            # Read bytes
            pdf_bytes = uploaded_file.read()
            pdf_name = uploaded_file.name

            # 1) Extract text
            t0 = time.time()
            pdf_io = io.BytesIO(pdf_bytes)
            text = extract_text_from_pdf(pdf_io)
            t_extract = time.time() - t0

            if not text.strip():
                st.error("No extractable text found in the PDF.")
            else:
                # 2) Chunk text
                t1 = time.time()
                chunks = split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
                t_chunk = time.time() - t1

                # 3) Embed chunks
                t2 = time.time()
                embedder = get_embedder()
                embeddings = embed_texts(chunks, embedder=embedder)
                t_embed = time.time() - t2

                # 4) Store in vector DB
                t3 = time.time()
                pdf_id = str(uuid.uuid4())
                index_chunks(pdf_id, pdf_name, chunks, embeddings)
                t_store = time.time() - t3

                t_total = time.time() - total_start

                st.session_state.current_pdf_id = pdf_id
                st.session_state.current_pdf_name = pdf_name

                st.success(
                    f"Indexed PDF ✅\n\n"
                    f"- Name: `{pdf_name}`\n"
                    f"- PDF ID: `{pdf_id}`"
                )
                st.info(
                    f"⏱️ Timing (indexing):\n"
                    f"- Extract text: {t_extract:.2f} s\n"
                    f"- Chunk text: {t_chunk:.2f} s\n"
                    f"- Embeddings: {t_embed:.2f} s\n"
                    f"- Store in DB: {t_store:.2f} s\n"
                    f"- **Total indexing**: {t_total:.2f} s"
                )

# -------------------------
# Q&A section
# -------------------------
if st.session_state.current_pdf_id:
    st.info(
        f"Current PDF for Q&A:\n\n"
        f"- **Name:** {st.session_state.current_pdf_name}\n"
        f"- **ID:** `{st.session_state.current_pdf_id}`"
    )

    st.subheader("Ask questions about this PDF")

    question = st.text_input("Your question")

    if question:
        if st.button("Get Answer"):
            with st.spinner("Retrieving from vector DB and querying Ollama..."):
                pdf_id = st.session_state.current_pdf_id

                # 1) Embed question + retrieve chunks
                qa_start = time.time()
                embedder = get_embedder()

                t_q_emb_start = time.time()
                q_emb = embed_texts(question, embedder=embedder)[0]
                t_q_emb = time.time() - t_q_emb_start

                t_retrieve_start = time.time()
                top_chunks, metadatas = query_chunks(
                    pdf_id=pdf_id,
                    query_embedding=q_emb,
                    top_k=TOP_K,
                )
                t_retrieve = time.time() - t_retrieve_start

                if not top_chunks:
                    st.warning("No relevant chunks found in the vector DB.")
                else:
                    # 2) LLM answer
                    t_llm_start = time.time()
                    answer = answer_question(question, top_chunks)
                    t_llm = time.time() - t_llm_start

                    qa_total = time.time() - qa_start

                    st.markdown("### Answer")
                    st.write(answer)

                    st.caption(
                        f"⏱️ Timing (Q&A): "
                        f"embed question {t_q_emb:.2f}s · "
                        f"retrieve {t_retrieve:.2f}s · "
                        f"LLM {t_llm:.2f}s · "
                        f"total {qa_total:.2f}s"
                    )

                    with st.expander("Show retrieved context (vector DB chunks)"):
                        for i, (ch, meta) in enumerate(zip(top_chunks, metadatas), start=1):
                            st.markdown(
                                f"**Chunk {i} (index={meta.get('chunk_index')}, "
                                f"pdf_name={meta.get('pdf_name')})**"
                            )
                            st.code(ch, language="text")
else:
    st.info(
        "👆 Upload a PDF and click **Build Q&A Index (Vector DB)** to start.\n\n"
        "PDF content is stored as embeddings in a persistent Chroma DB with a unique PDF ID. "
        "You can then ask questions, and the system retrieves relevant chunks and sends them "
        "to Ollama (Mistral) to generate answers."
    )