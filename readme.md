#  MacvarrAI – Local PDF Q&A with Ollama + Chroma

Ask questions about any PDF **locally on your Device**, using:

-  **Ollama** (local LLM – e.g. `mistral:latest`)
-  **Sentence-Transformers** for embeddings
-  **Chroma** as a persistent vector database
-  **Streamlit** for a simple web UI

No data leaves your machine. This is a small, modular **RAG (Retrieval-Augmented Generation)** demo you can also explain as a research-style architecture.

---

##  Features

- Upload any PDF and **extract text** automatically
- Split text into overlapping **chunks** for better retrieval
- Generate dense **embeddings** using `all-MiniLM-L6-v2`
- Store chunks + embeddings in **Chroma** (persistent `chroma_db/` folder)
- Ask questions and get answers:
  - Embed the question
  - Retrieve top-K relevant chunks from the vector DB
  - Send context + question to a **local Ollama model**
- Show detailed **timings**:
  - Indexing: extract / chunk / embed / store
  - Q&A: question embedding / retrieval / LLM / total

---

##  Tech Stack

- **Python** 3.9+
- **Streamlit** – UI / orchestration
- **PyPDF2** – PDF text extraction
- **Sentence-Transformers** – embeddings (`all-MiniLM-L6-v2`)
- **ChromaDB** – vector database (persistent)
- **Ollama** – local LLM server (e.g. `mistral:latest`)
- **NumPy**, **python-dotenv**, etc.

---

##  Project Structure

```bash
pdf-qa-ollama/
├── .venv/                  # Python virtual environment (local)
├── chroma_db/              # Chroma persistent storage (auto-created)
├── utils/
│   ├── __init__.py
│   ├── embeddings.py       # Sentence-transformer loading & embedding
│   ├── llm.py              # Ollama HTTP calls + answer function
│   ├── pdf_utils.py        # PDF text extraction & chunking
│   └── vectordb.py         # Chroma client, index & query helpers
├── .env                    # Configuration (LLM, chunking, paths)
├── app.py                  # Main Streamlit app (vector DB + timings)
├── app_ollama_pdf_qa.py    # Older simple in-memory version (optional)
├── requirements.txt
└── README.md               # (this file)


## Commands 

# 1. Clone
git clone <your-github-repo-url>
cd pdf-qa-ollama   # or whatever name

# 2. Create venv
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install deps
pip install -r requirements.txt

# 4. Create .env from template
cp .env.example .env
# (edit .env if they want a different model)

# 5. Install & start Ollama + model (once)
# https://ollama.com
ollama pull mistral:latest
ollama serve    # in a separate terminal

# 6. Run the app
streamlit run app.py