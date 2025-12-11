# llm.py
import requests
import os
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:latest")
DEFAULT_NUM_PREDICT= int(os.getenv("OLLAMA_NUM_PREDICT", 160))
DEFAULT_KEEP_ALIVE= os.getenv("OLLAMA_KEEP_ALIVE", "10m")

def ollama_generate(
    prompt: str,
    model: str = OLLAMA_MODEL,
    num_predict: int = DEFAULT_NUM_PREDICT,
) -> str:
    """
    Call local Ollama /api/generate (non-streaming).
    num_predict limits how many tokens to generate (smaller = faster).
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": num_predict,
        },
        "keep_alive": DEFAULT_KEEP_ALIVE,
    }

    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        return f"[Error calling Ollama: {e}]"

    data = resp.json()
    return data.get("response", "").strip()



def answer_question(question: str, context_chunks, model: str = OLLAMA_MODEL) -> str:
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