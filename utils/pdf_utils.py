# pdf_utils.py
from PyPDF2 import PdfReader


def extract_text_from_pdf(file_like) -> str:
    """Read all pages from the uploaded PDF and return plain text."""
    reader = PdfReader(file_like)
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