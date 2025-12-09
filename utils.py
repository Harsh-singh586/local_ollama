from pypdf import PdfReader
import requests

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
OLLAMA_CHAT_URL = "http://localhost:11434/api/generate"


def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def chunk_text(text, chunk_size=800):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks

def embed_text(text):
    payload = {
        "model": "llama3.1",
        "prompt": text
    }

    res = requests.post(OLLAMA_EMBED_URL, json=payload)
    res.raise_for_status()

    embedding = res.json()["embedding"]
    return embedding