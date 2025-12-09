
import os
from flask import Flask, request, jsonify
import uuid
import chromadb
import requests
from utils import extract_text_from_pdf, chunk_text, embed_text


app = Flask(__name__)

chroma_client = chromadb.PersistentClient(path="db")
collection = chroma_client.get_or_create_collection(name="documents")

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
OLLAMA_CHAT_URL = "http://localhost:11434/api/generate"




def chat_with_context(question, context):
    prompt = f"""
            You are a helpful assistant. Use the context below to answer.

            Context:
            {context}

            Question:
            {question}

            Answer:
            """

    payload = {
        "model": os.getenv("CHAT_MODEL", "llama3.1"),
        "prompt": prompt,
        "stream": False
    }

    res = requests.post(OLLAMA_CHAT_URL, json=payload)
    res.raise_for_status()
    return res.json()["response"]


@app.route("/upload", methods=["POST"])
def upload_document():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_path = f"documents/{file.filename}"
    file.save(file_path)

    # Extract text
    text = extract_text_from_pdf(file_path)
    print(text)

    # Chunk text
    chunks = chunk_text(text)

    # Embed and store in ChromaDB
    for chunk in chunks:
        emb = embed_text(chunk)
        collection.add(
            ids=[str(uuid.uuid4())],
            metadatas=[{"source": file.filename}],
            embeddings=[emb],
            documents=[chunk]
        )

    return jsonify({"message": "Document uploaded and indexed successfully"})


@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.json
    question = data.get("question")

    # Embed the question
    q_emb = embed_text(question)

    # Search top relevant chunks
    results = collection.query(query_embeddings=[q_emb], n_results=3)

    retrieved_chunks = "\n\n".join(results["documents"][0])

    # Generate final answer using LLM
    answer = chat_with_context(question, retrieved_chunks)

    return jsonify({
        "answer": answer
    })


@app.route("/", methods=["GET"])
def home():
    return {"message": "Local PDF RAG API running!"}


if __name__ == "__main__":
    os.makedirs("documents", exist_ok=True)
    app.run(host="0.0.0.0", port=5001)
