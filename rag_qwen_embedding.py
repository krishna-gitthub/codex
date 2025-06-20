import requests
import numpy as np
import pandas as pd
import faiss
from PyPDF2 import PdfReader
from tqdm import tqdm

# === CONFIG ===
pdf_path = "PDFs//2024_Indian_farmers'_protest.pdf"  # 👈 Replace this with your PDF
base_url = "http://localhost:1234"
model_name = "text-embedding-qwen3-embedding-4b"
chunk_size = 512
index_path = "embeddings\\farm_laws\\vector_index.faiss"
chunk_path = "embeddings\\farm_laws\\pdf_chunks.csv"

# === STEP 1: Chunk PDF ===
def load_pdf_text_chunks(pdf_path, chunk_size=512):
    reader = PdfReader(pdf_path)
    chunks = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            chunks.extend([text[i:i+chunk_size] for i in range(0, len(text), chunk_size)])
    return chunks

# === STEP 2: Get Embeddings via LM Studio ===
def get_embedding(texts, model_name, base_url):
    response = requests.post(
        f"{base_url}/v1/embeddings",
        headers={"Content-Type": "application/json"},
        json={"model": model_name, "input": texts}
    )
    response.raise_for_status()
    return [entry["embedding"] for entry in response.json()["data"]]

# === STEP 3: Embed All Chunks ===
def build_vector_store(chunks, model_name, base_url, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding chunks"):
        batch = chunks[i:i+batch_size]
        vectors = get_embedding(batch, model_name, base_url)
        embeddings.extend(vectors)
    return np.array(embeddings, dtype=np.float32)

# === STEP 4: Save FAISS & Metadata ===
def save_faiss_index(embeddings, chunks, index_path, chunk_path):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    pd.DataFrame({'chunk': chunks}).to_csv(chunk_path, index=False)
    np.save("embeddings\\farm_laws\\embeddings.npy", embeddings)

# === RUN ===
chunks = load_pdf_text_chunks(pdf_path, chunk_size)
embeddings = build_vector_store(chunks, model_name, base_url)
save_faiss_index(embeddings, chunks, index_path, chunk_path)
