import requests
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# === CONFIG ===
base_url = "http://localhost:1234"
chat_model = "qwen/qwen3-4b"  # or any other chat model loaded in LM Studio
index_path = "embeddings/farm_laws/vector_index.faiss"
chunk_path = "embeddings/farm_laws/pdf_chunks.csv"
embedding_path = "embeddings/farm_laws/embeddings.npy"
top_k = 20

# === LOAD EMBEDDINGS + CHUNKS ===
embeddings = np.load(embedding_path)
chunks_df = pd.read_csv(chunk_path)

# === SEARCH FUNCTION ===
def retrieve_relevant_chunks(user_query, top_k=20):
    # Get embedding for the query using the same model
    response = requests.post(
        f"{base_url}/v1/embeddings",
        headers={"Content-Type": "application/json"},
        json={"model": "text-embedding-qwen3-embedding-4b", "input": [user_query]}
    )
    response.raise_for_status()
    query_embedding = np.array(response.json()["data"][0]["embedding"]).reshape(1, -1)

    # Compute cosine similarity
    similarities = cosine_similarity(query_embedding, embeddings).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return chunks_df.iloc[top_indices]['chunk'].tolist()

# === CHAT CALL ===
def ask_with_context(user_query):
    context_chunks = retrieve_relevant_chunks(user_query, top_k)
    context = "\n\n".join(context_chunks)

    prompt = f"""[Context]
{context}

[User Question]
{user_query}

[Answer]
"""
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": chat_model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant who answers based only on the provided context."},
                {"role": "user", "content": prompt}
            ]
        }
    )
    return response.json()["choices"][0]["message"]["content"]

# === USAGE ===
while True:
    query = input("Ask your question: ")
    answer = ask_with_context(query)
    print("\n🔍 Answer:\n", answer, "\n\n")
