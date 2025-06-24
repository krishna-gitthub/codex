from __future__ import annotations

from pathlib import Path
from typing import List
import argparse
import httpx
import faiss
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import re
import textwrap


class Retriever:
    def __init__(self, base_url: str, embed_model: str, index_path: Path, chunk_path: Path) -> None:
        self.base_url = base_url.rstrip("/")
        self.embed_model = embed_model
        self.index = faiss.read_index(str(index_path))
        df = pd.read_csv(chunk_path)
        self.metadata = df.to_dict(orient="records")
        self.embeddings = np.load(index_path.parent / "embeddings.npy")
        self.embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)

    def embed_query(self, query: str) -> np.ndarray:
        resp = httpx.post(
            f"{self.base_url}/v1/embeddings",
            json={"model": self.embed_model, "input": [query]},
            timeout=30,
        )
        resp.raise_for_status()
        vec = np.array(resp.json()["data"][0]["embedding"], dtype=np.float32)
        vec = vec / np.linalg.norm(vec)
        return vec.reshape(1, -1)

    def has_number(self, text: str) -> bool:
        return bool(re.search(r"\d+", text))

    def search(self, query: str, k: int = 20) -> List[str]:
        qv = self.embed_query(query)
        scores = cosine_similarity(qv, self.embeddings)[0]
        ranked = [
            (score, self.metadata[i])
            for i, score in enumerate(scores)
            if "chunk" in self.metadata[i] and self.has_number(self.metadata[i]["chunk"])
        ]
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [entry["chunk"] for _, entry in ranked[:k]]


class ChatAssistant:
    def __init__(self, base_url: str, chat_model: str, retriever: Retriever) -> None:
        self.base_url = base_url.rstrip("/")
        self.chat_model = chat_model
        self.retriever = retriever

    def answer(self, query: str, k: int = 5) -> str:
        context_chunks = self.retriever.search(query, k)
        if not context_chunks:
            return "No relevant context found."
        context = "\n\n".join(context_chunks)
        prompt = textwrap.dedent(
            f"""\
            You are a reasoning assistant. Use the provided context to compute and explain your answer step by step.
            If the answer requires math, show logical estimates.

            [Context]
            {context}

            [User Question]
            {query}

            [Answer]
            """
        )
        resp = httpx.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.chat_model,
                "messages": [
                    {"role": "system", "content": "Answer using only the provided context."},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with reranked retrieval-augmented context")
    parser.add_argument("--base-url", default="http://localhost:1234")
    parser.add_argument("--embed-model", default="text-embedding-qwen3-embedding-4b")
    parser.add_argument("--chat-model", default="qwen/qwen3-4b")
    parser.add_argument("--index-path", default="embeddings/farm_laws-upgrade-2-NLTK/vector_index.faiss")
    parser.add_argument("--chunk-path", default="embeddings/farm_laws-upgrade-2-NLTK/pdf_chunks.csv")
    args = parser.parse_args()

    retriever = Retriever(
        base_url=args.base_url,
        embed_model=args.embed_model,
        index_path=Path(args.index_path),
        chunk_path=Path(args.chunk_path),
    )
    assistant = ChatAssistant(
        base_url=args.base_url,
        chat_model=args.chat_model,
        retriever=retriever,
    )
    while True:
        q = input("Ask your question: ")
        print("\n🔍 Answer:\n", assistant.answer(q, k=20), "\n")


if __name__ == "__main__":
    main()
