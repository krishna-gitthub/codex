from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List

import httpx
import numpy as np
import pandas as pd
import faiss
from PyPDF2 import PdfReader
import argparse
from nltk.tokenize import sent_tokenize
from tqdm.asyncio import tqdm_asyncio

class Embedder:
    """Client for batch embedding via LM Studio."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        batch_size: int = 32,
        concurrency: int = 4,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(concurrency)


    async def _embed_batch(
        self, client: httpx.AsyncClient, texts: List[str]
    ) -> List[List[float]]:
        async with self.semaphore:
            resp = await client.post(
                f"{self.base_url}/v1/embeddings",
                json={"model": self.model_name, "input": texts},
                timeout=600,
            )
            resp.raise_for_status()
            return [d["embedding"] for d in resp.json()["data"]]

    async def embed_texts(self, texts: List[str]) -> np.ndarray:
        async with httpx.AsyncClient() as client:
            tasks = [
                self._embed_batch(client, texts[i : i + self.batch_size])
                for i in range(0, len(texts), self.batch_size)
            ]
            # Progress bar for async tasks
            results_nested = []
            for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Embedding"):
                results_nested.append(await coro)

        results = [vec for batch in results_nested for vec in batch]
        return np.asarray(results, dtype=np.float32)

"""
    async def embed_texts(self, texts: List[str]) -> np.ndarray:
        async with httpx.AsyncClient() as client:
            tasks = [
                self._embed_batch(client, texts[i : i + self.batch_size])
                for i in range(0, len(texts), self.batch_size)
            ]
            results_nested = await asyncio.gather(*tasks)
        results = [vec for batch in results_nested for vec in batch]
        return np.asarray(results, dtype=np.float32)
"""
def chunk_pdf(pdf_path: Path, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """Split *pdf_path* into text chunks of ``chunk_size`` characters.

    ``overlap`` controls the sliding window so that the start of each chunk
    begins ``chunk_size - overlap`` characters after the previous one.
    """

    reader = PdfReader(str(pdf_path))

    # Collect the document text sentence by sentence for cleaner splits
    sentences: List[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        sentences.extend(sent_tokenize(page_text))

    text = " ".join(sentences)

    step = max(chunk_size - overlap, 1)
    chunks = [
        text[i : i + chunk_size].strip()
        for i in range(0, len(text), step)
        if text[i : i + chunk_size].strip()
    ]

    return chunks




def build_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32)
    index.add(embeddings)
    return index


def save_artifacts(
    index: faiss.Index, chunks: List[str], embeddings: np.ndarray, output_dir: Path
) -> None:
    """Persist FAISS index, chunks, and embeddings under *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_dir / "vector_index.faiss"))
    pd.DataFrame({"chunk": chunks}).to_csv(output_dir / "pdf_chunks.csv", index=False)
    np.save(str(output_dir / "embeddings.npy"), embeddings)


async def main(pdf_path: str, base_url: str, model: str, output_dir: str) -> None:
    chunks = chunk_pdf(Path(pdf_path), chunk_size=512, overlap=64)
    embedder = Embedder(base_url, model)
    embeddings = await embedder.embed_texts(chunks)
    index = build_index(embeddings)
    save_artifacts(index, chunks, embeddings, Path(output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings and FAISS index from a PDF")
    parser.add_argument("--pdf-path", default="PDFs/Mahabharata.pdf")
    parser.add_argument("--base-url", default="http://localhost:1234")
    parser.add_argument("--model", default="text-embedding-qwen3-embedding-0.6b")
    parser.add_argument("--output-dir", default="embeddings/Mahabharatam2")
    args = parser.parse_args()
    asyncio.run(main(args.pdf_path, args.base_url, args.model, args.output_dir))
