"""
Similarity Engine — Mac M2 Optimized
Uses FAISS + sentence-transformers (BAAI/bge-small-en-v1.5) for fast, 
accurate semantic matching. Embeddings are cached to avoid recomputation.
"""

import json
import pickle
import hashlib
import numpy as np
from pathlib import Path
from typing import Optional

# Lazy-loaded to avoid startup overhead
_model = None
_faiss_index = None

RESUME_PATH = Path(__file__).parent.parent / "resume" / "resume.json"
CACHE_DIR = Path(__file__).parent.parent / "embeddings"
EMBEDDINGS_CACHE = CACHE_DIR / "embeddings_cache.json"
RESUME_INDEX_PATH = CACHE_DIR / "resume.faiss"

SIMILARITY_THRESHOLD = 0.60  # Only pass jobs above this score to LLM
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # Small, fast, high accuracy


def _get_model():
    """Lazy-load embedding model to avoid startup cost."""
    global _model
    if _model is None:
        print("  📦 Loading embedding model (first time, will cache)...")
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"  ✅ Model loaded: {EMBEDDING_MODEL}")
    return _model


def _hash_text(text: str) -> str:
    """Create a short hash for cache key."""
    return hashlib.md5(text.encode()).hexdigest()[:16]


def _load_cache() -> dict:
    """Load embedding cache from disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if EMBEDDINGS_CACHE.exists():
        with open(EMBEDDINGS_CACHE, "r") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict):
    """Save embedding cache to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(EMBEDDINGS_CACHE, "w") as f:
        json.dump(cache, f)


def embed_text(text: str, use_cache: bool = True) -> np.ndarray:
    """
    Embed a single text string.
    Uses disk cache to avoid recomputing embeddings.
    """
    cache = _load_cache() if use_cache else {}
    key = _hash_text(text)

    if key in cache:
        return np.array(cache[key], dtype=np.float32)

    model = _get_model()
    embedding = model.encode(text, normalize_embeddings=True)

    if use_cache:
        cache[key] = embedding.tolist()
        _save_cache(cache)

    return embedding.astype(np.float32)


def embed_batch(texts: list[str], use_cache: bool = True) -> np.ndarray:
    """
    Embed a batch of texts efficiently.
    Only recomputes uncached embeddings — Mac M2 optimized.
    """
    cache = _load_cache() if use_cache else {}

    results = {}
    uncached_texts = []
    uncached_indices = []

    # Separate cached vs uncached
    for i, text in enumerate(texts):
        key = _hash_text(text)
        if key in cache:
            results[i] = np.array(cache[key], dtype=np.float32)
        else:
            uncached_texts.append(text)
            uncached_indices.append(i)

    # Batch encode only what's missing
    if uncached_texts:
        model = _get_model()
        print(f"  🔢 Encoding {len(uncached_texts)} new texts in batch...")
        embeddings = model.encode(
            uncached_texts,
            normalize_embeddings=True,
            batch_size=32,  # Reasonable for M2 RAM
            show_progress_bar=len(uncached_texts) > 10,
        )

        for i, (idx, text) in enumerate(zip(uncached_indices, uncached_texts)):
            results[idx] = embeddings[i].astype(np.float32)
            if use_cache:
                cache[_hash_text(text)] = embeddings[i].tolist()

        if use_cache:
            _save_cache(cache)

    # Return in original order
    return np.array([results[i] for i in range(len(texts))], dtype=np.float32)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity between two normalized vectors."""
    # Vectors are already normalized from sentence_transformers
    return float(np.dot(vec_a, vec_b))


def build_resume_faiss_index() -> tuple:
    """
    Build FAISS index from resume chunks.
    Returns (index, chunks) tuple.
    """
    import faiss

    # Load resume knowledge base
    with open(RESUME_PATH) as f:
        resume = json.load(f)

    chunks = resume.get("chunks", [])
    if not chunks:
        raise ValueError("Resume chunks are empty. Please update resume.json")

    print(f"  📄 Building FAISS index from {len(chunks)} resume chunks...")
    embeddings = embed_batch(chunks)

    # Build flat L2 index (accurate, fast for small datasets)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product = cosine for normalized vecs
    index.add(embeddings)

    # Save index
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(RESUME_INDEX_PATH))

    # Save chunks alongside
    chunks_path = CACHE_DIR / "resume_chunks.json"
    with open(chunks_path, "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"  ✅ FAISS index built and saved ({dimension}d, {len(chunks)} vectors)")
    return index, chunks


def load_resume_faiss_index() -> tuple:
    """Load existing FAISS index, or build it if missing."""
    import faiss

    chunks_path = CACHE_DIR / "resume_chunks.json"

    if RESUME_INDEX_PATH.exists() and chunks_path.exists():
        index = faiss.read_index(str(RESUME_INDEX_PATH))
        with open(chunks_path) as f:
            chunks = json.load(f)
        return index, chunks

    return build_resume_faiss_index()


def score_job_against_resume(job_description: str,
                              job_title: str = "",
                              top_k: int = 5) -> dict:
    """
    Score a job description against the resume using FAISS.
    Returns:
        {
            "score": 0.73,
            "top_matches": ["chunk1", "chunk2", ...],
            "passes_threshold": True
        }
    """
    if not job_description.strip():
        return {"score": 0.0, "top_matches": [], "passes_threshold": False}

    # Combine title + description for better matching
    query = f"{job_title} {job_description}".strip()[:2000]  # Limit input length

    index, chunks = load_resume_faiss_index()

    # Embed query
    query_vec = embed_text(query)
    query_vec = query_vec.reshape(1, -1)

    # Search FAISS
    distances, indices = index.search(query_vec, min(top_k, len(chunks)))

    scores = distances[0].tolist()
    top_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]

    # Average top-3 scores for final score
    avg_score = float(np.mean(scores[:3])) if scores else 0.0
    avg_score = max(0.0, min(1.0, avg_score))  # Clamp to [0, 1]

    return {
        "score": round(avg_score, 4),
        "top_matches": top_chunks,
        "passes_threshold": avg_score >= SIMILARITY_THRESHOLD,
    }


def filter_jobs_by_similarity(jobs: list[dict],
                               threshold: float = SIMILARITY_THRESHOLD) -> list[dict]:
    """
    Batch score all jobs and return only those above threshold.
    This replaces LLM calls for 90% of filtering — saves enormous CPU.
    """
    print(f"\n🔍 Scoring {len(jobs)} jobs against resume...")
    print(f"   Threshold: {threshold:.0%} — only matches above this go to LLM")

    # Batch embed all job descriptions at once
    descriptions = [
        f"{j.get('title', '')} {j.get('description', '')}".strip()[:2000]
        for j in jobs
    ]

    # Pre-warm FAISS index
    index, chunks = load_resume_faiss_index()

    # Embed all at once (much faster than one-by-one)
    query_embeddings = embed_batch(descriptions)

    # Batch FAISS search
    k = min(5, len(chunks))
    all_distances, all_indices = index.search(query_embeddings, k)

    passed = []
    for i, job in enumerate(jobs):
        scores = all_distances[i].tolist()
        avg_score = float(np.mean(scores[:3])) if scores else 0.0
        avg_score = max(0.0, min(1.0, avg_score))

        job["similarity_score"] = round(avg_score, 4)
        job["top_resume_matches"] = [
            chunks[idx] for idx in all_indices[i] if idx < len(chunks)
        ]

        if avg_score >= threshold:
            passed.append(job)
            print(f"  ✅ [{avg_score:.0%}] {job.get('title', 'N/A')} @ {job.get('company', 'N/A')}")
        else:
            print(f"  ❌ [{avg_score:.0%}] {job.get('title', 'N/A')} @ {job.get('company', 'N/A')}")

    print(f"\n📊 Result: {len(passed)}/{len(jobs)} jobs passed threshold ({threshold:.0%})")
    return passed


if __name__ == "__main__":
    # Test the similarity engine
    print("🧪 Testing Similarity Engine")
    print("=" * 50)

    # Build index from resume
    build_resume_faiss_index()

    # Test with a sample job
    sample_job_desc = """
    We are looking for an AI Engineer to build LLM-powered applications.
    Requirements: Python, LangChain, RAG, FastAPI, vector databases.
    Experience with Ollama or similar local LLM frameworks is a plus.
    You will design and deploy retrieval-augmented generation systems.
    """

    result = score_job_against_resume(
        job_description=sample_job_desc,
        job_title="AI Engineer"
    )

    print(f"\nScore: {result['score']:.0%}")
    print(f"Passes threshold: {result['passes_threshold']}")
    print(f"Top resume matches:")
    for match in result["top_matches"]:
        print(f"  • {match}")
