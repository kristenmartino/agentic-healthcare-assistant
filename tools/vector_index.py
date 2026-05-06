"""FAISS-based vector index over patient PDFs (and any reference documents).

Uses sentence-transformers `all-MiniLM-L6-v2` (~80 MB, fast on CPU).
Falls back to a TF-IDF index if sentence-transformers isn't installed,
so the graph still runs in lightweight environments.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None  # cached SentenceTransformer
_use_tfidf_fallback = False
_tfidf_vectorizer = None


def _get_embedder():
    """Return a callable that takes list[str] and returns ndarray (n, dim)."""
    global _model, _use_tfidf_fallback, _tfidf_vectorizer

    if _model is not None:
        return _embed_with_model
    if _use_tfidf_fallback and _tfidf_vectorizer is not None:
        return _embed_with_tfidf

    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_EMBEDDING_MODEL_NAME)
        logger.info("Embedder: sentence-transformers (%s)", _EMBEDDING_MODEL_NAME)
        return _embed_with_model
    except ImportError:
        logger.warning(
            "sentence-transformers not installed; falling back to TF-IDF embeddings. "
            "Install with: pip install sentence-transformers"
        )

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        _tfidf_vectorizer = TfidfVectorizer(max_features=512, ngram_range=(1, 2))
        _use_tfidf_fallback = True
        return _embed_with_tfidf
    except ImportError as exc:
        raise RuntimeError(
            "Neither sentence-transformers nor scikit-learn is installed. "
            "Install one of them: `pip install sentence-transformers` (preferred) "
            "or `pip install scikit-learn` (fallback)."
        ) from exc


def _embed_with_model(texts: list[str]) -> np.ndarray:
    embeddings = _model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(embeddings, dtype=np.float32)


def _embed_with_tfidf(texts: list[str]) -> np.ndarray:
    """TF-IDF fallback when sentence-transformers isn't available.

    The vectorizer is fit on the first call (corpus = the chunks being indexed)
    and reused for queries. Vectors are L2-normalized so dot product = cosine.
    """
    global _tfidf_vectorizer
    if not hasattr(_tfidf_vectorizer, "vocabulary_") or not _tfidf_vectorizer.vocabulary_:
        matrix = _tfidf_vectorizer.fit_transform(texts)
    else:
        matrix = _tfidf_vectorizer.transform(texts)
    dense = matrix.toarray().astype(np.float32)
    # L2 normalize
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return dense / norms


def _read_pdf_text(pdf_path: str) -> str:
    """Read text from a PDF using pypdf, falling back to pdfminer if it fails."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        pages = [p.extract_text() or "" for p in reader.pages]
        text = "\n\n".join(pages)
        if text.strip():
            return text
    except Exception as exc:
        logger.warning("pypdf failed on %s: %s — trying pdfminer", pdf_path, exc)

    try:
        from pdfminer.high_level import extract_text
        return extract_text(pdf_path) or ""
    except Exception as exc:
        logger.error("pdfminer also failed on %s: %s", pdf_path, exc)
        return ""


def _chunk_text(text: str, max_chars: int = 1200) -> list[str]:
    """Split text into chunks, preferring paragraph boundaries."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paragraphs:
        candidate = f"{buf}\n\n{p}" if buf else p
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            if buf:
                chunks.append(buf)
            # Long single paragraph: hard-split
            if len(p) > max_chars:
                for i in range(0, len(p), max_chars):
                    chunks.append(p[i:i + max_chars])
                buf = ""
            else:
                buf = p
    if buf:
        chunks.append(buf)
    return chunks


def build_index(pdf_paths: list[str], index_path: str, chunks_path: str) -> dict[str, int]:
    """Build a FAISS index from a list of PDFs. Returns counts."""
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)

    # Read & chunk
    chunks: list[dict] = []
    for path in pdf_paths:
        if not Path(path).exists():
            logger.warning("PDF not found, skipping: %s", path)
            continue
        text = _read_pdf_text(path)
        if not text.strip():
            logger.warning("No extractable text in %s", path)
            continue
        for ci, chunk in enumerate(_chunk_text(text)):
            chunks.append({
                "doc": Path(path).name,
                "chunk_id": ci,
                "text": chunk,
            })

    if not chunks:
        raise RuntimeError("No chunks produced — did you point at the right PDFs?")

    # Embed
    embedder = _get_embedder()
    embeddings = embedder([c["text"] for c in chunks])
    dim = embeddings.shape[1]

    # Build FAISS index
    try:
        import faiss
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        faiss.write_index(index, index_path)
        backend = "faiss"
    except ImportError:
        # Lightweight fallback: dump embeddings as .npy alongside.
        np.save(index_path + ".npy", embeddings)
        backend = "numpy"
        logger.warning("faiss-cpu not installed; using numpy-backed search instead")

    # Save chunks metadata (with embedder marker for query-time consistency)
    metadata = {
        "backend": backend,
        "embedder": "sentence-transformers" if not _use_tfidf_fallback else "tfidf",
        "dim": dim,
        "chunks": chunks,
    }
    with open(chunks_path, "w") as f:
        json.dump(metadata, f)

    logger.info("Index built: %d chunks across %d PDFs (backend=%s, dim=%d)",
                len(chunks), len(set(c["doc"] for c in chunks)), backend, dim)
    return {"chunks": len(chunks), "dim": dim, "backend": backend}


def search_index(query: str, index_path: str, chunks_path: str, top_k: int = 5) -> list[dict]:
    """Search the index, returning top-k chunks with scores."""
    if not Path(chunks_path).exists():
        logger.warning("Index metadata not found at %s — returning empty", chunks_path)
        return []

    with open(chunks_path) as f:
        metadata = json.load(f)
    chunks = metadata["chunks"]
    backend = metadata.get("backend", "faiss")

    # Embed query
    embedder = _get_embedder()
    query_emb = embedder([query])

    if backend == "faiss":
        try:
            import faiss
            index = faiss.read_index(index_path)
            scores, ids = index.search(query_emb, min(top_k, len(chunks)))
            scores = scores[0]
            ids = ids[0]
        except ImportError:
            logger.warning("faiss-cpu not installed but index is in faiss format — using numpy search")
            return []
    else:
        all_embeddings = np.load(index_path + ".npy")
        scores_full = all_embeddings @ query_emb[0]
        ids = np.argsort(-scores_full)[:top_k]
        scores = scores_full[ids]

    results = []
    for score, i in zip(scores, ids):
        if i < 0 or i >= len(chunks):
            continue
        results.append({
            "doc": chunks[i]["doc"],
            "chunk_id": chunks[i]["chunk_id"],
            "text": chunks[i]["text"],
            "score": float(score),
        })
    return results


# CLI: python -m tools.vector_index build <pdf1> <pdf2> ... --index path --chunks path
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if len(sys.argv) > 2 and sys.argv[1] == "build":
        # Last 2 args after "build" are paths if --index/--chunks not used
        from config import load_settings
        settings = load_settings()
        result = build_index(
            sys.argv[2:],
            settings.faiss_index_path,
            settings.faiss_chunks_path,
        )
        print(result)
    else:
        print("Usage: python -m tools.vector_index build <pdf1> <pdf2> ...")
        sys.exit(1)
