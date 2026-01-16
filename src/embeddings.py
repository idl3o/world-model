"""
Embedding Engine: Vector representations for semantic operations.

Provides text embeddings using sentence-transformers for:
- Semantic search (find similar content by meaning)
- Topic clustering (group related segments)
- Cross-content analysis
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np


@dataclass
class EmbeddingIndex:
    """Embedding index for a single video's transcript."""
    video_id: str
    model_name: str
    embeddings: np.ndarray  # Shape: (n_segments, embedding_dim)
    segment_texts: list[str]  # Original text for each embedding
    segment_timestamps: list[float]  # Start time for each segment

    def save(self, work_dir: Path) -> Path:
        """Save embedding index to disk."""
        # Save embeddings as compressed numpy
        embeddings_path = work_dir / 'embeddings.npz'
        np.savez_compressed(
            embeddings_path,
            embeddings=self.embeddings
        )

        # Save metadata as JSON
        meta_path = work_dir / 'embeddings_meta.json'
        meta = {
            'video_id': self.video_id,
            'model_name': self.model_name,
            'n_segments': len(self.segment_texts),
            'embedding_dim': self.embeddings.shape[1] if len(self.embeddings) > 0 else 0,
            'segment_texts': self.segment_texts,
            'segment_timestamps': self.segment_timestamps
        }
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)

        return embeddings_path

    @classmethod
    def load(cls, work_dir: Path) -> Optional['EmbeddingIndex']:
        """Load embedding index from disk."""
        embeddings_path = work_dir / 'embeddings.npz'
        meta_path = work_dir / 'embeddings_meta.json'

        if not embeddings_path.exists() or not meta_path.exists():
            return None

        # Load embeddings
        data = np.load(embeddings_path)
        embeddings = data['embeddings']

        # Load metadata
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        return cls(
            video_id=meta['video_id'],
            model_name=meta['model_name'],
            embeddings=embeddings,
            segment_texts=meta['segment_texts'],
            segment_timestamps=meta['segment_timestamps']
        )


# =============================================================================
# Model Cache
# =============================================================================

_embedding_model = None
_embedding_model_name = None


def get_embedding_model(model_name: str = 'all-MiniLM-L6-v2'):
    """
    Get cached sentence-transformer model.

    Args:
        model_name: Model to use. Options:
            - 'all-MiniLM-L6-v2' (default, fast, 384 dim)
            - 'all-mpnet-base-v2' (better quality, 768 dim)
            - 'multi-qa-MiniLM-L6-cos-v1' (optimized for search)

    Returns:
        SentenceTransformer model, or None if unavailable
    """
    global _embedding_model, _embedding_model_name

    if _embedding_model is not None and _embedding_model_name == model_name:
        return _embedding_model

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  Warning: sentence-transformers not installed.")
        print("  Run: pip install sentence-transformers")
        return None

    print(f"  Loading embedding model '{model_name}'...")
    _embedding_model = SentenceTransformer(model_name)
    _embedding_model_name = model_name
    print(f"  Embedding model loaded.")

    return _embedding_model


def unload_embedding_model():
    """Unload cached model to free memory."""
    global _embedding_model, _embedding_model_name
    _embedding_model = None
    _embedding_model_name = None


# =============================================================================
# Embedding Functions
# =============================================================================

def embed_text(text: str, model_name: str = 'all-MiniLM-L6-v2') -> Optional[np.ndarray]:
    """
    Embed a single text string.

    Args:
        text: Text to embed
        model_name: Model to use

    Returns:
        Embedding vector, or None if failed
    """
    model = get_embedding_model(model_name)
    if model is None:
        return None

    embedding = model.encode(text, convert_to_numpy=True)
    return embedding


def embed_batch(
    texts: list[str],
    model_name: str = 'all-MiniLM-L6-v2',
    batch_size: int = 32,
    show_progress: bool = True
) -> Optional[np.ndarray]:
    """
    Embed multiple texts efficiently.

    Args:
        texts: List of texts to embed
        model_name: Model to use
        batch_size: Batch size for processing
        show_progress: Whether to show progress bar

    Returns:
        Array of embeddings (n_texts, embedding_dim), or None if failed
    """
    model = get_embedding_model(model_name)
    if model is None:
        return None

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        batch_size=batch_size,
        show_progress_bar=show_progress
    )
    return embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def cosine_similarity_batch(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between query and all corpus vectors.

    Args:
        query: Single query vector (embedding_dim,)
        corpus: Matrix of vectors (n_vectors, embedding_dim)

    Returns:
        Array of similarities (n_vectors,)
    """
    # Normalize
    query_norm = query / np.linalg.norm(query)
    corpus_norms = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)

    # Dot product gives cosine similarity for normalized vectors
    similarities = corpus_norms @ query_norm

    return similarities


# =============================================================================
# Transcript Embedding
# =============================================================================

def embed_transcript(
    transcript: list,
    video_id: str,
    work_dir: Path,
    model_name: str = 'all-MiniLM-L6-v2',
    force_rebuild: bool = False
) -> Optional[EmbeddingIndex]:
    """
    Embed all segments in a transcript.

    Caches embeddings to disk for fast retrieval.

    Args:
        transcript: List of TranscriptSegment objects
        video_id: Video identifier
        work_dir: Working directory for this video
        model_name: Embedding model to use
        force_rebuild: If True, rebuild even if cache exists

    Returns:
        EmbeddingIndex, or None if failed
    """
    # Check cache
    if not force_rebuild:
        cached = EmbeddingIndex.load(work_dir)
        if cached is not None and cached.model_name == model_name:
            return cached

    if not transcript:
        return None

    # Extract texts and timestamps
    texts = [seg.text for seg in transcript]
    timestamps = [seg.start for seg in transcript]

    # Embed
    print(f"  Embedding {len(texts)} segments...")
    embeddings = embed_batch(texts, model_name=model_name, show_progress=True)

    if embeddings is None:
        return None

    # Create index
    index = EmbeddingIndex(
        video_id=video_id,
        model_name=model_name,
        embeddings=embeddings,
        segment_texts=texts,
        segment_timestamps=timestamps
    )

    # Save to cache
    index.save(work_dir)
    print(f"  Saved embeddings to {work_dir}")

    return index


def load_or_build_embeddings(
    video_id: str,
    output_dir: Path,
    model_name: str = 'all-MiniLM-L6-v2'
) -> Optional[EmbeddingIndex]:
    """
    Load embeddings from cache or build if needed.

    Args:
        video_id: Video identifier
        output_dir: Base output directory
        model_name: Embedding model

    Returns:
        EmbeddingIndex, or None if video not found
    """
    work_dir = output_dir / video_id

    # Try loading from cache
    cached = EmbeddingIndex.load(work_dir)
    if cached is not None and cached.model_name == model_name:
        return cached

    # Need to build - load transcript
    transcript_path = work_dir / 'transcript.json'
    if not transcript_path.exists():
        return None

    from .extract import load_transcript
    transcript = load_transcript(work_dir)

    if not transcript:
        return None

    return embed_transcript(transcript, video_id, work_dir, model_name)


def load_all_embeddings(
    output_dir: Path,
    model_name: str = 'all-MiniLM-L6-v2'
) -> dict[str, EmbeddingIndex]:
    """
    Load embeddings for all videos, building if needed.

    Args:
        output_dir: Base output directory
        model_name: Embedding model

    Returns:
        Dict mapping video_id to EmbeddingIndex
    """
    indices = {}

    for video_dir in output_dir.iterdir():
        if not video_dir.is_dir():
            continue

        video_id = video_dir.name
        index = load_or_build_embeddings(video_id, output_dir, model_name)

        if index is not None:
            indices[video_id] = index

    return indices


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.embeddings <video_id>")
        print("       python -m src.embeddings --all")
        sys.exit(1)

    output_dir = Path(__file__).parent.parent / 'output'

    if sys.argv[1] == '--all':
        print("Building embeddings for all videos...")
        indices = load_all_embeddings(output_dir)
        print(f"\nEmbedded {len(indices)} videos:")
        for vid, idx in indices.items():
            print(f"  {vid}: {len(idx.segment_texts)} segments")
    else:
        video_id = sys.argv[1]
        print(f"Building embeddings for {video_id}...")
        index = load_or_build_embeddings(video_id, output_dir)

        if index:
            print(f"\nEmbedded {len(index.segment_texts)} segments")
            print(f"Embedding dim: {index.embeddings.shape[1]}")
        else:
            print(f"Video not found: {video_id}")
