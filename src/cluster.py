"""
Topic Clustering: Discover themes across all digested content.

Uses embeddings and k-means clustering to group semantically
related segments across videos into topics.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter
from typing import Optional
import numpy as np


@dataclass
class ClusterSegment:
    """A segment belonging to a cluster."""
    video_id: str
    video_title: str
    timestamp: float
    text: str
    similarity: float  # Similarity to cluster centroid


@dataclass
class TopicCluster:
    """A cluster of semantically related segments."""
    id: int
    label: str  # Auto-generated from top terms
    size: int
    top_terms: list[str]
    segments: list[ClusterSegment] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'label': self.label,
            'size': self.size,
            'top_terms': self.top_terms,
            'segments': [
                {
                    'video_id': s.video_id,
                    'video_title': s.video_title,
                    'timestamp': s.timestamp,
                    'text': s.text,
                    'similarity': s.similarity
                }
                for s in self.segments[:10]  # Limit stored segments
            ]
        }


def cluster_content(
    output_dir: Path,
    n_clusters: int = 10,
    min_cluster_size: int = 5
) -> list[TopicCluster]:
    """
    Cluster all content into topics.

    Args:
        output_dir: Base output directory
        n_clusters: Number of clusters to create
        min_cluster_size: Minimum segments per cluster

    Returns:
        List of TopicCluster objects
    """
    try:
        from sklearn.cluster import KMeans
        from .embeddings import load_all_embeddings, cosine_similarity
    except ImportError as e:
        print(f"  Error: Missing dependency - {e}")
        print("  Run: pip install scikit-learn sentence-transformers")
        return []

    # Load all embeddings
    print("Loading embeddings from all videos...")
    indices = load_all_embeddings(output_dir)

    if not indices:
        print("No embeddings found. Process some videos first.")
        return []

    # Collect all segments with metadata
    all_embeddings = []
    all_metadata = []  # (video_id, title, timestamp, text)

    for video_id, index in indices.items():
        # Get video title
        manifest_path = output_dir / video_id / 'manifest.json'
        title = video_id
        if manifest_path.exists():
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
                title = manifest.get('metadata', {}).get('title', video_id)

        for i, embedding in enumerate(index.embeddings):
            all_embeddings.append(embedding)
            all_metadata.append((
                video_id,
                title,
                index.segment_timestamps[i],
                index.segment_texts[i]
            ))

    if len(all_embeddings) < n_clusters:
        print(f"Not enough segments ({len(all_embeddings)}) for {n_clusters} clusters")
        n_clusters = max(2, len(all_embeddings) // 5)

    embeddings_matrix = np.array(all_embeddings)
    print(f"Clustering {len(all_embeddings)} segments into {n_clusters} topics...")

    # Run k-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings_matrix)
    centroids = kmeans.cluster_centers_

    # Build clusters
    clusters = []

    for cluster_id in range(n_clusters):
        # Get segments in this cluster
        cluster_indices = np.where(labels == cluster_id)[0]

        if len(cluster_indices) < min_cluster_size:
            continue

        # Calculate similarities to centroid
        centroid = centroids[cluster_id]
        segments = []

        for idx in cluster_indices:
            meta = all_metadata[idx]
            embedding = embeddings_matrix[idx]
            sim = cosine_similarity(embedding, centroid)

            segments.append(ClusterSegment(
                video_id=meta[0],
                video_title=meta[1],
                timestamp=meta[2],
                text=meta[3],
                similarity=sim
            ))

        # Sort by similarity (most representative first)
        segments.sort(key=lambda s: s.similarity, reverse=True)

        # Generate label from top terms
        top_terms = _extract_top_terms(segments[:20])
        label = " / ".join(top_terms[:3]) if top_terms else f"Topic {cluster_id}"

        clusters.append(TopicCluster(
            id=cluster_id,
            label=label,
            size=len(segments),
            top_terms=top_terms,
            segments=segments
        ))

    # Sort by size
    clusters.sort(key=lambda c: c.size, reverse=True)

    # Save clusters
    _save_clusters(clusters, output_dir)

    return clusters


def _extract_top_terms(segments: list[ClusterSegment], n_terms: int = 5) -> list[str]:
    """Extract most frequent meaningful terms from segments."""
    # Stopwords to filter
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
        'that', 'this', 'these', 'those', 'it', 'its', "it's", 'i', 'you',
        'he', 'she', 'we', 'they', 'them', 'their', 'his', 'her', 'my',
        'your', 'our', 'what', 'which', 'who', 'whom', 'when', 'where',
        'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
        'other', 'some', 'such', 'no', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there',
        'then', 'once', 'if', 'because', 'about', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off',
        'over', 'under', 'again', 'further', 'while', 'where', "i'm", "you're",
        "we're", "they're", "i've", "you've", "we've", "don't", "doesn't",
        "didn't", "won't", "wouldn't", "couldn't", "shouldn't", "can't",
        "like", "know", "think", "going", "want", "say", "said", "get",
        "got", "make", "made", "see", "look", "come", "came", "way",
        "thing", "things", "something", "anything", "nothing", "everything",
        "really", "actually", "basically", "literally", "okay", "right",
        "well", "yeah", "yes", "no", "oh", "um", "uh", "kind", "sort"
    }

    # Count words
    word_counts = Counter()

    for segment in segments:
        words = segment.text.lower().split()
        for word in words:
            # Clean word
            word = ''.join(c for c in word if c.isalnum())
            if len(word) > 2 and word not in stopwords:
                word_counts[word] += 1

    # Return top terms
    return [word for word, _ in word_counts.most_common(n_terms)]


def _save_clusters(clusters: list[TopicCluster], output_dir: Path) -> Path:
    """Save clusters to JSON."""
    clusters_path = output_dir / 'topic_clusters.json'

    data = {
        'n_clusters': len(clusters),
        'total_segments': sum(c.size for c in clusters),
        'clusters': [c.to_dict() for c in clusters]
    }

    with open(clusters_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return clusters_path


def load_clusters(output_dir: Path) -> Optional[list[TopicCluster]]:
    """Load saved clusters."""
    clusters_path = output_dir / 'topic_clusters.json'

    if not clusters_path.exists():
        return None

    with open(clusters_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    clusters = []
    for c in data.get('clusters', []):
        cluster = TopicCluster(
            id=c['id'],
            label=c['label'],
            size=c['size'],
            top_terms=c['top_terms']
        )
        cluster.segments = [
            ClusterSegment(
                video_id=s['video_id'],
                video_title=s['video_title'],
                timestamp=s['timestamp'],
                text=s['text'],
                similarity=s['similarity']
            )
            for s in c.get('segments', [])
        ]
        clusters.append(cluster)

    return clusters


def find_cluster_for_query(
    query: str,
    output_dir: Path
) -> Optional[TopicCluster]:
    """
    Find the most relevant cluster for a query.

    Args:
        query: Search query
        output_dir: Base output directory

    Returns:
        Most relevant TopicCluster, or None
    """
    try:
        from .embeddings import embed_text, cosine_similarity
    except ImportError:
        return None

    clusters = load_clusters(output_dir)
    if not clusters:
        return None

    # Embed query
    query_embedding = embed_text(query)
    if query_embedding is None:
        return None

    # Load cluster centroids (need to recompute from segments)
    # For now, just match against cluster labels/terms
    query_lower = query.lower()

    best_cluster = None
    best_score = 0

    for cluster in clusters:
        score = 0
        # Check label
        if query_lower in cluster.label.lower():
            score += 10
        # Check top terms
        for term in cluster.top_terms:
            if query_lower in term or term in query_lower:
                score += 5

        if score > best_score:
            best_score = score
            best_cluster = cluster

    return best_cluster


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Topic clustering across all digested content"
    )
    parser.add_argument(
        '--n-clusters', '-n',
        type=int,
        default=10,
        help="Number of topic clusters (default: 10)"
    )
    parser.add_argument(
        '--query', '-q',
        type=str,
        help="Find cluster matching a query"
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path(__file__).parent.parent / 'output',
        help="Output directory"
    )

    args = parser.parse_args()
    output_dir = args.output

    if args.query:
        print(f"Finding cluster for: '{args.query}'")
        cluster = find_cluster_for_query(args.query, output_dir)

        if cluster:
            print(f"\nBest match: {cluster.label}")
            print(f"Size: {cluster.size} segments")
            print(f"Top terms: {', '.join(cluster.top_terms)}")
            print(f"\nSample content:")
            for seg in cluster.segments[:5]:
                ts = format_timestamp(seg.timestamp)
                print(f"  [{seg.video_title[:30]}] [{ts}]")
                print(f"    {seg.text[:80]}...")
        else:
            print("No matching cluster found. Run clustering first.")
    else:
        print("Clustering all content...")
        clusters = cluster_content(output_dir, n_clusters=args.n_clusters)

        if clusters:
            print(f"\nDiscovered {len(clusters)} topics:\n")
            for cluster in clusters:
                print(f"[{cluster.id}] {cluster.label}")
                print(f"    {cluster.size} segments | Terms: {', '.join(cluster.top_terms[:5])}")

                # Show sample
                if cluster.segments:
                    seg = cluster.segments[0]
                    print(f"    Sample: \"{seg.text[:60]}...\"")
                print()

            print(f"Saved to: {output_dir / 'topic_clusters.json'}")
        else:
            print("Clustering failed. Check that videos have been processed.")
