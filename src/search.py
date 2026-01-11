"""
Cross-Content Search: Query across all digested media.

Provides unified search across all processed videos,
enabling semantic retrieval from the world model's memory.
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class SearchResult:
    """A single search result."""
    video_id: str
    video_title: str
    timestamp: float
    text: str
    score: float
    context_before: str = ""
    context_after: str = ""


def search_all(
    query: str,
    output_dir: Path,
    max_results: int = 20,
    context_window: int = 1
) -> list[SearchResult]:
    """
    Search across all digested content.

    Args:
        query: Search query (supports simple text matching)
        output_dir: Base output directory containing all videos
        max_results: Maximum results to return
        context_window: Number of segments before/after to include

    Returns:
        List of SearchResult objects, sorted by relevance
    """
    results = []

    # Find all video directories
    video_dirs = [d for d in output_dir.iterdir() if d.is_dir()]

    for video_dir in video_dirs:
        video_id = video_dir.name
        transcript_path = video_dir / 'transcript.json'
        manifest_path = video_dir / 'manifest.json'

        if not transcript_path.exists():
            continue

        # Load metadata
        title = video_id
        if manifest_path.exists():
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
                title = manifest.get('metadata', {}).get('title', video_id)

        # Load and search transcript
        with open(transcript_path, 'r', encoding='utf-8') as f:
            segments = json.load(f)

        video_results = _search_transcript(
            query, segments, video_id, title, context_window
        )
        results.extend(video_results)

    # Sort by score (higher is better)
    results.sort(key=lambda r: r.score, reverse=True)

    return results[:max_results]


def _search_transcript(
    query: str,
    segments: list,
    video_id: str,
    title: str,
    context_window: int
) -> list[SearchResult]:
    """Search within a single transcript."""
    results = []
    query_lower = query.lower()
    query_words = set(query_lower.split())

    for i, seg in enumerate(segments):
        text = seg.get('text', '')
        text_lower = text.lower()

        # Calculate relevance score
        score = 0

        # Exact phrase match (highest score)
        if query_lower in text_lower:
            score += 10

        # Word matches
        text_words = set(text_lower.split())
        matching_words = query_words & text_words
        score += len(matching_words) * 2

        # Partial word matches
        for qword in query_words:
            if any(qword in tword for tword in text_words):
                score += 0.5

        if score > 0:
            # Get context
            context_before = ""
            context_after = ""

            if context_window > 0:
                if i > 0:
                    context_before = segments[i-1].get('text', '')[:100]
                if i < len(segments) - 1:
                    context_after = segments[i+1].get('text', '')[:100]

            results.append(SearchResult(
                video_id=video_id,
                video_title=title,
                timestamp=seg.get('start', 0),
                text=text,
                score=score,
                context_before=context_before,
                context_after=context_after
            ))

    return results


def search_by_topic(
    topic: str,
    output_dir: Path,
    max_results: int = 10
) -> list[SearchResult]:
    """
    Search for a topic across all content.

    Uses concept graph for expansion if available, falls back to hardcoded terms.
    """
    search_terms = [topic]

    # Try to use concept graph for expansion
    try:
        from .concepts import load_global_graph, get_related_entities, normalize_entity_id

        graph = load_global_graph(output_dir)
        if graph:
            # Check if topic is in the graph
            topic_normalized = normalize_entity_id(topic)
            topic_lower = topic.lower()

            entity_id = None
            if topic_normalized in graph.entities:
                entity_id = topic_normalized
            elif topic_lower in graph.entity_index:
                entity_id = graph.entity_index[topic_lower]

            if entity_id:
                # Get related entities for expansion
                related = get_related_entities(entity_id, graph, limit=5)
                for rel_id, _ in related:
                    if rel_id in graph.entities:
                        search_terms.append(graph.entities[rel_id].canonical_name)
    except Exception:
        pass

    # Fall back to hardcoded expansions if no graph results
    if len(search_terms) == 1:
        topic_expansions = {
            'consciousness': ['consciousness', 'aware', 'mind', 'perception', 'qualia', 'experience'],
            'free will': ['free will', 'determinism', 'choice', 'agency', 'volition'],
            'meditation': ['meditation', 'mindfulness', 'breathing', 'practice', 'awareness'],
            'yoga': ['yoga', 'pose', 'breath', 'stretch', 'body', 'movement'],
            'brain': ['brain', 'neuroscience', 'neural', 'cortex', 'neurons'],
            'stress': ['stress', 'anxiety', 'cortisol', 'pressure', 'tension'],
            'python': ['python', 'programming', 'code', 'function', 'variable'],
        }
        search_terms = topic_expansions.get(topic.lower(), [topic])

    all_results = []
    for term in search_terms:
        results = search_all(term, output_dir, max_results=max_results * 2)
        all_results.extend(results)

    # Deduplicate by (video_id, timestamp)
    seen = set()
    unique_results = []
    for r in all_results:
        key = (r.video_id, int(r.timestamp))
        if key not in seen:
            seen.add(key)
            unique_results.append(r)

    # Re-sort by score
    unique_results.sort(key=lambda r: r.score, reverse=True)

    return unique_results[:max_results]


def search_by_entity(
    entity_name: str,
    output_dir: Path,
    max_results: int = 20
) -> list[SearchResult]:
    """
    Search for a specific entity across all content using the concept graph.

    Args:
        entity_name: Name of entity to search for
        output_dir: Base output directory
        max_results: Maximum results to return

    Returns:
        List of SearchResult objects with entity mentions
    """
    try:
        from .concepts import (
            load_global_graph, load_video_concepts,
            normalize_entity_id, search_entities
        )
    except ImportError:
        # Fall back to regular search if concepts module not available
        return search_all(entity_name, output_dir, max_results)

    graph = load_global_graph(output_dir)
    if not graph:
        return search_all(entity_name, output_dir, max_results)

    # Find entity in graph
    matching_entities = search_entities(entity_name, graph, limit=1)
    if not matching_entities:
        return search_all(entity_name, output_dir, max_results)

    entity = matching_entities[0]
    results = []

    # Get mentions from each video
    for video_id, video_info in entity.videos.items():
        video_dir = output_dir / video_id
        video_concepts = load_video_concepts(video_dir)

        if not video_concepts or entity.id not in video_concepts.entities:
            continue

        video_entity = video_concepts.entities[entity.id]

        # Add each mention as a result
        for mention in video_entity.mentions:
            results.append(SearchResult(
                video_id=video_id,
                video_title=video_info.get('title', video_id),
                timestamp=mention.start,
                text=mention.context,
                score=15.0,  # Entity match scores higher than keyword
                context_before="",
                context_after=""
            ))

    # Sort by score (all same) then by timestamp
    results.sort(key=lambda r: (r.video_id, r.timestamp))

    return results[:max_results]


def get_video_summary(video_id: str, output_dir: Path) -> Optional[dict]:
    """Get summary info for a single video."""
    video_dir = output_dir / video_id
    manifest_path = video_dir / 'manifest.json'

    if not manifest_path.exists():
        return None

    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    return {
        'video_id': video_id,
        'title': manifest.get('metadata', {}).get('title', video_id),
        'channel': manifest.get('metadata', {}).get('channel', 'Unknown'),
        'duration': manifest.get('metadata', {}).get('duration', 0),
        'word_count': manifest.get('stats', {}).get('word_count', 0),
    }


def list_all_videos(output_dir: Path) -> list[dict]:
    """List all digested videos."""
    videos = []
    for video_dir in output_dir.iterdir():
        if video_dir.is_dir():
            summary = get_video_summary(video_dir.name, output_dir)
            if summary:
                videos.append(summary)

    # Sort by duration (longest first)
    videos.sort(key=lambda v: v['duration'], reverse=True)
    return videos


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def format_result(result: SearchResult) -> str:
    """Format a search result for display."""
    ts = format_timestamp(result.timestamp)
    return f"[{result.video_title[:30]}] [{ts}] {result.text[:80]}..."


if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Search across digested content")
    parser.add_argument('query', nargs='*', help="Search query")
    parser.add_argument('--entity', '-e', action='store_true',
                        help="Search by entity using concept graph")
    parser.add_argument('--topic', '-t', action='store_true',
                        help="Search by topic with expansion")
    parser.add_argument('--output', '-o', type=Path,
                        default=Path(__file__).parent.parent / 'output',
                        help="Output directory")

    args = parser.parse_args()
    output_dir = args.output

    if not args.query:
        print("Usage: python -m src.search <query>")
        print("       python -m src.search --entity <entity_name>")
        print("       python -m src.search --topic <topic>")
        print("\nAvailable videos:")
        for v in list_all_videos(output_dir):
            mins = int(v['duration'] // 60)
            print(f"  [{v['video_id']}] {v['title'][:50]} ({mins} min)")
        sys.exit(0)

    query = ' '.join(args.query)
    print(f"Searching for: '{query}'")

    if args.entity:
        print("(Entity search using concept graph)")
        results = search_by_entity(query, output_dir)
    elif args.topic:
        print("(Topic search with expansion)")
        results = search_by_topic(query, output_dir)
    else:
        results = search_all(query, output_dir)

    print("=" * 60)

    if not results:
        print("No results found.")
    else:
        print(f"Found {len(results)} results:\n")
        for r in results[:10]:
            ts = format_timestamp(r.timestamp)
            print(f"[{r.video_title[:35]}]")
            print(f"  Time: {ts} | Score: {r.score:.1f}")
            print(f"  {r.text[:120]}...")
            print()
