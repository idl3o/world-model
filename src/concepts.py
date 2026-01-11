"""
Concept Graph: Entity Extraction and Knowledge Graph Management

Extracts named entities from transcripts using SpaCy NER, tracks co-occurrences,
and builds a cross-video knowledge index for semantic search.

Usage:
    python -m src.concepts <video_id>           # Extract entities for one video
    python -m src.concepts --rebuild            # Rebuild global graph from all videos
    python -m src.concepts --query "Sam Harris" # Query entity across all content
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from .sync import FileSyncEngine


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EntityMention:
    """A single mention of an entity in transcript."""
    start: float              # Timestamp in seconds
    end: float
    text: str                 # Surface form as spoken
    context: str              # Surrounding text for display
    segment_index: int        # Reference to original transcript segment


@dataclass
class ExtractedEntity:
    """An entity extracted from a single video."""
    id: str                   # Normalized ID (lowercase, underscores)
    canonical_name: str       # Display name (most common form)
    entity_type: str          # PERSON, ORG, GPE, WORK_OF_ART, CONCEPT, etc.
    aliases: set[str] = field(default_factory=set)  # All surface forms seen
    mentions: list[EntityMention] = field(default_factory=list)

    @property
    def mention_count(self) -> int:
        return len(self.mentions)

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'canonical_name': self.canonical_name,
            'entity_type': self.entity_type,
            'aliases': list(self.aliases),
            'mentions': [
                {
                    'start': m.start,
                    'end': m.end,
                    'text': m.text,
                    'context': m.context,
                    'segment_index': m.segment_index
                }
                for m in self.mentions
            ],
            'mention_count': self.mention_count
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ExtractedEntity':
        entity = cls(
            id=data['id'],
            canonical_name=data['canonical_name'],
            entity_type=data['entity_type'],
            aliases=set(data.get('aliases', [])),
        )
        entity.mentions = [
            EntityMention(
                start=m['start'],
                end=m['end'],
                text=m['text'],
                context=m['context'],
                segment_index=m['segment_index']
            )
            for m in data.get('mentions', [])
        ]
        return entity


@dataclass
class Cooccurrence:
    """Two entities appearing near each other in time."""
    entity_a_id: str
    entity_b_id: str
    count: int
    timestamps: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'entity_a': self.entity_a_id,
            'entity_b': self.entity_b_id,
            'count': self.count,
            'timestamps': self.timestamps
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Cooccurrence':
        return cls(
            entity_a_id=data['entity_a'],
            entity_b_id=data['entity_b'],
            count=data['count'],
            timestamps=data.get('timestamps', [])
        )


@dataclass
class VideoConceptData:
    """All concept data for a single video."""
    video_id: str
    extraction_model: str = "en_core_web_md"
    entities: dict[str, ExtractedEntity] = field(default_factory=dict)  # id -> entity
    cooccurrences: list[Cooccurrence] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'version': '1.0',
            'video_id': self.video_id,
            'extraction_model': self.extraction_model,
            'entities': {k: v.to_dict() for k, v in self.entities.items()},
            'cooccurrences': [c.to_dict() for c in self.cooccurrences]
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'VideoConceptData':
        concept_data = cls(
            video_id=data['video_id'],
            extraction_model=data.get('extraction_model', 'unknown')
        )
        concept_data.entities = {
            k: ExtractedEntity.from_dict(v)
            for k, v in data.get('entities', {}).items()
        }
        concept_data.cooccurrences = [
            Cooccurrence.from_dict(c)
            for c in data.get('cooccurrences', [])
        ]
        return concept_data


@dataclass
class GlobalEntity:
    """Entity data aggregated across all videos."""
    id: str
    canonical_name: str
    entity_type: str
    videos: dict[str, dict] = field(default_factory=dict)  # video_id -> {mention_count, title}
    related_entities: dict[str, dict] = field(default_factory=dict)  # entity_id -> {count, videos}

    @property
    def total_mentions(self) -> int:
        return sum(v.get('mention_count', 0) for v in self.videos.values())

    def to_dict(self) -> dict:
        return {
            'canonical_name': self.canonical_name,
            'entity_type': self.entity_type,
            'videos': self.videos,
            'related_entities': self.related_entities,
            'total_mentions': self.total_mentions
        }

    @classmethod
    def from_dict(cls, entity_id: str, data: dict) -> 'GlobalEntity':
        return cls(
            id=entity_id,
            canonical_name=data['canonical_name'],
            entity_type=data['entity_type'],
            videos=data.get('videos', {}),
            related_entities=data.get('related_entities', {})
        )


@dataclass
class GlobalConceptGraph:
    """Cross-video concept index."""
    version: str = "1.0"
    last_updated: datetime = field(default_factory=datetime.now)
    entities: dict[str, GlobalEntity] = field(default_factory=dict)  # id -> global entity
    entity_index: dict[str, str] = field(default_factory=dict)  # alias -> canonical id

    def to_dict(self) -> dict:
        return {
            'version': self.version,
            'last_updated': self.last_updated.isoformat(),
            'total_videos': len(set(
                vid for e in self.entities.values() for vid in e.videos.keys()
            )),
            'total_entities': len(self.entities),
            'entities': {k: v.to_dict() for k, v in self.entities.items()},
            'entity_index': self.entity_index
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'GlobalConceptGraph':
        graph = cls(
            version=data.get('version', '1.0'),
            last_updated=datetime.fromisoformat(data['last_updated']) if data.get('last_updated') else datetime.now()
        )
        graph.entities = {
            k: GlobalEntity.from_dict(k, v)
            for k, v in data.get('entities', {}).items()
        }
        graph.entity_index = data.get('entity_index', {})
        return graph


# =============================================================================
# Normalization
# =============================================================================

def normalize_entity_id(name: str) -> str:
    """
    Convert entity name to normalized ID.

    Examples:
        "Sam Harris" -> "sam_harris"
        "New York City" -> "new_york_city"
        "AI" -> "ai"
    """
    # Lowercase
    normalized = name.lower()
    # Replace non-alphanumeric with underscore
    normalized = re.sub(r'[^a-z0-9]+', '_', normalized)
    # Remove leading/trailing underscores
    normalized = normalized.strip('_')
    # Collapse multiple underscores
    normalized = re.sub(r'_+', '_', normalized)
    return normalized


def choose_canonical_name(aliases: set[str]) -> str:
    """
    Choose the best canonical name from aliases.

    Prefers: capitalized, longer, more complete forms.
    """
    if not aliases:
        return ""

    def score(name: str) -> tuple:
        # Prefer capitalized
        is_capitalized = name[0].isupper() if name else False
        # Prefer longer (but not excessively)
        length = min(len(name), 50)
        # Prefer names with spaces (full names over single words)
        has_spaces = ' ' in name
        return (is_capitalized, has_spaces, length)

    return max(aliases, key=score)


# =============================================================================
# Entity Extraction
# =============================================================================

def extract_entities(
    transcript: list,
    model_name: str = "en_core_web_md"
) -> VideoConceptData:
    """
    Extract entities from transcript using SpaCy NER.

    Args:
        transcript: List of TranscriptSegment objects
        model_name: SpaCy model to use

    Returns:
        VideoConceptData with extracted entities
    """
    try:
        import spacy
    except ImportError:
        print("  Warning: SpaCy not installed. Run: pip install spacy")
        print("  Then download model: python -m spacy download en_core_web_md")
        return VideoConceptData(video_id="", extraction_model=model_name)

    # Load model (try requested, fall back to smaller)
    nlp = None
    for model in [model_name, "en_core_web_sm"]:
        try:
            nlp = spacy.load(model)
            if model != model_name:
                print(f"  Using fallback model: {model}")
            break
        except OSError:
            continue

    if nlp is None:
        print(f"  Error: No SpaCy model available. Run: python -m spacy download {model_name}")
        return VideoConceptData(video_id="", extraction_model=model_name)

    # Entity types we care about
    RELEVANT_TYPES = {
        'PERSON',      # People, including fictional
        'ORG',         # Companies, agencies, institutions
        'GPE',         # Countries, cities, states
        'LOC',         # Non-GPE locations
        'WORK_OF_ART', # Titles of books, songs, etc.
        'EVENT',       # Named hurricanes, battles, wars, sports events
        'PRODUCT',     # Objects, vehicles, foods, etc.
        'LAW',         # Named documents made into laws
        'NORP',        # Nationalities, religious/political groups
    }

    # Accumulate entities
    entities_raw: dict[str, dict] = defaultdict(lambda: {
        'aliases': set(),
        'type': None,
        'mentions': []
    })

    video_id = ""

    # Process each segment
    for seg_idx, segment in enumerate(transcript):
        if hasattr(segment, 'video_id') and not video_id:
            video_id = segment.video_id

        text = segment.text
        start = segment.start
        end = segment.end

        # Get surrounding context
        context_start = max(0, seg_idx - 1)
        context_end = min(len(transcript), seg_idx + 2)
        context_text = ' '.join(
            transcript[i].text for i in range(context_start, context_end)
        )[:200]

        # Run NER
        doc = nlp(text)

        for ent in doc.ents:
            if ent.label_ not in RELEVANT_TYPES:
                continue

            # Skip very short entities (likely noise)
            if len(ent.text.strip()) < 2:
                continue

            entity_id = normalize_entity_id(ent.text)
            if not entity_id:
                continue

            entities_raw[entity_id]['aliases'].add(ent.text)
            entities_raw[entity_id]['type'] = ent.label_
            entities_raw[entity_id]['mentions'].append(EntityMention(
                start=start,
                end=end,
                text=ent.text,
                context=context_text,
                segment_index=seg_idx
            ))

    # Convert to ExtractedEntity objects
    entities = {}
    for entity_id, data in entities_raw.items():
        canonical = choose_canonical_name(data['aliases'])
        entities[entity_id] = ExtractedEntity(
            id=entity_id,
            canonical_name=canonical,
            entity_type=data['type'],
            aliases=data['aliases'],
            mentions=data['mentions']
        )

    return VideoConceptData(
        video_id=video_id,
        extraction_model=model_name,
        entities=entities
    )


# =============================================================================
# Co-occurrence Tracking
# =============================================================================

def compute_cooccurrences(
    entities: dict[str, ExtractedEntity],
    window_seconds: float = 30.0
) -> list[Cooccurrence]:
    """
    Compute entity co-occurrences within time windows.

    Two entities co-occur if they are mentioned within window_seconds of each other.

    Args:
        entities: Dict of entity_id -> ExtractedEntity
        window_seconds: Time window for co-occurrence (default 30s)

    Returns:
        List of Cooccurrence objects
    """
    # Build timeline of all mentions
    timeline: list[tuple[float, str]] = []  # (timestamp, entity_id)

    for entity_id, entity in entities.items():
        for mention in entity.mentions:
            timeline.append((mention.start, entity_id))

    # Sort by timestamp
    timeline.sort(key=lambda x: x[0])

    # Count co-occurrences using sliding window
    cooccur_counts: dict[tuple[str, str], list[float]] = defaultdict(list)

    for i, (time_i, entity_i) in enumerate(timeline):
        for j in range(i + 1, len(timeline)):
            time_j, entity_j = timeline[j]

            # Outside window
            if time_j - time_i > window_seconds:
                break

            # Same entity doesn't co-occur with itself
            if entity_i == entity_j:
                continue

            # Normalize pair order for consistent counting
            pair = tuple(sorted([entity_i, entity_j]))
            cooccur_counts[pair].append(time_i)

    # Convert to Cooccurrence objects
    cooccurrences = []
    for (entity_a, entity_b), timestamps in cooccur_counts.items():
        cooccurrences.append(Cooccurrence(
            entity_a_id=entity_a,
            entity_b_id=entity_b,
            count=len(timestamps),
            timestamps=sorted(set(timestamps))[:50]  # Cap stored timestamps
        ))

    # Sort by count descending
    cooccurrences.sort(key=lambda c: c.count, reverse=True)

    return cooccurrences


# =============================================================================
# Persistence (Per-Video)
# =============================================================================

def save_video_concepts(data: VideoConceptData, work_dir: Path) -> Path:
    """Save per-video concept data to JSON."""
    concepts_path = work_dir / 'concepts.json'

    with open(concepts_path, 'w', encoding='utf-8') as f:
        json.dump(data.to_dict(), f, indent=2, ensure_ascii=False)

    return concepts_path


def load_video_concepts(work_dir: Path) -> Optional[VideoConceptData]:
    """Load existing concept data for a video."""
    concepts_path = work_dir / 'concepts.json'

    if not concepts_path.exists():
        return None

    with open(concepts_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return VideoConceptData.from_dict(data)


# =============================================================================
# Global Graph Management
# =============================================================================

def update_global_graph(
    video_data: VideoConceptData,
    video_title: str,
    output_dir: Path
) -> GlobalConceptGraph:
    """
    Add/update video in global concept graph.

    Args:
        video_data: Concept data for one video
        video_title: Title of the video
        output_dir: Base output directory

    Returns:
        Updated GlobalConceptGraph
    """
    graph = load_global_graph(output_dir) or GlobalConceptGraph()

    video_id = video_data.video_id

    # Add/update entities
    for entity_id, entity in video_data.entities.items():
        if entity_id not in graph.entities:
            graph.entities[entity_id] = GlobalEntity(
                id=entity_id,
                canonical_name=entity.canonical_name,
                entity_type=entity.entity_type
            )

        global_entity = graph.entities[entity_id]
        global_entity.videos[video_id] = {
            'mention_count': entity.mention_count,
            'title': video_title
        }

        # Update entity index with all aliases
        for alias in entity.aliases:
            alias_normalized = alias.lower()
            graph.entity_index[alias_normalized] = entity_id

    # Add co-occurrence relationships
    for cooccur in video_data.cooccurrences:
        entity_a = graph.entities.get(cooccur.entity_a_id)
        entity_b = graph.entities.get(cooccur.entity_b_id)

        if entity_a and entity_b:
            # Update entity_a's related_entities
            if cooccur.entity_b_id not in entity_a.related_entities:
                entity_a.related_entities[cooccur.entity_b_id] = {'count': 0, 'videos': []}
            entity_a.related_entities[cooccur.entity_b_id]['count'] += cooccur.count
            if video_id not in entity_a.related_entities[cooccur.entity_b_id]['videos']:
                entity_a.related_entities[cooccur.entity_b_id]['videos'].append(video_id)

            # Update entity_b's related_entities (symmetric)
            if cooccur.entity_a_id not in entity_b.related_entities:
                entity_b.related_entities[cooccur.entity_a_id] = {'count': 0, 'videos': []}
            entity_b.related_entities[cooccur.entity_a_id]['count'] += cooccur.count
            if video_id not in entity_b.related_entities[cooccur.entity_a_id]['videos']:
                entity_b.related_entities[cooccur.entity_a_id]['videos'].append(video_id)

    graph.last_updated = datetime.now()
    save_global_graph(graph, output_dir)

    return graph


def update_global_graph_incremental(
    new_entities: dict[str, 'ExtractedEntity'],
    new_cooccurrences: list[tuple[str, str]],
    video_id: str,
    video_title: str,
    output_dir: Path,
    sync_engine: Optional['FileSyncEngine'] = None
) -> GlobalConceptGraph:
    """
    Incrementally update global graph with new entities from a chunk.

    Unlike update_global_graph(), this is designed for real-time streaming
    where entities are discovered incrementally. Uses file locking if
    sync_engine is provided for multi-device sync.

    Args:
        new_entities: Dict of entity_id -> ExtractedEntity for new/updated entities
        new_cooccurrences: List of (entity_a_id, entity_b_id) pairs
        video_id: Video/stream identifier
        video_title: Title for display
        output_dir: Base output directory
        sync_engine: Optional FileSyncEngine for locking

    Returns:
        Updated GlobalConceptGraph
    """
    # Load graph (with optional locking)
    if sync_engine:
        try:
            data, version = sync_engine.read_with_lock('concept_graph.json')
            graph = GlobalConceptGraph.from_dict(data) if data else GlobalConceptGraph()
        except TimeoutError:
            # Can't acquire lock, load without locking
            graph = load_global_graph(output_dir) or GlobalConceptGraph()
            version = None
    else:
        graph = load_global_graph(output_dir) or GlobalConceptGraph()
        version = None

    # Add/update entities
    for entity_id, entity in new_entities.items():
        if entity_id not in graph.entities:
            graph.entities[entity_id] = GlobalEntity(
                id=entity_id,
                canonical_name=entity.canonical_name,
                entity_type=entity.entity_type
            )

        global_entity = graph.entities[entity_id]

        # Update video reference
        if video_id not in global_entity.videos:
            global_entity.videos[video_id] = {
                'mention_count': 0,
                'title': video_title
            }
        global_entity.videos[video_id]['mention_count'] = entity.mention_count

        # Update entity index with aliases
        for alias in entity.aliases:
            alias_normalized = alias.lower()
            graph.entity_index[alias_normalized] = entity_id

    # Add co-occurrence relationships
    for entity_a_id, entity_b_id in new_cooccurrences:
        entity_a = graph.entities.get(entity_a_id)
        entity_b = graph.entities.get(entity_b_id)

        if entity_a and entity_b:
            # Update entity_a's related_entities
            if entity_b_id not in entity_a.related_entities:
                entity_a.related_entities[entity_b_id] = {'count': 0, 'videos': []}
            entity_a.related_entities[entity_b_id]['count'] += 1
            if video_id not in entity_a.related_entities[entity_b_id]['videos']:
                entity_a.related_entities[entity_b_id]['videos'].append(video_id)

            # Update entity_b's related_entities (symmetric)
            if entity_a_id not in entity_b.related_entities:
                entity_b.related_entities[entity_a_id] = {'count': 0, 'videos': []}
            entity_b.related_entities[entity_a_id]['count'] += 1
            if video_id not in entity_b.related_entities[entity_a_id]['videos']:
                entity_b.related_entities[entity_a_id]['videos'].append(video_id)

    graph.last_updated = datetime.now()

    # Save graph (with optional locking)
    if sync_engine and version is not None:
        try:
            sync_engine.write_with_lock('concept_graph.json', graph.to_dict(), version)
        except TimeoutError:
            # Fall back to unlocked save
            save_global_graph(graph, output_dir)
    else:
        save_global_graph(graph, output_dir)

    return graph


def remove_from_global_graph(video_id: str, output_dir: Path) -> GlobalConceptGraph:
    """Remove a video from the global graph."""
    graph = load_global_graph(output_dir)
    if not graph:
        return GlobalConceptGraph()

    # Remove video from all entities
    entities_to_remove = []
    for entity_id, entity in graph.entities.items():
        if video_id in entity.videos:
            del entity.videos[video_id]

        # Remove entity if no videos left
        if not entity.videos:
            entities_to_remove.append(entity_id)

    for entity_id in entities_to_remove:
        del graph.entities[entity_id]
        # Clean up index
        graph.entity_index = {
            k: v for k, v in graph.entity_index.items()
            if v != entity_id
        }

    graph.last_updated = datetime.now()
    save_global_graph(graph, output_dir)

    return graph


def rebuild_global_graph(output_dir: Path) -> GlobalConceptGraph:
    """
    Full rebuild of global graph from all video concept files.

    Use this for consistency checks or after manual edits.
    """
    graph = GlobalConceptGraph()

    # Find all video directories with concept data
    for video_dir in output_dir.iterdir():
        if not video_dir.is_dir():
            continue

        concepts_path = video_dir / 'concepts.json'
        manifest_path = video_dir / 'manifest.json'

        if not concepts_path.exists():
            continue

        # Load video concepts
        video_data = load_video_concepts(video_dir)
        if not video_data:
            continue

        # Get video title from manifest
        video_title = video_dir.name
        if manifest_path.exists():
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
                video_title = manifest.get('metadata', {}).get('title', video_dir.name)

        # Update video_id in case it's missing
        video_data.video_id = video_dir.name

        # Add to graph (without saving intermediate states)
        for entity_id, entity in video_data.entities.items():
            if entity_id not in graph.entities:
                graph.entities[entity_id] = GlobalEntity(
                    id=entity_id,
                    canonical_name=entity.canonical_name,
                    entity_type=entity.entity_type
                )

            global_entity = graph.entities[entity_id]
            global_entity.videos[video_data.video_id] = {
                'mention_count': entity.mention_count,
                'title': video_title
            }

            for alias in entity.aliases:
                graph.entity_index[alias.lower()] = entity_id

        # Add co-occurrences
        for cooccur in video_data.cooccurrences:
            entity_a = graph.entities.get(cooccur.entity_a_id)
            entity_b = graph.entities.get(cooccur.entity_b_id)

            if entity_a and entity_b:
                if cooccur.entity_b_id not in entity_a.related_entities:
                    entity_a.related_entities[cooccur.entity_b_id] = {'count': 0, 'videos': []}
                entity_a.related_entities[cooccur.entity_b_id]['count'] += cooccur.count
                if video_data.video_id not in entity_a.related_entities[cooccur.entity_b_id]['videos']:
                    entity_a.related_entities[cooccur.entity_b_id]['videos'].append(video_data.video_id)

                if cooccur.entity_a_id not in entity_b.related_entities:
                    entity_b.related_entities[cooccur.entity_a_id] = {'count': 0, 'videos': []}
                entity_b.related_entities[cooccur.entity_a_id]['count'] += cooccur.count
                if video_data.video_id not in entity_b.related_entities[cooccur.entity_a_id]['videos']:
                    entity_b.related_entities[cooccur.entity_a_id]['videos'].append(video_data.video_id)

    graph.last_updated = datetime.now()
    save_global_graph(graph, output_dir)

    print(f"Rebuilt global graph: {len(graph.entities)} entities across {len(set(v for e in graph.entities.values() for v in e.videos))} videos")

    return graph


def load_global_graph(output_dir: Path) -> Optional[GlobalConceptGraph]:
    """Load global concept graph."""
    graph_path = output_dir / 'concept_graph.json'

    if not graph_path.exists():
        return None

    with open(graph_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return GlobalConceptGraph.from_dict(data)


def save_global_graph(graph: GlobalConceptGraph, output_dir: Path) -> Path:
    """Save global concept graph."""
    graph_path = output_dir / 'concept_graph.json'

    with open(graph_path, 'w', encoding='utf-8') as f:
        json.dump(graph.to_dict(), f, indent=2, ensure_ascii=False)

    return graph_path


# =============================================================================
# Query Interface
# =============================================================================

def search_entities(
    query: str,
    graph: GlobalConceptGraph,
    limit: int = 10
) -> list[GlobalEntity]:
    """
    Search for entities matching a query string.

    Args:
        query: Search string
        graph: Global concept graph
        limit: Max results to return

    Returns:
        List of matching GlobalEntity objects
    """
    query_lower = query.lower()
    query_normalized = normalize_entity_id(query)

    results = []

    for entity_id, entity in graph.entities.items():
        # Exact ID match
        if entity_id == query_normalized:
            results.append((entity, 100))
            continue

        # Check index (aliases)
        if query_lower in graph.entity_index and graph.entity_index[query_lower] == entity_id:
            results.append((entity, 90))
            continue

        # Partial match on canonical name
        if query_lower in entity.canonical_name.lower():
            results.append((entity, 70))
            continue

        # Partial match on ID
        if query_lower in entity_id:
            results.append((entity, 50))

    # Sort by score, then by total mentions
    results.sort(key=lambda x: (x[1], x[0].total_mentions), reverse=True)

    return [r[0] for r in results[:limit]]


def get_related_entities(
    entity_id: str,
    graph: GlobalConceptGraph,
    limit: int = 10
) -> list[tuple[str, int]]:
    """
    Get entities that frequently co-occur with given entity.

    Args:
        entity_id: ID of entity to find relations for
        graph: Global concept graph
        limit: Max results to return

    Returns:
        List of (entity_id, cooccurrence_count) tuples
    """
    if entity_id not in graph.entities:
        return []

    entity = graph.entities[entity_id]

    # Sort related entities by co-occurrence count
    related = sorted(
        entity.related_entities.items(),
        key=lambda x: x[1]['count'],
        reverse=True
    )

    return [(r[0], r[1]['count']) for r in related[:limit]]


def get_entity_videos(
    entity_id: str,
    graph: GlobalConceptGraph
) -> list[dict]:
    """
    Get all videos where an entity appears.

    Args:
        entity_id: ID of entity
        graph: Global concept graph

    Returns:
        List of {video_id, title, mention_count} dicts
    """
    if entity_id not in graph.entities:
        return []

    entity = graph.entities[entity_id]

    results = [
        {
            'video_id': vid,
            'title': info.get('title', vid),
            'mention_count': info.get('mention_count', 0)
        }
        for vid, info in entity.videos.items()
    ]

    # Sort by mention count descending
    results.sort(key=lambda x: x['mention_count'], reverse=True)

    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Concept graph management for world-model"
    )
    parser.add_argument(
        'video_id',
        nargs='?',
        help="Video ID to extract concepts from"
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path(__file__).parent.parent / 'output',
        help="Output directory (default: ./output)"
    )
    parser.add_argument(
        '--rebuild',
        action='store_true',
        help="Rebuild global graph from all videos"
    )
    parser.add_argument(
        '--query', '-q',
        type=str,
        help="Query for an entity across all content"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='en_core_web_md',
        help="SpaCy model to use (default: en_core_web_md)"
    )

    args = parser.parse_args()

    if args.rebuild:
        print("Rebuilding global concept graph...")
        graph = rebuild_global_graph(args.output)
        print(f"Done. Graph saved to {args.output / 'concept_graph.json'}")
        return

    if args.query:
        print(f"Searching for: {args.query}")
        graph = load_global_graph(args.output)
        if not graph:
            print("No global graph found. Run --rebuild first.")
            return

        results = search_entities(args.query, graph)
        if not results:
            print("No matching entities found.")
            return

        for entity in results:
            print(f"\n{entity.canonical_name} ({entity.entity_type})")
            print(f"  Total mentions: {entity.total_mentions}")
            print(f"  Appears in: {len(entity.videos)} video(s)")

            # Show related entities
            related = get_related_entities(entity.id, graph, limit=5)
            if related:
                print("  Related:")
                for rel_id, count in related:
                    rel_entity = graph.entities.get(rel_id)
                    if rel_entity:
                        print(f"    - {rel_entity.canonical_name} (co-occurs {count}x)")
        return

    if not args.video_id:
        parser.print_help()
        return

    # Extract concepts for single video
    work_dir = args.output / args.video_id

    if not work_dir.exists():
        print(f"Video directory not found: {work_dir}")
        return

    # Load transcript
    transcript_path = work_dir / 'transcript.json'
    if not transcript_path.exists():
        print(f"No transcript found at {transcript_path}")
        return

    from .extract import load_transcript
    transcript = load_transcript(work_dir)

    print(f"Extracting concepts from {args.video_id}...")
    print(f"  Transcript: {len(transcript)} segments")

    # Extract entities
    video_data = extract_entities(transcript, model_name=args.model)
    video_data.video_id = args.video_id

    print(f"  Entities: {len(video_data.entities)} found")

    # Compute co-occurrences
    video_data.cooccurrences = compute_cooccurrences(video_data.entities)
    print(f"  Co-occurrences: {len(video_data.cooccurrences)} pairs")

    # Save per-video data
    save_video_concepts(video_data, work_dir)
    print(f"  Saved: {work_dir / 'concepts.json'}")

    # Update global graph
    manifest_path = work_dir / 'manifest.json'
    video_title = args.video_id
    if manifest_path.exists():
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
            video_title = manifest.get('metadata', {}).get('title', args.video_id)

    update_global_graph(video_data, video_title, args.output)
    print(f"  Updated global graph")

    # Show top entities
    print(f"\nTop entities:")
    sorted_entities = sorted(
        video_data.entities.values(),
        key=lambda e: e.mention_count,
        reverse=True
    )
    for entity in sorted_entities[:10]:
        print(f"  {entity.canonical_name} ({entity.entity_type}): {entity.mention_count} mentions")


if __name__ == '__main__':
    main()
