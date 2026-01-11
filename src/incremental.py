"""
Incremental Processing Engine

Provides per-chunk entity extraction and progressive digest generation
for real-time stream processing.

Usage:
    extractor = IncrementalEntityExtractor()
    for chunk_segments in stream:
        result = extractor.process_chunk(chunk_segments)
        print(f"New entities: {result.new_entities}")
    final_data = extractor.flush_to_video_data("video_id")
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
from collections import defaultdict

from .extract import TranscriptSegment
from .concepts import (
    EntityMention, ExtractedEntity, Cooccurrence, VideoConceptData,
    normalize_entity_id, choose_canonical_name
)


@dataclass
class ChunkExtractionResult:
    """Results from processing a single chunk."""
    new_entities: list[str]          # IDs of newly discovered entities
    updated_entities: list[str]      # IDs with new mentions added
    new_cooccurrences: list[tuple[str, str]]  # New (entity_a, entity_b) pairs
    segment_count: int
    word_count: int


class IncrementalEntityExtractor:
    """
    Extract entities incrementally as transcript chunks arrive.

    Maintains accumulated state across chunks and computes co-occurrences
    using a sliding time window.
    """

    # Entity types we extract
    RELEVANT_TYPES = {
        'PERSON', 'ORG', 'GPE', 'LOC', 'WORK_OF_ART',
        'EVENT', 'PRODUCT', 'LAW', 'NORP'
    }

    def __init__(
        self,
        model_name: str = "en_core_web_md",
        window_seconds: float = 30.0
    ):
        self.model_name = model_name
        self.window_seconds = window_seconds

        # Accumulated state
        self.entities: dict[str, ExtractedEntity] = {}
        self.all_segments: list[TranscriptSegment] = []

        # For sliding window co-occurrence
        # List of (timestamp, entity_id) for recent mentions
        self.mention_timeline: list[tuple[float, str]] = []

        # Track which entity pairs we've already counted
        self.seen_cooccurrences: set[tuple[str, str]] = set()

        # SpaCy model (lazy loaded)
        self._nlp = None
        self._nlp_loaded = False

    def _load_nlp(self):
        """Lazy load SpaCy model."""
        if self._nlp_loaded:
            return self._nlp

        try:
            import spacy
        except ImportError:
            print("Warning: SpaCy not installed")
            self._nlp_loaded = True
            return None

        for model in [self.model_name, "en_core_web_sm"]:
            try:
                self._nlp = spacy.load(model)
                if model != self.model_name:
                    print(f"  Using fallback SpaCy model: {model}")
                break
            except OSError:
                continue

        self._nlp_loaded = True
        return self._nlp

    def process_chunk(
        self,
        segments: list[TranscriptSegment],
        context_segments: list[TranscriptSegment] = None
    ) -> ChunkExtractionResult:
        """
        Process a chunk of transcript segments.

        Args:
            segments: New transcript segments from this chunk
            context_segments: Optional previous segments for context

        Returns:
            ChunkExtractionResult with new/updated entities
        """
        nlp = self._load_nlp()

        new_entities = []
        updated_entities = []
        new_cooccurrences = []
        word_count = 0

        if nlp is None:
            # No model available, just count words
            for seg in segments:
                word_count += len(seg.text.split())
            return ChunkExtractionResult(
                new_entities=[],
                updated_entities=[],
                new_cooccurrences=[],
                segment_count=len(segments),
                word_count=word_count
            )

        # Get segment offset for indexing
        seg_offset = len(self.all_segments)

        # Process each segment
        for local_idx, segment in enumerate(segments):
            seg_idx = seg_offset + local_idx
            text = segment.text
            word_count += len(text.split())

            # Build context from surrounding segments
            all_segs = (context_segments or []) + self.all_segments[-5:] + segments
            context_text = ' '.join(s.text for s in all_segs[-5:])[:200]

            # Run NER
            doc = nlp(text)

            for ent in doc.ents:
                if ent.label_ not in self.RELEVANT_TYPES:
                    continue

                if len(ent.text.strip()) < 2:
                    continue

                entity_id = normalize_entity_id(ent.text)
                if not entity_id:
                    continue

                # Create mention
                mention = EntityMention(
                    start=segment.start,
                    end=segment.end,
                    text=ent.text,
                    context=context_text,
                    segment_index=seg_idx
                )

                # Add to timeline for co-occurrence
                self.mention_timeline.append((segment.start, entity_id))

                # Update or create entity
                if entity_id in self.entities:
                    self.entities[entity_id].aliases.add(ent.text)
                    self.entities[entity_id].mentions.append(mention)
                    if entity_id not in updated_entities:
                        updated_entities.append(entity_id)
                else:
                    self.entities[entity_id] = ExtractedEntity(
                        id=entity_id,
                        canonical_name=ent.text,  # Will be refined later
                        entity_type=ent.label_,
                        aliases={ent.text},
                        mentions=[mention]
                    )
                    new_entities.append(entity_id)

        # Update canonical names for new/updated entities
        for entity_id in new_entities + updated_entities:
            if entity_id in self.entities:
                self.entities[entity_id].canonical_name = choose_canonical_name(
                    self.entities[entity_id].aliases
                )

        # Compute new co-occurrences from this chunk
        new_cooccurrences = self._compute_new_cooccurrences()

        # Add segments to accumulated list
        self.all_segments.extend(segments)

        # Prune old timeline entries
        self._prune_timeline()

        return ChunkExtractionResult(
            new_entities=new_entities,
            updated_entities=updated_entities,
            new_cooccurrences=new_cooccurrences,
            segment_count=len(segments),
            word_count=word_count
        )

    def _compute_new_cooccurrences(self) -> list[tuple[str, str]]:
        """
        Find new entity co-occurrences within the sliding window.

        Only returns pairs we haven't seen before.
        """
        new_pairs = []

        # Get mentions within window
        if not self.mention_timeline:
            return []

        latest_time = self.mention_timeline[-1][0]
        window_start = latest_time - self.window_seconds

        # Get mentions in window
        window_mentions = [
            (t, eid) for t, eid in self.mention_timeline
            if t >= window_start
        ]

        # Find pairs
        for i, (t1, e1) in enumerate(window_mentions):
            for t2, e2 in window_mentions[i+1:]:
                if e1 != e2:
                    # Normalize pair order
                    pair = tuple(sorted([e1, e2]))
                    if pair not in self.seen_cooccurrences:
                        self.seen_cooccurrences.add(pair)
                        new_pairs.append(pair)

        return new_pairs

    def _prune_timeline(self) -> None:
        """Remove timeline entries older than 2x window size."""
        if not self.mention_timeline:
            return

        latest_time = self.mention_timeline[-1][0]
        cutoff = latest_time - (2 * self.window_seconds)

        self.mention_timeline = [
            (t, eid) for t, eid in self.mention_timeline
            if t >= cutoff
        ]

    def get_sliding_cooccurrences(self) -> list[Cooccurrence]:
        """
        Compute full co-occurrence list from accumulated entities.

        This is more expensive and should be called at finalization.
        """
        # Group mentions by entity
        entity_mentions: dict[str, list[float]] = defaultdict(list)
        for entity_id, entity in self.entities.items():
            for mention in entity.mentions:
                entity_mentions[entity_id].append(mention.start)

        # Count co-occurrences
        cooccurrence_counts: dict[tuple[str, str], dict] = {}

        for e1_id, e1_times in entity_mentions.items():
            for e2_id, e2_times in entity_mentions.items():
                if e1_id >= e2_id:  # Avoid duplicates
                    continue

                # Find overlapping windows
                pair = (e1_id, e2_id)
                timestamps = []

                for t1 in e1_times:
                    for t2 in e2_times:
                        if abs(t1 - t2) <= self.window_seconds:
                            timestamps.append(min(t1, t2))
                            break

                if timestamps:
                    # Deduplicate timestamps
                    unique_times = sorted(set(timestamps))[:50]
                    cooccurrence_counts[pair] = {
                        'count': len(unique_times),
                        'timestamps': unique_times
                    }

        # Convert to Cooccurrence objects
        result = []
        for (e1, e2), data in cooccurrence_counts.items():
            result.append(Cooccurrence(
                entity_a_id=e1,
                entity_b_id=e2,
                count=data['count'],
                timestamps=data['timestamps']
            ))

        # Sort by count descending
        result.sort(key=lambda c: c.count, reverse=True)
        return result

    def get_accumulated_entities(self) -> dict[str, ExtractedEntity]:
        """Get all entities extracted so far."""
        return dict(self.entities)

    def flush_to_video_data(self, video_id: str) -> VideoConceptData:
        """
        Create final VideoConceptData from accumulated state.

        Args:
            video_id: Video identifier

        Returns:
            VideoConceptData ready for saving
        """
        cooccurrences = self.get_sliding_cooccurrences()

        return VideoConceptData(
            video_id=video_id,
            extraction_model=self.model_name,
            entities=dict(self.entities),
            cooccurrences=cooccurrences
        )

    def reset(self) -> None:
        """Reset all accumulated state."""
        self.entities.clear()
        self.all_segments.clear()
        self.mention_timeline.clear()
        self.seen_cooccurrences.clear()


class IncrementalDigestGenerator:
    """
    Generate partial digests that update as content arrives.

    Produces digest.partial.md during streaming, then finalizes
    to digest.md when complete.
    """

    def __init__(self, work_dir: Path, title: str = "Stream"):
        self.work_dir = Path(work_dir)
        self.title = title

        # Accumulated content
        self.segments: list[TranscriptSegment] = []
        self.entities: dict[str, ExtractedEntity] = {}
        self.cooccurrences: list[Cooccurrence] = []

        # Metadata
        self.start_time: Optional[datetime] = None
        self.channel: str = ""
        self.description: str = ""
        self.is_live: bool = True

    def set_metadata(
        self,
        title: str = None,
        channel: str = None,
        description: str = None
    ) -> None:
        """Update stream metadata."""
        if title:
            self.title = title
        if channel:
            self.channel = channel
        if description:
            self.description = description

    def update(
        self,
        segments: list[TranscriptSegment] = None,
        entities: dict[str, ExtractedEntity] = None,
        cooccurrences: list[Cooccurrence] = None
    ) -> None:
        """
        Update with new content.

        Args:
            segments: New transcript segments to add
            entities: Updated entity dict (replaces existing)
            cooccurrences: Updated co-occurrence list
        """
        if segments:
            self.segments.extend(segments)
        if entities is not None:
            self.entities = entities
        if cooccurrences is not None:
            self.cooccurrences = cooccurrences

        if self.start_time is None:
            self.start_time = datetime.now()

    def write_partial(self) -> Path:
        """
        Write current state to digest.partial.md.

        Returns:
            Path to partial digest file
        """
        self.work_dir.mkdir(parents=True, exist_ok=True)
        partial_path = self.work_dir / "digest.partial.md"

        content = self._generate_markdown(is_final=False)

        with open(partial_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return partial_path

    def finalize(self) -> Path:
        """
        Write final digest.md and remove partial.

        Returns:
            Path to final digest file
        """
        self.is_live = False
        digest_path = self.work_dir / "digest.md"

        content = self._generate_markdown(is_final=True)

        with open(digest_path, 'w', encoding='utf-8') as f:
            f.write(content)

        # Remove partial
        partial_path = self.work_dir / "digest.partial.md"
        if partial_path.exists():
            partial_path.unlink()

        return digest_path

    def _generate_markdown(self, is_final: bool = False) -> str:
        """Generate markdown content."""
        lines = []

        # Header
        lines.append(f"# {self.title}")
        lines.append("")

        if not is_final:
            lines.append("> **LIVE** - Content updating in real-time")
            lines.append(f"> Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append("")

        # Metadata
        if self.channel:
            lines.append(f"**Channel:** {self.channel}")
        if self.start_time:
            duration = datetime.now() - self.start_time
            mins = int(duration.total_seconds() // 60)
            secs = int(duration.total_seconds() % 60)
            lines.append(f"**Duration:** {mins}m {secs}s")
        lines.append(f"**Words:** {sum(len(s.text.split()) for s in self.segments):,}")
        lines.append("")

        # Description
        if self.description:
            lines.append("## Description")
            lines.append("")
            lines.append(self.description[:500])
            lines.append("")

        # Transcript
        lines.append("## Transcript")
        lines.append("")

        if not self.segments:
            lines.append("*Waiting for content...*")
        else:
            # Group by ~2 second pauses
            paragraphs = self._group_into_paragraphs()
            for para in paragraphs:
                # Timestamp at start of paragraph
                start_time = para[0].start
                mins = int(start_time // 60)
                secs = int(start_time % 60)
                lines.append(f"**[{mins:02d}:{secs:02d}]** ", )
                lines.append(' '.join(s.text for s in para))
                lines.append("")

        # Entities
        if self.entities:
            lines.append("## Key Entities")
            lines.append("")

            # Group by type
            by_type: dict[str, list] = defaultdict(list)
            for entity in self.entities.values():
                by_type[entity.entity_type].append(entity)

            for etype, entities in sorted(by_type.items()):
                # Sort by mention count
                entities.sort(key=lambda e: e.mention_count, reverse=True)
                top_entities = entities[:5]

                lines.append(f"### {etype}")
                for e in top_entities:
                    lines.append(f"- **{e.canonical_name}** ({e.mention_count} mentions)")
                lines.append("")

        # Relationships
        if self.cooccurrences:
            lines.append("## Key Relationships")
            lines.append("")
            for cooc in self.cooccurrences[:10]:
                e1 = self.entities.get(cooc.entity_a_id)
                e2 = self.entities.get(cooc.entity_b_id)
                if e1 and e2:
                    lines.append(f"- {e1.canonical_name} <-> {e2.canonical_name} ({cooc.count}x)")
            lines.append("")

        # Footer
        lines.append("---")
        lines.append(f"*Generated by World Model*")

        return '\n'.join(lines)

    def _group_into_paragraphs(self, pause_threshold: float = 2.0) -> list[list]:
        """Group segments into paragraphs based on pauses."""
        if not self.segments:
            return []

        paragraphs = []
        current = [self.segments[0]]

        for seg in self.segments[1:]:
            prev_end = current[-1].end
            if seg.start - prev_end > pause_threshold:
                paragraphs.append(current)
                current = [seg]
            else:
                current.append(seg)

        if current:
            paragraphs.append(current)

        return paragraphs
