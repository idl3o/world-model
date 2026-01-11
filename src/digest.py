"""
Digest Stage: Fragments → Unified Document

Synthesizes extracted content into AI-native format.
"""

import json
from pathlib import Path
from datetime import timedelta
from typing import Optional


def digest(assets, content, work_dir: Path) -> Path:
    """
    Create unified digest document from extracted content.

    Args:
        assets: MediaAssets from transform stage
        content: ExtractedContent from extract stage
        work_dir: Working directory for this video

    Returns:
        Path to generated digest.md
    """
    # Generate markdown digest
    markdown = _generate_markdown(assets, content)
    digest_path = work_dir / 'digest.md'
    with open(digest_path, 'w', encoding='utf-8') as f:
        f.write(markdown)

    # Generate manifest JSON
    manifest = _generate_manifest(assets, content, work_dir)
    manifest_path = work_dir / 'manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return digest_path


def _generate_markdown(assets, content) -> str:
    """Generate human/AI-readable markdown digest."""
    lines = []

    # Header
    lines.append(f"# {assets.title}")
    lines.append("")
    lines.append(f"**Channel**: {assets.channel}  ")
    lines.append(f"**Duration**: {_format_duration(assets.duration)}  ")
    if assets.upload_date:
        lines.append(f"**Published**: {_format_date(assets.upload_date)}  ")
    lines.append("")

    # Description (truncated if very long)
    if assets.description:
        desc = assets.description[:1000]
        if len(assets.description) > 1000:
            desc += "..."
        lines.append("## Description")
        lines.append("")
        lines.append(desc)
        lines.append("")

    # Chapters (if available)
    if assets.chapters:
        lines.append("## Chapters")
        lines.append("")
        for chapter in assets.chapters:
            start = _format_duration(chapter.get('start_time', 0))
            title = chapter.get('title', 'Untitled')
            lines.append(f"- [{start}] {title}")
        lines.append("")

    # Full transcript with timestamps
    if content.transcript:
        lines.append("## Transcript")
        lines.append("")
        lines.extend(_format_transcript(content.transcript))
        lines.append("")

    # Keyframes (if any)
    if content.keyframes:
        lines.append("## Keyframes")
        lines.append("")
        for kf in content.keyframes:
            ts = _format_duration(kf.timestamp)
            lines.append(f"### [{ts}] {kf.trigger}")
            lines.append(f"![Keyframe]({kf.path.name})")
            if kf.ocr_text:
                lines.append(f"**Visual text**: {kf.ocr_text}")
            lines.append("")

    # Concepts and Entities (if extracted)
    if content.concepts and content.concepts.entities:
        lines.append("## Key Concepts & Entities")
        lines.append("")

        # Group entities by type
        from collections import defaultdict
        by_type = defaultdict(list)
        for entity in content.concepts.entities.values():
            by_type[entity.entity_type].append(entity)

        # Display order
        type_order = ['PERSON', 'ORG', 'GPE', 'WORK_OF_ART', 'EVENT', 'PRODUCT', 'NORP', 'LOC', 'LAW']

        for entity_type in type_order:
            if entity_type not in by_type:
                continue

            entities = sorted(by_type[entity_type], key=lambda e: -e.mention_count)[:8]
            lines.append(f"### {entity_type}")
            for entity in entities:
                first_mention = min(entity.mentions, key=lambda m: m.start)
                ts = _format_duration(first_mention.start)
                lines.append(f"- **{entity.canonical_name}** ({entity.mention_count} mentions, first at [{ts}])")
            lines.append("")

        # Top co-occurrences
        if content.concepts.cooccurrences:
            lines.append("### Relationships")
            for cooccur in content.concepts.cooccurrences[:5]:
                entity_a = content.concepts.entities.get(cooccur.entity_a_id)
                entity_b = content.concepts.entities.get(cooccur.entity_b_id)
                if entity_a and entity_b:
                    lines.append(f"- {entity_a.canonical_name} ↔ {entity_b.canonical_name} (co-occur {cooccur.count}x)")
            lines.append("")

    # Footer
    lines.append("---")
    lines.append(f"*Processed by World Model v0.1*")

    return "\n".join(lines)


def _format_transcript(segments) -> list[str]:
    """Format transcript with intelligent paragraph grouping."""
    if not segments:
        return ["*No transcript available*"]

    lines = []
    current_paragraph = []
    last_end = 0

    for seg in segments:
        # Start new paragraph on significant pause (>2 seconds)
        if seg.start - last_end > 2.0 and current_paragraph:
            timestamp = _format_duration(current_paragraph[0].start)
            text = " ".join(s.text for s in current_paragraph)
            lines.append(f"**[{timestamp}]** {text}")
            lines.append("")
            current_paragraph = []

        current_paragraph.append(seg)
        last_end = seg.end

    # Flush remaining
    if current_paragraph:
        timestamp = _format_duration(current_paragraph[0].start)
        text = " ".join(s.text for s in current_paragraph)
        lines.append(f"**[{timestamp}]** {text}")

    return lines


def _generate_manifest(assets, content, work_dir: Path) -> dict:
    """Generate structured manifest for machine consumption."""
    manifest = {
        "version": "1.0",
        "video_id": assets.video_id,
        "metadata": {
            "title": assets.title,
            "channel": assets.channel,
            "duration": assets.duration,
            "upload_date": assets.upload_date,
            "description": assets.description,
        },
        "chapters": assets.chapters,
        "files": {
            "audio": str(assets.audio_path) if assets.audio_path else None,
            "thumbnail": str(assets.thumbnail_path) if assets.thumbnail_path else None,
            "transcript": str(work_dir / "transcript.json"),
            "digest": str(work_dir / "digest.md"),
        },
        "stats": {
            "transcript_segments": len(content.transcript),
            "keyframes": len(content.keyframes),
            "word_count": sum(len(s.text.split()) for s in content.transcript),
            "entity_count": len(content.concepts.entities) if content.concepts else 0,
        }
    }

    # Add concepts summary if available
    if content.concepts and content.concepts.entities:
        top_entities = sorted(
            content.concepts.entities.values(),
            key=lambda e: -e.mention_count
        )[:20]

        manifest["concepts"] = {
            "total_entities": len(content.concepts.entities),
            "total_cooccurrences": len(content.concepts.cooccurrences),
            "top_entities": [
                {
                    "name": e.canonical_name,
                    "type": e.entity_type,
                    "mentions": e.mention_count
                }
                for e in top_entities
            ]
        }

    return manifest


def _format_duration(seconds: float) -> str:
    """Format seconds as HH:MM:SS or MM:SS."""
    if seconds is None:
        return "00:00"
    td = timedelta(seconds=int(seconds))
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _format_date(date_str: str) -> str:
    """Format YYYYMMDD to readable date."""
    if len(date_str) == 8:
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    return date_str


def generate_partial_digest(
    assets,
    content,
    work_dir: Path,
    is_live: bool = True
) -> Path:
    """
    Generate a partial/streaming digest with live indicator.

    Used during real-time stream capture to provide an updating view
    of the content as it's being processed.

    Args:
        assets: MediaAssets (may have partial data)
        content: ExtractedContent (accumulating)
        work_dir: Working directory
        is_live: Whether stream is still live

    Returns:
        Path to digest.partial.md (or digest.md if finalized)
    """
    from datetime import datetime

    lines = []

    # Live banner
    if is_live:
        lines.append("> **LIVE** - Content updating in real-time")
        lines.append(f"> Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

    # Header
    lines.append(f"# {assets.title}")
    lines.append("")
    lines.append(f"**Channel**: {getattr(assets, 'channel', 'Unknown')}  ")
    lines.append(f"**Duration**: {_format_duration(getattr(assets, 'duration', 0))}  ")
    lines.append("")

    # Stats
    word_count = sum(len(s.text.split()) for s in content.transcript) if content.transcript else 0
    entity_count = len(content.concepts.entities) if content.concepts else 0
    lines.append(f"**Words**: {word_count:,}  ")
    lines.append(f"**Entities**: {entity_count}  ")
    lines.append("")

    # Transcript
    if content.transcript:
        lines.append("## Transcript")
        lines.append("")
        lines.extend(_format_transcript(content.transcript))
        lines.append("")

    # Entities (if available)
    if content.concepts and content.concepts.entities:
        lines.append("## Key Entities")
        lines.append("")

        from collections import defaultdict
        by_type = defaultdict(list)
        for entity in content.concepts.entities.values():
            by_type[entity.entity_type].append(entity)

        for entity_type in ['PERSON', 'ORG', 'GPE', 'WORK_OF_ART']:
            if entity_type not in by_type:
                continue
            entities = sorted(by_type[entity_type], key=lambda e: -e.mention_count)[:5]
            lines.append(f"### {entity_type}")
            for entity in entities:
                lines.append(f"- **{entity.canonical_name}** ({entity.mention_count} mentions)")
            lines.append("")

    # Footer
    lines.append("---")
    lines.append("*Generated by World Model (streaming)*")

    # Write to file
    filename = "digest.partial.md" if is_live else "digest.md"
    digest_path = work_dir / filename
    with open(digest_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    return digest_path


if __name__ == '__main__':
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python digest.py <video_id>")
        sys.exit(1)

    video_id = sys.argv[1]
    output_dir = Path(__file__).parent.parent / 'output'
    work_dir = output_dir / video_id

    # Load previous stages
    from extract import load_transcript, ExtractedContent

    meta_path = work_dir / 'transform_meta.json'
    with open(meta_path) as f:
        meta = json.load(f)

    class Assets:
        pass

    assets = Assets()
    for key, value in meta.items():
        setattr(assets, key, value)

    content = ExtractedContent(video_id=video_id)
    content.transcript = load_transcript(work_dir)

    digest_path = digest(assets, content, work_dir)
    print(f"Digest written to: {digest_path}")
