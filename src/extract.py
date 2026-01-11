"""
Extract Stage: Raw Assets → Semantic Fragments

Handles transcription, keyframe extraction, and semantic analysis.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING
import re

if TYPE_CHECKING:
    from .concepts import VideoConceptData


# =============================================================================
# Whisper Model Cache
# =============================================================================

# Module-level cache for Whisper model (expensive to load)
_whisper_model = None
_whisper_model_name = None


def get_whisper_model(model_name: str = "base"):
    """
    Get cached Whisper model instance.

    Loads model on first call, reuses on subsequent calls.
    This significantly speeds up streaming transcription.

    Args:
        model_name: Whisper model name ("tiny", "base", "small", "medium", "large")

    Returns:
        Loaded Whisper model, or None if import fails
    """
    global _whisper_model, _whisper_model_name

    # Return cached if same model requested
    if _whisper_model is not None and _whisper_model_name == model_name:
        return _whisper_model

    try:
        import whisper
    except ImportError:
        print("  Warning: Whisper not installed. Run: pip install openai-whisper")
        return None

    print(f"  Loading Whisper model '{model_name}'...")
    _whisper_model = whisper.load_model(model_name)
    _whisper_model_name = model_name
    print(f"  Whisper model loaded.")

    return _whisper_model


def unload_whisper_model():
    """Unload cached Whisper model to free memory."""
    global _whisper_model, _whisper_model_name
    _whisper_model = None
    _whisper_model_name = None


@dataclass
class TranscriptSegment:
    """A segment of transcribed speech."""
    start: float  # seconds
    end: float
    text: str
    speaker: Optional[str] = None  # Future: diarization


@dataclass
class Keyframe:
    """A semantically-sampled video frame."""
    timestamp: float
    path: Path
    trigger: str  # Why this frame was captured
    ocr_text: Optional[str] = None


@dataclass
class ExtractedContent:
    """Container for all extracted fragments."""
    video_id: str
    transcript: list[TranscriptSegment] = field(default_factory=list)
    keyframes: list[Keyframe] = field(default_factory=list)
    topics: list[dict] = field(default_factory=list)  # Future: topic segmentation
    concepts: Optional['VideoConceptData'] = None  # Entity/concept graph data


def extract(assets, work_dir: Path, extract_concepts: bool = True) -> ExtractedContent:
    """
    Extract semantic fragments from media assets.

    Args:
        assets: MediaAssets from transform stage
        work_dir: Working directory for this video
        extract_concepts: Whether to extract entities/concepts (default True)

    Returns:
        ExtractedContent with transcript, keyframes, and concepts
    """
    content = ExtractedContent(video_id=assets.video_id)

    # Try existing subtitles first, fall back to Whisper
    if assets.subtitles_path and assets.subtitles_path.exists():
        content.transcript = _parse_existing_subtitles(assets.subtitles_path)
        print(f"  Using existing subtitles: {len(content.transcript)} segments")
    elif assets.audio_path and assets.audio_path.exists():
        content.transcript = _transcribe_with_whisper(assets.audio_path)
        print(f"  Transcribed with Whisper: {len(content.transcript)} segments")

    # Save transcript
    _save_transcript(content.transcript, work_dir)

    # Extract concepts/entities
    if extract_concepts and content.transcript:
        try:
            from .concepts import extract_entities, compute_cooccurrences, save_video_concepts
            print(f"  Extracting concepts...")
            concept_data = extract_entities(content.transcript)
            concept_data.video_id = assets.video_id
            concept_data.cooccurrences = compute_cooccurrences(concept_data.entities)
            save_video_concepts(concept_data, work_dir)
            content.concepts = concept_data
            print(f"  Concepts: {len(concept_data.entities)} entities, {len(concept_data.cooccurrences)} co-occurrences")
        except ImportError as e:
            print(f"  Skipping concept extraction: {e}")
        except Exception as e:
            print(f"  Warning: Concept extraction failed: {e}")

    # TODO: Semantic keyframe extraction (when video available)
    # content.keyframes = _extract_semantic_keyframes(assets, content.transcript)

    return content


def _parse_existing_subtitles(subtitles_path: Path) -> list[TranscriptSegment]:
    """Parse existing subtitle file into segments."""
    segments = []

    if subtitles_path.suffix == '.json3':
        segments = _parse_json3_subtitles(subtitles_path)
    elif subtitles_path.suffix == '.vtt':
        segments = _parse_vtt_subtitles(subtitles_path)

    return segments


def _parse_json3_subtitles(path: Path) -> list[TranscriptSegment]:
    """Parse YouTube's json3 subtitle format."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    segments = []
    events = data.get('events', [])

    for event in events:
        if 'segs' not in event:
            continue

        start = event.get('tStartMs', 0) / 1000
        duration = event.get('dDurationMs', 0) / 1000
        text = ''.join(seg.get('utf8', '') for seg in event['segs'])
        text = text.strip()

        if text:
            segments.append(TranscriptSegment(
                start=start,
                end=start + duration,
                text=text
            ))

    return segments


def _parse_vtt_subtitles(path: Path) -> list[TranscriptSegment]:
    """Parse VTT subtitle format."""
    segments = []

    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Simple VTT parsing
    pattern = r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})\n(.+?)(?=\n\n|\Z)'
    matches = re.findall(pattern, content, re.DOTALL)

    for start_str, end_str, text in matches:
        start = _parse_timestamp(start_str)
        end = _parse_timestamp(end_str)
        text = text.strip().replace('\n', ' ')

        if text:
            segments.append(TranscriptSegment(
                start=start,
                end=end,
                text=text
            ))

    return segments


def _parse_timestamp(ts: str) -> float:
    """Convert HH:MM:SS.mmm to seconds."""
    parts = ts.replace(',', '.').split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def _transcribe_with_whisper(audio_path: Path) -> list[TranscriptSegment]:
    """Transcribe audio using OpenAI Whisper."""
    try:
        import whisper
    except ImportError:
        print("  Warning: Whisper not installed. Run: pip install openai-whisper")
        return []

    print(f"  Loading Whisper model...")
    model = whisper.load_model("base")  # Options: tiny, base, small, medium, large

    print(f"  Transcribing {audio_path}...")
    result = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        verbose=False
    )

    segments = []
    for seg in result.get('segments', []):
        segments.append(TranscriptSegment(
            start=seg['start'],
            end=seg['end'],
            text=seg['text'].strip()
        ))

    return segments


def _save_transcript(segments: list[TranscriptSegment], work_dir: Path):
    """Save transcript to JSON."""
    transcript_path = work_dir / 'transcript.json'

    data = [
        {
            'start': seg.start,
            'end': seg.end,
            'text': seg.text,
            'speaker': seg.speaker
        }
        for seg in segments
    ]

    with open(transcript_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_transcript(work_dir: Path) -> list[TranscriptSegment]:
    """Load previously saved transcript."""
    transcript_path = work_dir / 'transcript.json'

    if not transcript_path.exists():
        return []

    with open(transcript_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return [
        TranscriptSegment(
            start=seg['start'],
            end=seg['end'],
            text=seg['text'],
            speaker=seg.get('speaker')
        )
        for seg in data
    ]


def transcribe_chunk(audio_path: Path, time_offset: float = 0.0) -> list[TranscriptSegment]:
    """
    Transcribe a single audio chunk for streaming ingestion.

    Args:
        audio_path: Path to audio chunk file
        time_offset: Time offset to add to timestamps (seconds from stream start)

    Returns:
        List of TranscriptSegments with adjusted timestamps
    """
    try:
        import whisper
    except ImportError:
        print("  Warning: Whisper not installed. Run: pip install openai-whisper")
        return []

    # Use base model for balance of speed/quality
    # Could cache model instance for better performance
    model = whisper.load_model("base")

    result = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        verbose=False
    )

    segments = []
    for seg in result.get('segments', []):
        segments.append(TranscriptSegment(
            start=seg['start'] + time_offset,
            end=seg['end'] + time_offset,
            text=seg['text'].strip()
        ))

    return segments


def transcribe_chunk_fast(
    audio_path: Path,
    time_offset: float = 0.0,
    model_name: str = "base"
) -> list[TranscriptSegment]:
    """
    Fast transcription using cached Whisper model.

    Significantly faster than transcribe_chunk() for streaming because
    it reuses the loaded model instead of loading it fresh each time.

    Args:
        audio_path: Path to audio chunk file
        time_offset: Time offset to add to timestamps (seconds from stream start)
        model_name: Whisper model to use

    Returns:
        List of TranscriptSegments with adjusted timestamps
    """
    model = get_whisper_model(model_name)
    if model is None:
        return []

    result = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        verbose=False
    )

    segments = []
    for seg in result.get('segments', []):
        segments.append(TranscriptSegment(
            start=seg['start'] + time_offset,
            end=seg['end'] + time_offset,
            text=seg['text'].strip()
        ))

    return segments


def merge_overlapping_segments(
    existing: list[TranscriptSegment],
    new_segments: list[TranscriptSegment],
    overlap_duration: float = 3.0
) -> list[TranscriptSegment]:
    """
    Merge new segments with existing ones, handling overlap.

    Used for streaming where chunks have overlapping audio.

    Args:
        existing: Previously accumulated segments
        new_segments: New segments from latest chunk
        overlap_duration: Expected overlap in seconds

    Returns:
        Merged segment list without duplicates
    """
    if not existing:
        return new_segments

    if not new_segments:
        return existing

    # Find the overlap boundary
    last_existing_time = existing[-1].end

    # Filter new segments - skip any that fall entirely within overlap zone
    # and would duplicate existing content
    merged = existing.copy()

    for seg in new_segments:
        # Skip segments that are clearly in the overlap zone
        if seg.end <= last_existing_time - overlap_duration:
            continue

        # Check for fuzzy match with recent segments (simple approach)
        is_duplicate = False
        for recent in existing[-5:]:  # Check last 5 segments
            if _segments_similar(seg, recent):
                is_duplicate = True
                break

        if not is_duplicate:
            merged.append(seg)

    return merged


def _segments_similar(a: TranscriptSegment, b: TranscriptSegment, threshold: float = 0.8) -> bool:
    """Check if two segments are similar enough to be duplicates."""
    # Simple word overlap check
    words_a = set(a.text.lower().split())
    words_b = set(b.text.lower().split())

    if not words_a or not words_b:
        return False

    intersection = len(words_a & words_b)
    union = len(words_a | words_b)

    return (intersection / union) >= threshold if union > 0 else False


if __name__ == '__main__':
    # Test with existing transform output
    from pathlib import Path
    import sys

    if len(sys.argv) < 2:
        print("Usage: python extract.py <video_id>")
        sys.exit(1)

    video_id = sys.argv[1]
    output_dir = Path(__file__).parent.parent / 'output'
    work_dir = output_dir / video_id

    # Load transform metadata
    meta_path = work_dir / 'transform_meta.json'
    if not meta_path.exists():
        print(f"No transform metadata found at {meta_path}")
        sys.exit(1)

    from transform import MediaAssets
    with open(meta_path) as f:
        meta = json.load(f)

    # Create minimal assets object for testing
    class Assets:
        pass

    assets = Assets()
    assets.video_id = meta['video_id']
    assets.subtitles_path = Path(meta['subtitles_path']) if meta.get('subtitles_path') else None
    assets.audio_path = Path(meta['audio_path']) if meta.get('audio_path') else None

    content = extract(assets, work_dir)
    print(f"\nExtracted {len(content.transcript)} transcript segments")
