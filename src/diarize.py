"""
Speaker Diarization Module

Uses WhisperX for combined transcription + speaker diarization.
Provides word-level speaker assignment with automatic alignment.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .extract import TranscriptSegment


@dataclass
class DiarizationSegment:
    """A speaker segment from diarization output."""
    start: float
    end: float
    speaker: str
    confidence: float = 1.0


@dataclass
class DiarizationResult:
    """Complete diarization output for an audio file."""
    segments: list[DiarizationSegment] = field(default_factory=list)
    num_speakers: int = 0


# =============================================================================
# WhisperX Model Cache
# =============================================================================

_whisperx_model = None
_whisperx_model_name = None
_align_model = None
_align_metadata = None
_diarize_model = None


def get_whisperx_model(model_name: str = "base", device: str = "cpu"):
    """
    Get cached WhisperX model instance.

    Args:
        model_name: Model size ("tiny", "base", "small", "medium", "large-v2")
        device: "cpu" or "cuda"

    Returns:
        Loaded WhisperX model, or None if import fails
    """
    global _whisperx_model, _whisperx_model_name

    if _whisperx_model is not None and _whisperx_model_name == model_name:
        return _whisperx_model

    try:
        import whisperx
    except ImportError:
        print("  Warning: WhisperX not installed. Run: pip install whisperx")
        return None

    print(f"  Loading WhisperX model '{model_name}'...")
    _whisperx_model = whisperx.load_model(model_name, device=device)
    _whisperx_model_name = model_name
    print(f"  WhisperX model loaded.")

    return _whisperx_model


def get_diarize_model(hf_token: str, device: str = "cpu"):
    """
    Get cached diarization pipeline.

    Args:
        hf_token: HuggingFace token (required for pyannote models)
        device: "cpu" or "cuda"

    Returns:
        Diarization pipeline, or None if unavailable
    """
    global _diarize_model

    if _diarize_model is not None:
        return _diarize_model

    try:
        import whisperx
    except ImportError:
        print("  Warning: WhisperX not installed")
        return None

    print(f"  Loading diarization model...")
    _diarize_model = whisperx.DiarizationPipeline(
        use_auth_token=hf_token,
        device=device
    )
    print(f"  Diarization model loaded.")

    return _diarize_model


def unload_models():
    """Unload all cached models to free memory."""
    global _whisperx_model, _whisperx_model_name
    global _align_model, _align_metadata, _diarize_model

    _whisperx_model = None
    _whisperx_model_name = None
    _align_model = None
    _align_metadata = None
    _diarize_model = None


def transcribe_and_diarize(
    audio_path: Path,
    hf_token: Optional[str] = None,
    model_name: str = "base",
    device: str = "cpu",
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None
) -> tuple[list['TranscriptSegment'], DiarizationResult]:
    """
    Transcribe audio with speaker diarization using WhisperX.

    Performs:
    1. Transcription with WhisperX (faster-whisper backend)
    2. Word-level timestamp alignment
    3. Speaker diarization via pyannote
    4. Speaker assignment to words and segments

    Args:
        audio_path: Path to audio file
        hf_token: HuggingFace token (or HF_TOKEN env var)
        model_name: Whisper model size
        device: "cpu" or "cuda"
        num_speakers: Exact number of speakers (if known)
        min_speakers: Minimum expected speakers
        max_speakers: Maximum expected speakers

    Returns:
        Tuple of (transcript segments with speakers, diarization result)
    """
    from .extract import TranscriptSegment

    # Get token
    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "HuggingFace token required for diarization. "
            "Set HF_TOKEN env var or pass hf_token parameter. "
            "Get token at: https://huggingface.co/settings/tokens"
        )

    try:
        import whisperx
    except ImportError:
        raise ImportError(
            "WhisperX not installed. Run: pip install whisperx"
        )

    # Load audio
    print(f"  Loading audio...")
    audio = whisperx.load_audio(str(audio_path))

    # Step 1: Transcribe
    model = get_whisperx_model(model_name, device)
    if model is None:
        raise RuntimeError("Failed to load WhisperX model")

    print(f"  Transcribing...")
    result = model.transcribe(audio, batch_size=16)
    detected_language = result.get("language", "en")
    print(f"  Detected language: {detected_language}")

    # Step 2: Align (for word-level timestamps)
    global _align_model, _align_metadata

    if _align_model is None:
        print(f"  Loading alignment model...")
        _align_model, _align_metadata = whisperx.load_align_model(
            language_code=detected_language,
            device=device
        )

    print(f"  Aligning transcript...")
    result = whisperx.align(
        result["segments"],
        _align_model,
        _align_metadata,
        audio,
        device,
        return_char_alignments=False
    )

    # Step 3: Diarize
    diarize_model = get_diarize_model(token, device)
    if diarize_model is None:
        raise RuntimeError("Failed to load diarization model")

    print(f"  Running speaker diarization...")
    diarize_kwargs = {}
    if num_speakers is not None:
        diarize_kwargs["num_speakers"] = num_speakers
    if min_speakers is not None:
        diarize_kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        diarize_kwargs["max_speakers"] = max_speakers

    diarize_segments = diarize_model(audio, **diarize_kwargs)

    # Step 4: Assign speakers to words
    print(f"  Assigning speakers to transcript...")
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # Convert to TranscriptSegment
    segments = []
    for seg in result.get("segments", []):
        segments.append(TranscriptSegment(
            start=seg["start"],
            end=seg["end"],
            text=seg.get("text", "").strip(),
            speaker=seg.get("speaker")
        ))

    # Build diarization result
    diar_segments = []
    unique_speakers = set()

    for seg in result.get("segments", []):
        if "speaker" in seg:
            unique_speakers.add(seg["speaker"])
            diar_segments.append(DiarizationSegment(
                start=seg["start"],
                end=seg["end"],
                speaker=seg["speaker"]
            ))

    diarization = DiarizationResult(
        segments=diar_segments,
        num_speakers=len(unique_speakers)
    )

    print(f"  Diarization complete: {len(segments)} segments, {diarization.num_speakers} speakers")

    return segments, diarization


def align_diarization_to_transcript(
    transcript: list['TranscriptSegment'],
    diarization: DiarizationResult
) -> list['TranscriptSegment']:
    """
    Align diarization output to existing transcript segments.

    Used when you have a transcript (e.g., from YouTube subtitles)
    and want to add speaker labels from a separate diarization run.

    Args:
        transcript: Existing transcript segments (speaker field will be updated)
        diarization: Diarization result to align

    Returns:
        Updated transcript segments with speaker labels
    """
    for seg in transcript:
        # Find overlapping diarization segments
        overlaps = []

        for diar in diarization.segments:
            overlap_start = max(seg.start, diar.start)
            overlap_end = min(seg.end, diar.end)
            overlap_duration = overlap_end - overlap_start

            if overlap_duration > 0:
                overlaps.append((diar.speaker, overlap_duration))

        if overlaps:
            # Assign speaker with most time overlap
            seg.speaker = max(overlaps, key=lambda x: x[1])[0]

    return transcript


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.diarize <audio_file>")
        print("       HF_TOKEN=xxx python -m src.diarize <audio_file>")
        sys.exit(1)

    audio_path = Path(sys.argv[1])
    if not audio_path.exists():
        print(f"Error: File not found: {audio_path}")
        sys.exit(1)

    segments, diarization = transcribe_and_diarize(audio_path)

    print(f"\nResults:")
    print(f"  Segments: {len(segments)}")
    print(f"  Speakers: {diarization.num_speakers}")
    print()

    # Show first 10 segments
    for seg in segments[:10]:
        speaker = seg.speaker or "?"
        print(f"  [{seg.start:.1f}s] {speaker}: {seg.text[:60]}...")
