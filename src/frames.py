"""
Keyframe Extraction: Visual perception at semantic trigger points.

Extracts video frames at moments identified by semantic analysis,
giving the AI "eyes" into the video content.
"""

import subprocess
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExtractedFrame:
    """A single extracted video frame."""
    timestamp: float
    path: Path
    trigger_type: str
    trigger_context: str
    width: int = 0
    height: int = 0
    ocr_text: Optional[str] = None  # Extracted text from frame


def download_video(url: str, work_dir: Path) -> Optional[Path]:
    """
    Download video file (not just audio) for frame extraction.

    Args:
        url: YouTube URL
        work_dir: Working directory for this video

    Returns:
        Path to downloaded video file, or None if failed
    """
    output_template = str(work_dir / 'video.%(ext)s')

    result = subprocess.run([
        'yt-dlp',
        '-f', 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best',
        '--merge-output-format', 'mp4',
        '-o', output_template,
        url
    ], capture_output=True, text=True)

    video_path = work_dir / 'video.mp4'
    if video_path.exists():
        return video_path

    # Try alternative formats
    for ext in ['webm', 'mkv', 'mp4']:
        alt_path = work_dir / f'video.{ext}'
        if alt_path.exists():
            return alt_path

    return None


def extract_frame_at_timestamp(
    video_path: Path,
    timestamp: float,
    output_path: Path
) -> bool:
    """
    Extract a single frame from video at given timestamp.

    Args:
        video_path: Path to video file
        timestamp: Time in seconds
        output_path: Where to save the frame (jpg)

    Returns:
        True if successful
    """
    result = subprocess.run([
        'ffmpeg',
        '-ss', str(timestamp),
        '-i', str(video_path),
        '-vframes', '1',
        '-q:v', '2',  # High quality JPEG
        '-y',  # Overwrite
        str(output_path)
    ], capture_output=True, text=True)

    return output_path.exists()


def extract_keyframes(
    video_path: Path,
    triggers: list,
    output_dir: Path,
    max_frames: int = 50,
    enable_ocr: bool = False
) -> list[ExtractedFrame]:
    """
    Extract frames at semantic trigger points.

    Args:
        video_path: Path to video file
        triggers: List of SemanticTrigger objects
        output_dir: Directory to save frames
        max_frames: Maximum number of frames to extract
        enable_ocr: Whether to run OCR on extracted frames

    Returns:
        List of ExtractedFrame objects
    """
    frames_dir = output_dir / 'frames'
    frames_dir.mkdir(exist_ok=True)

    extracted = []

    # Limit to max_frames, prioritizing by confidence if available
    sorted_triggers = sorted(
        triggers[:max_frames * 2],  # Take more than needed, then filter
        key=lambda t: getattr(t, 'confidence', 0.5),
        reverse=True
    )[:max_frames]

    # Re-sort by timestamp for sequential extraction
    sorted_triggers = sorted(sorted_triggers, key=lambda t: t.timestamp)

    for i, trigger in enumerate(sorted_triggers):
        frame_name = f'frame_{i:04d}_{int(trigger.timestamp)}s.jpg'
        frame_path = frames_dir / frame_name

        success = extract_frame_at_timestamp(
            video_path,
            trigger.timestamp,
            frame_path
        )

        if success:
            ocr_text = None
            if enable_ocr:
                ocr_text = _run_ocr_on_frame(frame_path)

            extracted.append(ExtractedFrame(
                timestamp=trigger.timestamp,
                path=frame_path,
                trigger_type=trigger.trigger_type,
                trigger_context=trigger.context[:100],
                ocr_text=ocr_text
            ))

    return extracted


def _run_ocr_on_frame(frame_path: Path) -> Optional[str]:
    """
    Run OCR on a single frame.

    Args:
        frame_path: Path to frame image

    Returns:
        Extracted text, or None if no text found
    """
    try:
        from .ocr import extract_text
        result = extract_text(frame_path, min_confidence=0.4)
        if result.text and result.confidence > 0.3:
            return result.text
    except Exception as e:
        print(f"  Warning: OCR failed for {frame_path.name}: {e}")
    return None


def extract_keyframes_for_video(
    video_id: str,
    url: str,
    output_dir: Path,
    max_frames: int = 30,
    enable_ocr: bool = False
) -> list[ExtractedFrame]:
    """
    Full keyframe extraction pipeline for a video.

    Downloads video if needed, runs semantic analysis, extracts frames.

    Args:
        video_id: Video ID
        url: Original URL (for downloading)
        output_dir: Base output directory
        max_frames: Maximum frames to extract
        enable_ocr: Whether to run OCR on extracted frames

    Returns:
        List of ExtractedFrame objects
    """
    from .semantic import analyze_content
    from .extract import load_transcript

    work_dir = output_dir / video_id

    # Check if video exists, download if not
    video_path = work_dir / 'video.mp4'
    if not video_path.exists():
        print(f"  Downloading video for frame extraction...")
        video_path = download_video(url, work_dir)
        if not video_path:
            print(f"  Warning: Could not download video")
            return []

    # Load transcript and run semantic analysis
    transcript = load_transcript(work_dir)
    if not transcript:
        print(f"  Warning: No transcript found")
        return []

    analysis = analyze_content(transcript)
    triggers = analysis['best_result']['all_triggers']

    if not triggers:
        print(f"  Warning: No semantic triggers found")
        return []

    # Extract frames
    ocr_msg = " with OCR" if enable_ocr else ""
    print(f"  Extracting {min(len(triggers), max_frames)} keyframes{ocr_msg}...")
    frames = extract_keyframes(video_path, triggers, work_dir, max_frames, enable_ocr)

    # Save frame manifest
    _save_frame_manifest(frames, work_dir)

    return frames


def _save_frame_manifest(frames: list[ExtractedFrame], work_dir: Path):
    """Save extracted frames metadata to JSON."""
    frames_with_ocr = sum(1 for f in frames if f.ocr_text)

    manifest = {
        'frame_count': len(frames),
        'frames_with_ocr': frames_with_ocr,
        'frames': [
            {
                'timestamp': f.timestamp,
                'path': str(f.path.name),
                'trigger_type': f.trigger_type,
                'trigger_context': f.trigger_context,
                'ocr_text': f.ocr_text
            }
            for f in frames
        ]
    }

    manifest_path = work_dir / 'frames_manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract keyframes at semantic trigger points"
    )
    parser.add_argument('video_id', help="Video ID to process")
    parser.add_argument('url', nargs='?', help="Original URL (optional)")
    parser.add_argument(
        '--ocr',
        action='store_true',
        help="Run OCR on extracted frames to extract visual text"
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=30,
        help="Maximum number of frames to extract (default: 30)"
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path(__file__).parent.parent / 'output',
        help="Output directory"
    )

    args = parser.parse_args()

    video_id = args.video_id
    url = args.url or f"https://youtube.com/watch?v={video_id}"
    output_dir = args.output

    ocr_msg = " with OCR" if args.ocr else ""
    print(f"Extracting keyframes{ocr_msg} for {video_id}")

    frames = extract_keyframes_for_video(
        video_id,
        url,
        output_dir,
        max_frames=args.max_frames,
        enable_ocr=args.ocr
    )

    print(f"Extracted {len(frames)} frames")
    frames_with_ocr = sum(1 for f in frames if f.ocr_text)
    if args.ocr:
        print(f"  {frames_with_ocr} frames with detected text")

    for f in frames[:5]:
        ts = format_timestamp(f.timestamp)
        ocr_preview = f" | OCR: {f.ocr_text[:40]}..." if f.ocr_text else ""
        print(f"  [{ts}] {f.trigger_type}: {f.trigger_context[:40]}...{ocr_preview}")
