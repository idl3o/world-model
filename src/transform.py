"""
Transform Stage: URL/File → Raw Assets

Handles media acquisition and initial processing.
"""

import json
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class MediaAssets:
    """Container for transformed media assets."""
    video_id: str
    title: str
    description: str
    duration: float  # seconds (0 for ongoing live streams)
    channel: str
    upload_date: str

    audio_path: Optional[Path] = None
    video_path: Optional[Path] = None
    thumbnail_path: Optional[Path] = None
    subtitles_path: Optional[Path] = None
    chapters: list = None

    # Stream-specific fields
    is_live_stream: bool = False
    stream_type: Optional[str] = None  # "hls", "rtmp", "youtube_live"
    stream_ended: bool = False
    capture_duration: float = 0.0  # Total captured duration for streams

    def __post_init__(self):
        if self.chapters is None:
            self.chapters = []


def extract_video_id(url: str) -> str:
    """Extract video ID from various URL formats."""
    import re
    patterns = [
        r'(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})',
        r'([a-zA-Z0-9_-]{11})'  # Fallback: raw ID
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract video ID from: {url}")


def transform(url: str, output_dir: Path) -> MediaAssets:
    """
    Transform a URL into raw media assets.

    Args:
        url: YouTube URL or video ID
        output_dir: Base directory for outputs

    Returns:
        MediaAssets with paths to downloaded content
    """
    video_id = extract_video_id(url)
    work_dir = output_dir / video_id
    work_dir.mkdir(parents=True, exist_ok=True)

    # Fetch metadata first
    metadata = _fetch_metadata(url)

    # Download audio (and optionally video)
    audio_path = _download_audio(url, work_dir)

    # Download subtitles if available
    subtitles_path = _download_subtitles(url, work_dir)

    # Download thumbnail
    thumbnail_path = _download_thumbnail(url, work_dir)

    assets = MediaAssets(
        video_id=video_id,
        title=metadata.get('title', 'Unknown'),
        description=metadata.get('description', ''),
        duration=metadata.get('duration', 0),
        channel=metadata.get('channel', metadata.get('uploader', 'Unknown')),
        upload_date=metadata.get('upload_date', ''),
        chapters=metadata.get('chapters', []),
        audio_path=audio_path,
        subtitles_path=subtitles_path,
        thumbnail_path=thumbnail_path,
    )

    # Save metadata
    _save_metadata(assets, work_dir)

    return assets


def _fetch_metadata(url: str) -> dict:
    """Fetch video metadata without downloading."""
    result = subprocess.run(
        ['yt-dlp', '--dump-json', '--no-download', url],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to fetch metadata: {result.stderr}")
    return json.loads(result.stdout)


def _download_audio(url: str, work_dir: Path) -> Path:
    """Download audio track."""
    output_template = str(work_dir / 'audio.%(ext)s')
    result = subprocess.run([
        'yt-dlp',
        '-x',  # Extract audio
        '--audio-format', 'mp3',
        '--audio-quality', '0',  # Best quality
        '-o', output_template,
        url
    ], capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Failed to download audio: {result.stderr}")

    return work_dir / 'audio.mp3'


def _download_subtitles(url: str, work_dir: Path) -> Optional[Path]:
    """Download subtitles if available."""
    output_template = str(work_dir / 'subtitles')
    result = subprocess.run([
        'yt-dlp',
        '--write-auto-sub',
        '--sub-lang', 'en',
        '--sub-format', 'json3',
        '--skip-download',
        '-o', output_template,
        url
    ], capture_output=True, text=True)

    # Look for downloaded subtitle file
    for ext in ['.en.json3', '.json3', '.en.vtt', '.vtt']:
        sub_path = work_dir / f'subtitles{ext}'
        if sub_path.exists():
            return sub_path

    return None


def _download_thumbnail(url: str, work_dir: Path) -> Optional[Path]:
    """Download video thumbnail."""
    output_template = str(work_dir / 'thumbnail.%(ext)s')
    result = subprocess.run([
        'yt-dlp',
        '--write-thumbnail',
        '--skip-download',
        '--convert-thumbnails', 'jpg',
        '-o', output_template,
        url
    ], capture_output=True, text=True)

    thumb_path = work_dir / 'thumbnail.jpg'
    return thumb_path if thumb_path.exists() else None


def _save_metadata(assets: MediaAssets, work_dir: Path):
    """Save assets metadata to JSON."""
    meta_path = work_dir / 'transform_meta.json'

    # Convert to dict, handling Path objects
    data = asdict(assets)
    for key, value in data.items():
        if isinstance(value, Path):
            data[key] = str(value)

    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python transform.py <youtube_url>")
        sys.exit(1)

    url = sys.argv[1]
    output_dir = Path(__file__).parent.parent / 'output'

    print(f"Transforming: {url}")
    assets = transform(url, output_dir)
    print(f"Assets saved to: {output_dir / assets.video_id}")
    print(f"  Audio: {assets.audio_path}")
    print(f"  Duration: {assets.duration}s")
