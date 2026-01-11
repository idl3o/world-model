"""
Stream Module: Live Stream Ingestion

Captures live streams (Twitch, YouTube Live, HLS, RTMP) and processes them
into chunks for transcription.

Usage:
    python -m src.stream "https://twitch.tv/channel" --duration 60
    python -m src.stream "https://example.com/stream.m3u8" --duration 120
"""

import re
import subprocess
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum


class StreamType(Enum):
    """Supported stream protocols."""
    HLS = "hls"
    RTMP = "rtmp"
    YOUTUBE_LIVE = "youtube_live"
    TWITCH = "twitch"


class SessionStatus(Enum):
    """Stream session states."""
    INITIALIZING = "initializing"
    CAPTURING = "capturing"
    PROCESSING = "processing"
    PAUSED = "paused"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class StreamSource:
    """Represents a live stream source."""
    url: str
    stream_type: StreamType
    stream_id: str
    title: str
    start_time: datetime = field(default_factory=datetime.now)
    channel: Optional[str] = None
    description: Optional[str] = None


@dataclass
class StreamChunk:
    """A captured segment of the stream."""
    chunk_id: int
    start_offset: float  # Seconds from stream start
    duration: float
    audio_path: Path
    is_processed: bool = False
    transcript_segments: list = field(default_factory=list)


@dataclass
class StreamSession:
    """Manages state for a stream capture session."""
    source: StreamSource
    work_dir: Path
    status: SessionStatus = SessionStatus.INITIALIZING
    chunks: list[StreamChunk] = field(default_factory=list)
    full_transcript: list = field(default_factory=list)
    total_duration: float = 0.0
    capture_start: Optional[datetime] = None


def detect_stream_type(url: str) -> Optional[StreamType]:
    """
    Detect if URL is a live stream and what type.

    Args:
        url: Media URL to check

    Returns:
        StreamType if stream detected, None otherwise
    """
    url_lower = url.lower()

    # HLS patterns
    if '.m3u8' in url_lower or '/hls/' in url_lower:
        return StreamType.HLS

    # RTMP patterns
    if url_lower.startswith('rtmp://') or url_lower.startswith('rtmps://'):
        return StreamType.RTMP

    # Twitch - always treat as live stream
    if 'twitch.tv/' in url_lower:
        return StreamType.TWITCH

    # YouTube Live patterns
    youtube_patterns = [
        r'youtube\.com/live/',
        r'youtu\.be/live/',
    ]
    for pattern in youtube_patterns:
        if re.search(pattern, url_lower):
            return StreamType.YOUTUBE_LIVE

    # Check if YouTube URL is a live stream via yt-dlp
    if 'youtube.com' in url_lower or 'youtu.be' in url_lower:
        if _is_youtube_live(url):
            return StreamType.YOUTUBE_LIVE

    return None


def _is_youtube_live(url: str) -> bool:
    """Check if YouTube URL is a live stream."""
    try:
        result = subprocess.run(
            ['yt-dlp', '--dump-json', '--no-download', url],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            return data.get('is_live', False)
    except Exception:
        pass
    return False


def generate_stream_id(url: str) -> str:
    """Generate unique ID for a stream session."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    return f"LIVE_{timestamp}_{url_hash}"


def get_stream_url(source: StreamSource) -> str:
    """
    Get the actual stream URL for FFmpeg.

    For YouTube Live and Twitch, extracts the HLS URL via yt-dlp.
    """
    if source.stream_type in (StreamType.YOUTUBE_LIVE, StreamType.TWITCH):
        result = subprocess.run(
            ['yt-dlp', '-g', '-f', 'bestaudio', source.url],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            return result.stdout.strip()
        raise RuntimeError(f"Failed to get stream URL: {result.stderr}")

    return source.url


def fetch_stream_metadata(url: str, stream_type: StreamType) -> dict:
    """Fetch available metadata for a stream."""
    metadata = {
        'title': 'Live Stream',
        'channel': None,
        'description': None,
    }

    if stream_type in (StreamType.YOUTUBE_LIVE, StreamType.TWITCH):
        try:
            result = subprocess.run(
                ['yt-dlp', '--dump-json', '--no-download', url],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                metadata['title'] = data.get('title', 'Live Stream')
                metadata['channel'] = data.get('channel', data.get('uploader'))
                metadata['description'] = data.get('description', '')
        except Exception:
            pass

    return metadata


def create_session(url: str, output_dir: Path) -> StreamSession:
    """
    Create a new stream capture session.

    Args:
        url: Stream URL
        output_dir: Base output directory

    Returns:
        Initialized StreamSession
    """
    stream_type = detect_stream_type(url)
    if stream_type is None:
        raise ValueError(f"URL does not appear to be a live stream: {url}")

    stream_id = generate_stream_id(url)
    metadata = fetch_stream_metadata(url, stream_type)

    source = StreamSource(
        url=url,
        stream_type=stream_type,
        stream_id=stream_id,
        title=metadata['title'],
        channel=metadata.get('channel'),
        description=metadata.get('description'),
    )

    work_dir = output_dir / stream_id
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / 'chunks').mkdir(exist_ok=True)

    session = StreamSession(
        source=source,
        work_dir=work_dir,
    )

    return session


def capture_stream(
    session: StreamSession,
    chunk_duration: int = 30,
    max_duration: Optional[int] = None,
    on_chunk_ready: Optional[callable] = None
) -> None:
    """
    Capture stream using yt-dlp piped to ffmpeg (blocking).

    Args:
        session: StreamSession to capture to
        chunk_duration: Duration of each chunk in seconds
        max_duration: Max total capture duration (required for testing, None = manual stop)
        on_chunk_ready: Callback for each chunk after capture completes
    """
    session.status = SessionStatus.CAPTURING
    session.capture_start = datetime.now()

    chunks_dir = session.work_dir / 'chunks'
    chunks_dir.mkdir(exist_ok=True)
    chunk_pattern = str(chunks_dir / 'chunk_%04d.mp3')

    # Build ffmpeg options
    duration_opt = f'-t {max_duration}' if max_duration else ''

    # Pipe yt-dlp to ffmpeg for segmented capture
    cmd = (
        f'yt-dlp -f bestaudio -o - "{session.source.url}" | '
        f'ffmpeg -i pipe:0 -f segment -segment_time {chunk_duration} '
        f'-ar 16000 -ac 1 -acodec libmp3lame {duration_opt} "{chunk_pattern}"'
    )

    print(f"Capturing: {session.source.title}", flush=True)
    print(f"  Stream ID: {session.source.stream_id}", flush=True)
    print(f"  Chunk duration: {chunk_duration}s", flush=True)
    if max_duration:
        print(f"  Max duration: {max_duration}s", flush=True)
    print(f"  Output: {chunks_dir}", flush=True)
    print(f"  Recording...", flush=True)

    # Run capture (blocking)
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        # Wait with generous timeout
        timeout = (max_duration + 30) if max_duration else None
        proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
    except KeyboardInterrupt:
        proc.terminate()
        proc.communicate()

    # Collect chunks
    print(f"  Processing chunks...", flush=True)
    for chunk_file in sorted(chunks_dir.glob('chunk_*.mp3')):
        chunk_id = len(session.chunks)
        chunk = StreamChunk(
            chunk_id=chunk_id,
            start_offset=chunk_id * chunk_duration,
            duration=chunk_duration,
            audio_path=chunk_file,
        )
        session.chunks.append(chunk)
        session.total_duration = (chunk_id + 1) * chunk_duration
        print(f"    Chunk {chunk_id}: {chunk_file.name}", flush=True)

        if on_chunk_ready:
            try:
                on_chunk_ready(session, chunk)
            except Exception as e:
                print(f"    Warning: callback error: {e}", flush=True)

    session.status = SessionStatus.COMPLETE
    print(f"  Capture complete: {len(session.chunks)} chunks, {session.total_duration}s", flush=True)


def save_session(session: StreamSession):
    """Save session state to disk."""
    state_path = session.work_dir / 'session_state.json'

    import json
    state = {
        'source': {
            'url': session.source.url,
            'stream_type': session.source.stream_type.value,
            'stream_id': session.source.stream_id,
            'title': session.source.title,
            'channel': session.source.channel,
            'start_time': session.source.start_time.isoformat(),
        },
        'status': session.status.value,
        'chunk_count': len(session.chunks),
        'total_duration': session.total_duration,
        'capture_start': session.capture_start.isoformat() if session.capture_start else None,
        'chunks': [
            {
                'chunk_id': c.chunk_id,
                'start_offset': c.start_offset,
                'duration': c.duration,
                'audio_path': str(c.audio_path),
                'is_processed': c.is_processed,
            }
            for c in session.chunks
        ]
    }

    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)


def load_session(work_dir: Path) -> StreamSession:
    """Load session state from disk."""
    import json

    state_path = work_dir / 'session_state.json'
    with open(state_path) as f:
        state = json.load(f)

    source = StreamSource(
        url=state['source']['url'],
        stream_type=StreamType(state['source']['stream_type']),
        stream_id=state['source']['stream_id'],
        title=state['source']['title'],
        channel=state['source'].get('channel'),
        start_time=datetime.fromisoformat(state['source']['start_time']),
    )

    session = StreamSession(
        source=source,
        work_dir=work_dir,
        status=SessionStatus(state['status']),
        total_duration=state['total_duration'],
    )

    for c in state.get('chunks', []):
        session.chunks.append(StreamChunk(
            chunk_id=c['chunk_id'],
            start_offset=c['start_offset'],
            duration=c['duration'],
            audio_path=Path(c['audio_path']),
            is_processed=c.get('is_processed', False),
        ))

    return session


def finalize_session(session: StreamSession) -> Path:
    """
    Finalize a stream session by generating digest and cleaning up.

    Args:
        session: Completed StreamSession

    Returns:
        Path to digest.md
    """
    from .transform import MediaAssets
    from .extract import ExtractedContent, TranscriptSegment, _save_transcript
    from .digest import digest

    # Create MediaAssets from stream data
    assets = MediaAssets(
        video_id=session.source.stream_id,
        title=session.source.title,
        description=session.source.description or '',
        duration=session.total_duration,
        channel=session.source.channel or 'Unknown',
        upload_date=session.source.start_time.strftime('%Y%m%d'),
        is_live_stream=True,
        stream_type=session.source.stream_type.value,
        stream_ended=True,
        capture_duration=session.total_duration,
    )

    # Create ExtractedContent from accumulated transcript
    content = ExtractedContent(video_id=session.source.stream_id)
    content.transcript = session.full_transcript

    # Save the full transcript
    _save_transcript(content.transcript, session.work_dir)

    # Generate digest
    digest_path = digest(assets, content, session.work_dir)

    print(f"Digest generated: {digest_path}")
    return digest_path


def run_stream_pipeline(
    url: str,
    output_dir: Path,
    duration: int = 60,
    chunk_duration: int = 30
) -> StreamSession:
    """
    Run the full stream capture and processing pipeline.

    Args:
        url: Stream URL
        output_dir: Output directory
        duration: Capture duration in seconds
        chunk_duration: Duration of each chunk in seconds

    Returns:
        Completed StreamSession
    """
    from .extract import transcribe_chunk

    session = create_session(url, output_dir)

    def on_chunk(sess: StreamSession, chunk: StreamChunk):
        """Transcribe each chunk."""
        segments = transcribe_chunk(chunk.audio_path, chunk.start_offset)
        chunk.transcript_segments = segments
        chunk.is_processed = True
        sess.full_transcript.extend(segments)

    # Capture and process
    capture_stream(session, chunk_duration=chunk_duration, max_duration=duration, on_chunk_ready=on_chunk)

    # Save and finalize
    save_session(session)
    finalize_session(session)

    return session


# =============================================================================
# Real-Time Capture Functions
# =============================================================================

def _is_file_complete(path: Path, stable_seconds: float = 0.5) -> bool:
    """
    Check if a file is done being written.

    Waits for file size to stabilize.
    """
    import time

    try:
        size1 = path.stat().st_size
        time.sleep(stable_seconds)
        size2 = path.stat().st_size
        return size1 == size2 and size1 > 0
    except OSError:
        return False


def watch_chunks_async(
    chunks_dir: Path,
    callback: callable,
    stop_event,
    poll_interval: float = 1.0
) -> None:
    """
    Watch for new chunk files and invoke callback (runs in thread).

    Args:
        chunks_dir: Directory to watch for chunk_*.mp3 files
        callback: Function to call with each new chunk Path
        stop_event: threading.Event to signal stop
        poll_interval: How often to check for new files
    """
    import time

    seen_chunks = set()

    while not stop_event.is_set():
        try:
            current_chunks = set(chunks_dir.glob('chunk_*.mp3'))
            new_chunks = current_chunks - seen_chunks

            for chunk_path in sorted(new_chunks, key=lambda p: p.name):
                if _is_file_complete(chunk_path):
                    try:
                        callback(chunk_path)
                    except Exception as e:
                        print(f"Warning: chunk callback error: {e}")
                    seen_chunks.add(chunk_path)

            time.sleep(poll_interval)

        except Exception as e:
            print(f"Warning: watcher error: {e}")
            time.sleep(poll_interval)


def run_stream_pipeline_realtime(
    url: str,
    output_dir: Path,
    duration: int = 60,
    chunk_duration: int = 30,
    enable_display: bool = True,
    enable_concepts: bool = True
) -> 'StreamSession':
    """
    Run stream capture with real-time display and incremental processing.

    Features:
    - Live terminal display showing transcript as it arrives
    - Incremental entity extraction per-chunk
    - Progressive digest generation
    - File sync support for multi-device access

    Args:
        url: Stream URL
        output_dir: Output directory
        duration: Capture duration in seconds
        chunk_duration: Duration of each chunk
        enable_display: Show rich terminal display
        enable_concepts: Extract entities incrementally

    Returns:
        Completed StreamSession
    """
    import threading
    from .extract import transcribe_chunk_fast, get_whisper_model
    from .realtime import EventBus, StreamContext, create_stream_context, EVENT_TRANSCRIPT_UPDATE
    from .sync import FileSyncEngine

    # Create session
    session = create_session(url, output_dir)

    # Initialize components
    event_bus = EventBus()
    sync_engine = FileSyncEngine(session.work_dir)
    context = create_stream_context(
        session_id=session.source.stream_id,
        work_dir=session.work_dir,
        event_bus=event_bus
    )

    # Set up display
    display = None
    if enable_display:
        try:
            from .display import create_display
            display = create_display(use_rich=True)
            display.subscribe_to_events(event_bus)
            display.start(title=session.source.title)
        except ImportError:
            print("Note: Install 'rich' for live display: pip install rich")

    # Set up incremental extraction
    extractor = None
    if enable_concepts:
        try:
            from .incremental import IncrementalEntityExtractor, IncrementalDigestGenerator
            extractor = IncrementalEntityExtractor()
            digest_gen = IncrementalDigestGenerator(
                session.work_dir,
                title=session.source.title
            )
            digest_gen.set_metadata(
                channel=session.source.channel,
                description=session.source.description
            )
        except ImportError:
            print("Note: Incremental extraction unavailable")

    # Pre-load Whisper model
    get_whisper_model("base")

    # Chunk processing state
    processed_chunks = set()
    chunk_lock = threading.Lock()

    def process_chunk(chunk_path: Path):
        """Process a single chunk with transcription and entity extraction."""
        nonlocal processed_chunks

        with chunk_lock:
            if chunk_path in processed_chunks:
                return
            processed_chunks.add(chunk_path)

        # Parse chunk ID from filename
        chunk_name = chunk_path.stem  # "chunk_0001"
        try:
            chunk_id = int(chunk_name.split('_')[1])
        except (IndexError, ValueError):
            chunk_id = len(session.chunks)

        time_offset = chunk_id * chunk_duration

        # Transcribe
        segments = transcribe_chunk_fast(chunk_path, time_offset)

        # Create chunk object
        chunk = StreamChunk(
            chunk_id=chunk_id,
            start_offset=time_offset,
            duration=chunk_duration,
            audio_path=chunk_path,
            is_processed=True,
            transcript_segments=segments
        )

        # Update session
        with chunk_lock:
            session.chunks.append(chunk)
            session.full_transcript.extend(segments)
            session.total_duration = max(session.total_duration, time_offset + chunk_duration)

        # Update context and emit events
        context.add_transcript_segments(segments)
        context.chunks_processed = len(session.chunks)

        # Incremental entity extraction
        if extractor and segments:
            result = extractor.process_chunk(segments)
            for entity_id in result.new_entities:
                entity = extractor.entities.get(entity_id)
                if entity:
                    context.add_entity(entity_id, entity)

            # Update digest generator
            if digest_gen:
                digest_gen.update(
                    segments=segments,
                    entities=extractor.get_accumulated_entities()
                )
                digest_gen.write_partial()

        # Save sync state
        sync_engine.append_with_lock(
            'transcript.json',
            [{'start': s.start, 'end': s.end, 'text': s.text} for s in segments]
        )

        # Update display
        if display:
            display.on_chunk_processed()

    # Start capture in main thread with watcher in background
    stop_event = threading.Event()

    watcher_thread = threading.Thread(
        target=watch_chunks_async,
        args=(session.work_dir / 'chunks', process_chunk, stop_event, 1.0),
        daemon=True
    )
    watcher_thread.start()

    # Run capture (blocking)
    try:
        capture_stream(
            session,
            chunk_duration=chunk_duration,
            max_duration=duration,
            on_chunk_ready=None  # Watcher handles this now
        )
    finally:
        # Signal watcher to stop
        stop_event.set()
        watcher_thread.join(timeout=2.0)

        # Stop display
        if display:
            display.stop()

    # Process any remaining chunks
    import time
    time.sleep(1.0)  # Give time for last chunks
    for chunk_path in sorted((session.work_dir / 'chunks').glob('chunk_*.mp3')):
        if chunk_path not in processed_chunks:
            process_chunk(chunk_path)

    # Finalize
    save_session(session)

    # Generate final digest with concepts
    if extractor:
        video_data = extractor.flush_to_video_data(session.source.stream_id)
        from .concepts import save_video_concepts, update_global_graph
        save_video_concepts(video_data, session.work_dir)
        update_global_graph(video_data, session.source.title, output_dir)

        if digest_gen:
            digest_gen.update(
                entities=video_data.entities,
                cooccurrences=video_data.cooccurrences
            )
            digest_gen.finalize()
    else:
        finalize_session(session)

    # Save final sync manifest
    sync_engine.save_sync_manifest()
    sync_engine.release_all_locks()

    return session


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Capture and process live streams"
    )
    parser.add_argument(
        'url',
        nargs='?',
        help="Stream URL (Twitch, YouTube Live, HLS, RTMP)"
    )
    parser.add_argument(
        '--duration', '-d',
        type=int,
        default=60,
        help="Capture duration in seconds (default: 60)"
    )
    parser.add_argument(
        '--chunk-duration',
        type=int,
        default=30,
        help="Chunk duration in seconds (default: 30)"
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path(__file__).parent.parent / 'output',
        help="Output directory"
    )
    parser.add_argument(
        '--detect-only',
        action='store_true',
        help="Just detect stream type, don't capture"
    )
    parser.add_argument(
        '--live',
        action='store_true',
        help="Enable real-time display and incremental processing"
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help="Disable rich terminal display (with --live)"
    )
    parser.add_argument(
        '--no-concepts',
        action='store_true',
        help="Disable entity extraction (faster processing)"
    )

    args = parser.parse_args()

    if not args.url:
        parser.print_help()
        return

    if args.detect_only:
        stream_type = detect_stream_type(args.url)
        if stream_type:
            print(f"Detected: {stream_type.value}")
        else:
            print("Not detected as a live stream")
        return

    print("World Model - Stream Capture")
    print("=" * 40)

    if args.live:
        session = run_stream_pipeline_realtime(
            args.url,
            args.output,
            duration=args.duration,
            chunk_duration=args.chunk_duration,
            enable_display=not args.no_display,
            enable_concepts=not args.no_concepts
        )
    else:
        session = run_stream_pipeline(
            args.url,
            args.output,
            duration=args.duration,
            chunk_duration=args.chunk_duration
        )

    print()
    print("=" * 40)
    print(f"Complete: {session.source.stream_id}")
    print(f"  Output: {session.work_dir}")
    print(f"  Chunks: {len(session.chunks)}")
    print(f"  Duration: {session.total_duration:.0f}s")


if __name__ == '__main__':
    main()
