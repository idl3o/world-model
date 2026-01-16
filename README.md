# World Model

> Giving digital intelligence eyes, ears, and taste for media.

Transform rich media (video, podcasts, live streams, local files) into AI-native formats that preserve semantic richness, temporal structure, and multimodal relationships.

**[Read the Whitepaper](WHITEPAPER.md)** — Technical details on architecture, design philosophy, and empirical validation.

## Features

- **Multi-format ingestion** - YouTube videos, podcasts, local audio/video files (MP3, M4A, WAV, MP4), live streams
- **Real-time processing** - Live transcript display, incremental entity extraction, multi-device sync
- **Semantic analysis** - Auto-detects content type (podcast/lecture/tutorial), 50+ trigger patterns
- **Speaker diarization** - Who said what when (via WhisperX + pyannote)
- **Topical proximity index** - Entity extraction, temporal co-occurrence, cross-video concept index
- **Cross-content search** - Query by keyword, semantic similarity, entity, or topic
- **Topic clustering** - Discover themes across all content automatically
- **Keyframe extraction** - Visual frames at semantic trigger points with OCR
- **Unified output** - Markdown for reasoning, JSON for precision

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/world-model.git
cd world-model

# Install dependencies
pip install -r requirements.txt

# Download SpaCy model for entity extraction
python -m spacy download en_core_web_md
```

### Requirements

- Python 3.10+
- FFmpeg (must be in PATH)
- yt-dlp (installed via requirements.txt)

## Quick Start

```bash
# Process a YouTube video
python -m src.main "https://youtube.com/watch?v=..."

# Process a local audio file
python -m src.main "/path/to/podcast.mp3"
python -m src.main "/path/to/recording.m4a" --title "Custom Title"

# With speaker diarization
python -m src.main "https://youtube.com/watch?v=..." --diarize --num-speakers 2

# Capture a live stream with real-time display
python -m src.stream "https://twitch.tv/channel" --duration 120 --live

# Search across all processed content
python -m src.search "consciousness"
python -m src.search --semantic "meaning of life"
python -m src.search --entity "Sam Harris"
```

## Usage

### Process Video/Podcast/Local File

```bash
# YouTube
python -m src.main "https://youtube.com/watch?v=VIDEO_ID"

# Local files (auto-detected)
python -m src.main "/path/to/audio.mp3"
python -m src.main "/path/to/video.m4a" --title "Episode Title"
python -m src.main "/path/to/recording.wav" --diarize --num-speakers 2
```

This runs the full pipeline:
1. **Transform** - Downloads/processes audio, subtitles, metadata
2. **Extract** - Transcribes audio, extracts entities, generates embeddings
3. **Digest** - Generates unified markdown + JSON output

### Capture Live Streams

```bash
# Basic capture
python -m src.stream "https://twitch.tv/channel" --duration 60

# Real-time mode with live terminal UI
python -m src.stream "https://twitch.tv/channel" --duration 120 --live

# Real-time without display (headless)
python -m src.stream "https://twitch.tv/channel" --duration 120 --live --no-display
```

Supported stream types:
- Twitch (`twitch.tv/...`)
- YouTube Live (`youtube.com/watch?v=...` with live stream)
- HLS (`.m3u8` URLs)
- RTMP (`rtmp://...`)

### Search Content

```bash
# Keyword search
python -m src.search "meditation"

# Semantic search (embedding-based)
python -m src.search --semantic "meaning of life"

# Entity-based search
python -m src.search --entity "Sam Harris"

# Topic search with expansion
python -m src.search --topic "consciousness"
```

### Topic Clustering

```bash
# Discover topics across all content
python -m src.cluster

# Custom number of clusters
python -m src.cluster --n-clusters 15

# Find cluster for a query
python -m src.cluster --query "free will"
```

### Manage Concept Graph

```bash
# Extract entities for a specific video
python -m src.concepts VIDEO_ID

# Rebuild global graph from all videos
python -m src.concepts --rebuild

# Query entity across all content
python -m src.concepts --query "OpenAI"
```

### Extract Keyframes

```bash
# Extract keyframes at semantic triggers
python -m src.frames VIDEO_ID

# With OCR (reads text from slides/code)
python -m src.frames VIDEO_ID --ocr
```

## Output Structure

Each processed video produces:

```
output/
├── concept_graph.json          # Global cross-video entity index
├── topic_clusters.json         # Discovered topic clusters
└── {video_id}/
    ├── manifest.json           # Structured metadata
    ├── digest.md               # AI-readable narrative
    ├── transcript.json         # Timestamped transcript
    ├── concepts.json           # Entities and topical proximity
    ├── embeddings.npz          # Semantic embeddings
    ├── audio.mp3               # Downloaded/converted audio
    ├── thumbnail.jpg           # Video thumbnail
    ├── frames/                 # Keyframes at trigger points
    └── frames_manifest.json    # Frame metadata with OCR
```

Live streams additionally produce:
- `digest.partial.md` - Updates in real-time during capture
- `session_state.json` - Stream session metadata
- `sync_manifest.json` - Multi-device sync state
- `chunks/` - Audio segments

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           WORLD MODEL                                    │
├─────────────┬─────────────────┬──────────────────┬─────────────────────┤
│  TRANSFORM  │     EXTRACT     │     SEMANTIC     │       DIGEST        │
│             │                 │                  │                     │
│  URL/File   │  Audio → Text   │  Content-Type    │  Unified Document   │
│  → Assets   │  Video → Frames │  Detection       │  + Proximity Index  │
│             │                 │  Trigger Points  │  + Search Index     │
└─────────────┴─────────────────┴──────────────────┴─────────────────────┘
```

### Core Modules

| Module | Purpose |
|--------|---------|
| `transform.py` | URL/File → Assets (audio, subtitles, metadata) |
| `extract.py` | Assets → Transcript + Concepts |
| `semantic.py` | Content-type detection & trigger extraction |
| `concepts.py` | Entity extraction & topical proximity |
| `digest.py` | Fragments → Unified document |
| `search.py` | Cross-content search (keyword + semantic + entity) |
| `embeddings.py` | Text embeddings for semantic operations |
| `cluster.py` | Topic clustering across content |
| `frames.py` | Keyframe extraction at semantic triggers |
| `ocr.py` | Text extraction from visual frames |
| `stream.py` | Live stream capture |
| `diarize.py` | Speaker diarization (WhisperX) |

### Real-Time Modules

| Module | Purpose |
|--------|---------|
| `realtime.py` | Event bus for publish/subscribe |
| `sync.py` | File locking for multi-device sync |
| `display.py` | Rich terminal UI |
| `incremental.py` | Per-chunk entity extraction |

## Semantic Analysis

The system auto-detects content type and extracts meaningful moments:

| Content Type | Detected Patterns |
|--------------|-------------------|
| **Podcast** | Topic shifts, insights, emphasis, speaker turns |
| **Lecture** | Definitions, structure markers, concept introductions |
| **Tutorial** | Technical definitions, code demos, feature explanations |

## Tech Stack

| Component | Tool |
|-----------|------|
| Media download | yt-dlp |
| Local file processing | FFmpeg + ffprobe |
| Audio processing | FFmpeg |
| Transcription | YouTube subtitles / OpenAI Whisper |
| Speaker diarization | WhisperX + pyannote |
| Entity extraction | SpaCy NER |
| Embeddings | sentence-transformers |
| Topic clustering | scikit-learn K-means |
| OCR | EasyOCR |
| Terminal UI | Rich |
| File sync | filelock |

## Configuration

The system uses sensible defaults. Key parameters:

- **Chunk duration**: 30 seconds (for streaming)
- **Topical proximity window**: 30 seconds (deliberate semantic parameter)
- **Whisper model**: "base" (balance of speed/quality)
- **SpaCy model**: "en_core_web_md" (falls back to "en_core_web_sm")
- **Embedding model**: "all-MiniLM-L6-v2"

## License

MIT

## Contributing

Contributions welcome. Please open an issue first to discuss major changes.
