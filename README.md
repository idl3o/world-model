# World Model

> Giving digital intelligence eyes, ears, and taste for media.

Transform rich media (video, podcasts, live streams) into AI-native formats that preserve semantic richness, temporal structure, and multimodal relationships.

## Features

- **Multi-format ingestion** - YouTube videos, podcasts, Twitch/YouTube Live streams
- **Real-time processing** - Live transcript display, incremental entity extraction, multi-device sync
- **Semantic analysis** - Auto-detects content type (podcast/lecture/tutorial) and extracts meaningful moments
- **Knowledge graph** - Entity extraction, co-occurrence tracking, cross-video concept index
- **Cross-content search** - Query across all digested media by keyword or entity
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

# Capture a live stream with real-time display
python -m src.stream "https://twitch.tv/channel" --duration 120 --live

# Search across all processed content
python -m src.search "consciousness"
python -m src.search --entity "Sam Harris"
```

## Usage

### Process Video/Podcast

```bash
python -m src.main "https://youtube.com/watch?v=VIDEO_ID"
```

This runs the full pipeline:
1. **Transform** - Downloads audio, subtitles, thumbnail
2. **Extract** - Transcribes audio, extracts entities
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

# Entity-based search
python -m src.search --entity "Sam Harris"

# Topic search with expansion
python -m src.search --topic "consciousness"
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

## Output Structure

Each processed video produces:

```
output/
├── concept_graph.json          # Global cross-video entity index
└── {video_id}/
    ├── manifest.json           # Structured metadata
    ├── digest.md               # AI-readable narrative
    ├── transcript.json         # Timestamped transcript
    ├── concepts.json           # Entities and co-occurrences
    ├── audio.mp3               # Downloaded audio
    └── thumbnail.jpg           # Video thumbnail
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
│  → Assets   │  Video → Frames │  Detection       │  + Keyframes        │
│             │                 │  Trigger Points  │  + Search Index     │
└─────────────┴─────────────────┴──────────────────┴─────────────────────┘
```

### Core Modules

| Module | Purpose |
|--------|---------|
| `transform.py` | URL → Assets (audio, subtitles, metadata) |
| `extract.py` | Assets → Transcript + Concepts |
| `semantic.py` | Content-type detection & trigger extraction |
| `concepts.py` | Entity extraction & knowledge graph |
| `digest.py` | Fragments → Unified document |
| `search.py` | Cross-content search |
| `stream.py` | Live stream capture |

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
| Audio processing | FFmpeg |
| Transcription | YouTube subtitles / OpenAI Whisper |
| Entity extraction | SpaCy NER |
| Terminal UI | Rich |
| File sync | filelock |

## Configuration

The system uses sensible defaults. Key parameters:

- **Chunk duration**: 30 seconds (for streaming)
- **Co-occurrence window**: 30 seconds
- **Whisper model**: "base" (balance of speed/quality)
- **SpaCy model**: "en_core_web_md" (falls back to "en_core_web_sm")

## License

MIT

## Contributing

Contributions welcome. Please open an issue first to discuss major changes.
