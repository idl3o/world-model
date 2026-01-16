# World Model

> Giving digital intelligence eyes, ears, and taste for media.

**[Read the Whitepaper](WHITEPAPER.md)** — Technical details on architecture, design philosophy, and empirical validation.

## Vision

Build a **transformation-extraction-digestion** pipeline that converts rich media (video, podcasts, streams, local files) into native formats that AI systems can perceive and reason about.

The goal is not mere transcription—it's **sense-making**. We want to preserve the semantic richness, temporal structure, and multimodal relationships that make media meaningful.

## Current Capabilities

### Digested Content (6.2 hours, 65,000+ words)
| Content | Duration | Type Detected | Source |
|---------|----------|---------------|--------|
| Form & Time Podcast | 19 min | PODCAST | YouTube |
| Consciousness Iceberg | 15 min | LECTURE | YouTube |
| Fireship Python | 2 min | TUTORIAL | YouTube |
| Lex Fridman x Sam Harris | 197 min | PODCAST | YouTube |
| Diary of CEO x Tara Swart | 124 min | PODCAST | YouTube |
| World Model Whitepaper Review | 13 min | PODCAST | NotebookLM (local) |

### Features Working
- **Multi-format ingestion**: YouTube videos, podcasts, local audio/video files (MP3, M4A, WAV, MP4, etc.)
- **Live stream capture**: Twitch, YouTube Live, HLS, RTMP with chunked processing
- **Real-time sync**: Live transcript display, incremental entity extraction, multi-device file sync
- **Semantic analysis**: Auto-detects content type (podcast/lecture/tutorial)
- **Speaker diarization**: Who said what when (via WhisperX + pyannote)
- **Topical proximity index**: Entity extraction, temporal co-occurrence, cross-video knowledge index
- **Cross-content search**: Query across all digested media (keyword, entity, semantic)
- **Topic clustering**: Discover themes across all content automatically
- **Keyframe extraction**: Visual frames at semantic trigger points with OCR
- **Unified output**: Markdown + JSON for AI consumption

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

## Usage

```bash
# Full pipeline: Transform + Extract + Digest + Concepts
python -m src.main "https://youtube.com/watch?v=..."

# Local audio/video files (auto-detected)
python -m src.main "/path/to/podcast.mp3"
python -m src.main "/path/to/recording.m4a" --title "Custom Title"
python -m src.main "/path/to/video.mp4" --diarize --num-speakers 2

# With speaker diarization (who said what)
python -m src.main "https://youtube.com/watch?v=..." --diarize
python -m src.main "https://youtube.com/watch?v=..." --diarize --num-speakers 2

# Live stream capture (auto-detected from URL)
python -m src.main "https://twitch.tv/channel"
python -m src.stream "https://twitch.tv/channel" --duration 120

# Real-time stream capture with live display
python -m src.stream "https://twitch.tv/channel" --duration 120 --live

# Cross-content search
python -m src.search "consciousness"
python -m src.search --semantic "meaning of life"  # Embedding-based semantic search
python -m src.search --entity "Sam Harris"         # Entity-based search
python -m src.search --topic "meditation"          # Topic with expansion

# Topic clustering
python -m src.cluster                    # Discover topics across all content
python -m src.cluster --n-clusters 15    # Custom number of topics
python -m src.cluster --query "free will" # Find cluster for a query

# Concept graph management
python -m src.concepts <video_id>             # Extract entities for one video
python -m src.concepts --rebuild              # Rebuild global graph
python -m src.concepts --query "Sam Harris"   # Query entity across all content

# Extract keyframes at semantic triggers
python -m src.frames <video_id> [url]

# Extract keyframes with OCR (reads text from slides/code)
python -m src.frames <video_id> --ocr

# Check dependencies
python src/utils.py
```

## Semantic Analysis

The semantic engine auto-detects content type and extracts meaningful moments:

### Content Types
| Type | Trigger Patterns |
|------|------------------|
| **PODCAST** | Topic shifts ("tell me about"), insights ("I would say"), emphasis ("it's hard") |
| **LECTURE** | Definitions ("X is defined as"), structure ("the first layer"), concepts |
| **TUTORIAL** | Technical definitions, code demos ("create a variable"), feature lists |

### What Gets Detected
- Topic transitions and conversation flow
- Key insight moments (advice, conclusions)
- Emotional emphasis points
- Definition and explanation moments
- Code demonstration points (tutorials)

## Project Structure

```
world-model/
├── CLAUDE.md              # This file (README)
├── WHITEPAPER.md          # Technical whitepaper
├── requirements.txt       # Python dependencies
├── src/
│   ├── __init__.py
│   ├── main.py            # Pipeline orchestration
│   ├── transform.py       # Stage 1: URL/File → Assets
│   ├── extract.py         # Stage 2: Assets → Transcript + Concepts
│   ├── diarize.py         # Speaker diarization (WhisperX)
│   ├── semantic.py        # Content-type detection & triggers
│   ├── concepts.py        # Entity extraction & topical proximity
│   ├── digest.py          # Stage 3: Fragments → Document
│   ├── frames.py          # Keyframe extraction
│   ├── ocr.py             # OCR for visual text extraction
│   ├── search.py          # Cross-content search (keyword + entity + semantic)
│   ├── embeddings.py      # Text embeddings for semantic operations
│   ├── cluster.py         # Topic clustering across content
│   ├── stream.py          # Live stream capture (Twitch/YouTube/HLS/RTMP)
│   ├── realtime.py        # Event bus for real-time updates
│   ├── sync.py            # File locking for multi-device sync
│   ├── display.py         # Rich terminal UI for live streams
│   ├── incremental.py     # Incremental entity extraction
│   └── utils.py           # Dependency checks
├── output/                # Processed media
│   ├── concept_graph.json # Global cross-video entity index
│   └── {video_id}/
│       ├── manifest.json
│       ├── digest.md
│       ├── digest.partial.md  # (streaming) Live-updating digest
│       ├── transcript.json
│       ├── concepts.json  # Per-video entity data
│       ├── sync_manifest.json # Multi-device sync state
│       ├── frames/
│       │   └── frame_XXXX_XXs.jpg
│       └── frames_manifest.json
└── tests/
```

## Output Format

Each digested video produces:

| File | Purpose |
|------|---------|
| `manifest.json` | Structured metadata, stats, file paths |
| `digest.md` | AI-readable narrative with timestamps |
| `transcript.json` | Full timestamped transcript |
| `concepts.json` | Extracted entities, topical proximity |
| `embeddings.npz` | Semantic embeddings for search/clustering |
| `frames/` | Keyframes at semantic trigger points |
| `frames_manifest.json` | Frame metadata with trigger context |

Global index:

| File | Purpose |
|------|---------|
| `concept_graph.json` | Cross-video entity index with topical proximity |
| `topic_clusters.json` | Discovered topic clusters across all content |

Live streams additionally produce:

| File | Purpose |
|------|---------|
| `session_state.json` | Stream session metadata and chunk list |
| `chunks/` | Audio segments for incremental processing |

## Technical Stack

| Component | Tool | Status |
|-----------|------|--------|
| Download | `yt-dlp` | Working |
| Local file ingestion | `ffmpeg` + `ffprobe` | Working |
| Audio processing | `ffmpeg` | Working |
| Transcription | YouTube subs / Whisper | Working |
| Semantic analysis | Custom patterns (50+) | Working |
| Entity extraction | SpaCy NER (`en_core_web_md`) | Working |
| Topical proximity | Custom (JSON-based, 30s window) | Working |
| Frame extraction | `ffmpeg` | Working |
| OCR | EasyOCR | Working |
| Cross-content search | Keyword + Entity + Semantic | Working |
| Topic clustering | K-means + sentence-transformers | Working |
| Live streaming | `yt-dlp` + `ffmpeg` chunked | Working |
| Real-time display | `rich` terminal UI | Working |
| Multi-device sync | File locking + versioning | Working |
| Incremental processing | Per-chunk extraction | Working |
| Speaker diarization | WhisperX + pyannote | Working |

## Design Principles

1. **Semantic over mechanical**: Sample when meaningful, not at fixed intervals
2. **Preserve temporality**: Time is structure—keep timestamps as first-class data
3. **Layered output**: JSON for precision, Markdown for reasoning
4. **Content-aware**: Different patterns for different content types
5. **Graceful degradation**: Missing components shouldn't break the pipeline

## Next Directions

- [x] Speaker diarization (who said what when)
- [x] Real-time/streaming processing
- [x] OCR on keyframes (extract text from slides/code)
- [x] Entity extraction & topical proximity index
- [x] Embedding-based semantic search
- [x] Topic clustering across content
- [x] Local file ingestion (MP3, M4A, WAV, MP4, etc.)
- [ ] Relation extraction (typed relationships beyond proximity)
- [ ] Multi-modal fusion (audio + visual + text signals)
- [ ] Abstractive summarization at multiple granularities
- [ ] Cross-lingual support

---

*This document serves as both specification and context for AI-assisted development.*
