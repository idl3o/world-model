# World Model

> Giving digital intelligence eyes, ears, and taste for media.

## Vision

Build a **transformation-extraction-digestion** pipeline that converts rich media (video, podcasts, streams) into native formats that AI systems can perceive and reason about.

The goal is not mere transcription—it's **sense-making**. We want to preserve the semantic richness, temporal structure, and multimodal relationships that make media meaningful.

## Current Capabilities

### Digested Content (6 hours, 61,680 words)
| Content | Duration | Type Detected |
|---------|----------|---------------|
| Form & Time Podcast | 19 min | PODCAST |
| Consciousness Iceberg | 15 min | LECTURE |
| Fireship Python | 2 min | TUTORIAL |
| Lex Fridman x Sam Harris | 197 min | PODCAST |
| Diary of CEO x Tara Swart | 124 min | PODCAST |

### Features Working
- **Multi-format ingestion**: YouTube videos, podcasts
- **Live stream capture**: Twitch, YouTube Live, HLS, RTMP with chunked processing
- **Real-time sync**: Live transcript display, incremental entity extraction, multi-device file sync
- **Semantic analysis**: Auto-detects content type (podcast/lecture/tutorial)
- **Concept graph**: Entity extraction, co-occurrence tracking, cross-video knowledge index
- **Cross-content search**: Query across all digested media (keyword + entity-based)
- **Keyframe extraction**: Visual frames at semantic trigger points
- **Unified output**: Markdown + JSON for AI consumption

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

## Usage

```bash
# Full pipeline: Transform + Extract + Digest + Concepts
python -m src.main "https://youtube.com/watch?v=..."

# Live stream capture (auto-detected from URL)
python -m src.main "https://twitch.tv/channel"
python -m src.stream "https://twitch.tv/channel" --duration 120

# Real-time stream capture with live display
python -m src.stream "https://twitch.tv/channel" --duration 120 --live

# Cross-content search
python -m src.search "consciousness"
python -m src.search --entity "Sam Harris"    # Entity-based search
python -m src.search --topic "meditation"     # Topic with expansion

# Concept graph management
python -m src.concepts <video_id>             # Extract entities for one video
python -m src.concepts --rebuild              # Rebuild global graph
python -m src.concepts --query "Sam Harris"   # Query entity across all content

# Extract keyframes at semantic triggers
python -m src.frames <video_id> [url]

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
├── CLAUDE.md              # This file
├── requirements.txt       # Python dependencies
├── src/
│   ├── __init__.py
│   ├── main.py            # Pipeline orchestration
│   ├── transform.py       # Stage 1: URL → Assets
│   ├── extract.py         # Stage 2: Assets → Transcript + Concepts
│   ├── semantic.py        # Content-type detection & triggers
│   ├── concepts.py        # Entity extraction & knowledge graph
│   ├── digest.py          # Stage 3: Fragments → Document
│   ├── frames.py          # Keyframe extraction
│   ├── search.py          # Cross-content search (keyword + entity)
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
| `concepts.json` | Extracted entities, co-occurrences |
| `frames/` | Keyframes at semantic trigger points |
| `frames_manifest.json` | Frame metadata with trigger context |

Global index:

| File | Purpose |
|------|---------|
| `concept_graph.json` | Cross-video entity index with relationships |

Live streams additionally produce:

| File | Purpose |
|------|---------|
| `session_state.json` | Stream session metadata and chunk list |
| `chunks/` | Audio segments for incremental processing |

## Technical Stack

| Component | Tool | Status |
|-----------|------|--------|
| Download | `yt-dlp` | Working |
| Audio processing | `ffmpeg` | Working |
| Transcription | YouTube subs / Whisper | Working |
| Semantic analysis | Custom patterns | Working |
| Entity extraction | SpaCy NER (`en_core_web_md`) | Working |
| Concept graph | Custom (JSON-based) | Working |
| Frame extraction | `ffmpeg` | Working |
| Cross-content search | Keyword + Entity | Working |
| Live streaming | `yt-dlp` + `ffmpeg` chunked | Working |
| Real-time display | `rich` terminal UI | Working |
| Multi-device sync | File locking + versioning | Working |
| Incremental processing | Per-chunk extraction | Working |
| Speaker diarization | - | Planned |

## Design Principles

1. **Semantic over mechanical**: Sample when meaningful, not at fixed intervals
2. **Preserve temporality**: Time is structure—keep timestamps as first-class data
3. **Layered output**: JSON for precision, Markdown for reasoning
4. **Content-aware**: Different patterns for different content types
5. **Graceful degradation**: Missing components shouldn't break the pipeline

## Next Directions

- [ ] Speaker diarization (who said what when)
- [x] Real-time/streaming processing
- [ ] OCR on keyframes (extract text from slides/code)
- [x] Entity extraction & concept graph
- [ ] Embedding-based semantic search (upgrade from co-occurrence)
- [ ] Topic clustering across content

---

*This document serves as both specification and context for AI-assisted development.*
