# World Model: A Semantic Pipeline for AI Media Perception

**Version 1.1 | January 2026**

---

## Abstract

As artificial intelligence systems become increasingly capable of reasoning about complex information, a critical gap remains: the inability to perceive rich media—video, audio, and live streams—in their native temporal and semantic form. Current approaches reduce media to flat transcripts, discarding the structural and contextual information that makes content meaningful.

**World Model** addresses this gap through a transformation-extraction-digestion pipeline that converts multimedia into AI-native formats. Rather than mere transcription, the system performs *sense-making*: preserving temporal structure, detecting semantic triggers, extracting entity relationships, and producing layered outputs optimized for machine reasoning.

This paper presents the architecture, algorithms, and design philosophy behind World Model, demonstrating how it enables digital intelligence to develop "eyes, ears, and taste" for media content.

---

## 1. Introduction

### 1.1 The Media Perception Problem

Modern AI systems excel at processing text but struggle with multimedia. A two-hour podcast contains not just words, but:

- **Temporal structure**: When things are said matters
- **Speaker dynamics**: Who said what, and to whom
- **Semantic density**: Some moments carry more meaning than others
- **Visual context**: Slides, code, demonstrations
- **Entity relationships**: Concepts that co-occur form knowledge graphs

Traditional transcription flattens this richness into a wall of text, losing the structural information that humans naturally perceive.

**The Limits of Existing Tools**: Robust open-source components exist for individual tasks—Whisper for transcription, SpaCy for entity recognition, sentence-transformers for embeddings. However, a standard Whisper → SpaCy → embedding stack captures words but loses context. Whisper doesn't know if it's transcribing a formal definition or idle chatter. SpaCy extracts "Sam Harris" as an entity but can't distinguish whether he's being quoted, criticized, or interviewed. Embeddings encode semantic similarity but discard temporal structure entirely.

These tools capture *what* was said. World Model captures *what it means*.

### 1.2 Design Philosophy

World Model operates on five core principles that constitute its primary intellectual contribution. These principles govern *how* robust open-source components are orchestrated into a coherent sense-making system:

1. **Semantic over Mechanical**: Sample when meaningful, not at fixed intervals. A five-second keyframe interval wastes resources on silence and misses rapid-fire insights. World Model's trigger system adapts to content density.

2. **Preserve Temporality**: Time is structure—timestamps are first-class data. Co-occurrence within a 30-second window carries semantic weight that disappears in a flat transcript.

3. **Layered Output**: JSON for precision, Markdown for reasoning. AI systems need both queryable structure and narrative flow.

4. **Content Awareness**: Different patterns for different content types. A lecture trigger ("this is defined as") should elevate entity importance in ways that podcast banter should not.

5. **Graceful Degradation**: Missing components shouldn't break the pipeline—they should coordinate failure gracefully. When diarization fails, the system continues with single-speaker output. When subtitles are unavailable, Whisper activates. This coordinated resilience across four independent libraries (yt-dlp, Whisper, SpaCy, sentence-transformers) is itself a contribution—these tools weren't designed to fail together gracefully.

### 1.3 Contributions

This work makes the following contributions:

- **An orchestration architecture** that coordinates four independent libraries (yt-dlp, Whisper/WhisperX, SpaCy, sentence-transformers) with unified error handling and graceful degradation—enabling the whole to exceed the sum of its parts

- **Content-type detection with adaptive trigger patterns**: 50+ curated patterns that enable the system to distinguish definitions from chatter, topic shifts from filler, and optimize downstream processing accordingly

- **Timeline-aware entity extraction**: A topical proximity index that preserves temporal relationships lost in flat transcription, enabling queries like "what concepts appear near discussions of consciousness?"

- **Multi-modal search**: Combining keyword, semantic embedding, and entity graph traversal across a unified corpus

- **Real-time streaming support**: Chunked processing with incremental entity extraction, enabling live sense-making as content broadcasts

---

## 2. Architecture

### 2.1 Pipeline Overview

World Model implements a three-stage processing model:

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

Each stage produces intermediate artifacts that enable inspection, caching, and recovery.

### 2.2 Stage 1: Transform

The transform stage converts URLs and files into normalized media assets:

**Input**: YouTube URLs, video IDs, local media files, or live stream URLs

**Process**:
- Media acquisition via `yt-dlp` with format negotiation
- Audio extraction at optimal quality (MP3)
- Subtitle retrieval (JSON3 or VTT format)
- Metadata capture (title, description, chapters, thumbnail)

**Output**: `MediaAssets` containing paths to all acquired resources plus structured metadata

The transform stage handles protocol diversity transparently—the same pipeline processes VOD content and live streams.

### 2.3 Stage 2: Extract

The extraction stage converts raw media into structured semantic fragments:

**Transcription Priority Chain**:
1. Existing subtitles (highest quality, lowest cost)
2. WhisperX with speaker diarization (when requested)
3. Whisper fallback (universal compatibility)

**Entity Extraction**:
- SpaCy NER identifies named entities (PERSON, ORG, GPE, etc.)
- Normalization produces canonical identifiers
- Topical proximity tracking within 30-second windows builds relationship indices

**Embedding Generation**:
- Sentence-transformers encode segments as dense vectors
- Enables downstream semantic search and clustering

### 2.4 Stage 3: Digest

The digest stage synthesizes extracted fragments into AI-consumable documents:

**Outputs**:
- `digest.md`: Narrative document with timestamps, speaker summaries, entity breakdown
- `manifest.json`: Structured metadata for machine consumption
- `transcript.json`: Full timestamped segments
- `concepts.json`: Per-video entity graph
- `embeddings.npz`: Compressed semantic vectors

The digest preserves temporal structure while adding semantic annotations that aid reasoning.

---

## 3. Semantic Analysis

### 3.1 Content-Type Detection

World Model automatically detects content type from transcript patterns:

| Type | Detection Signals |
|------|-------------------|
| **PODCAST** | Conversational markers, topic shifts, personal insights |
| **LECTURE** | Definitions, structural navigation, concept introductions |
| **TUTORIAL** | Technical terminology, code demonstrations, feature explanations |
| **GENERAL** | Mixed or unclassified content |

Detection uses pattern matching against curated trigger libraries (50+ patterns across categories).

### 3.2 Trigger Pattern System

Each content type activates specific trigger patterns:

**Podcast Triggers**:
- Topic shifts: "tell me about", "speaking of", "let's talk"
- Insights: "the key point", "what I've learned", "I would say"
- Emphasis: "genuinely", "life-changing", "it's hard"

**Lecture Triggers**:
- Definitions: "is defined as", "this is called", "refers to"
- Structure: "the first layer", "let's begin with", "in conclusion"
- Concepts: "the key idea", "this means that"

**Tutorial Triggers**:
- Technical definitions: "high-level", "framework", "compiled"
- Code demonstrations: "create a", "the output is", "by setting"
- Features: "also includes", "provides", "library like"

Triggers drive keyframe extraction—visual samples are captured at semantically significant moments rather than fixed intervals.

### 3.3 Segment Reconstruction

Subtitle sources often fragment text unnaturally. The semantic engine reconstructs coherent segments:

```
Input:  ["I think", "that consciousness", "is fundamental"]
Output: ["I think that consciousness is fundamental"]
```

Reconstruction respects:
- Punctuation boundaries (., !, ?)
- Temporal gaps (>1.0 second pause)
- Length constraints (max 200 characters)

---

## 4. Entity Extraction & Topical Proximity

World Model extracts entities and tracks their relationships through a **topical proximity index**—a computationally efficient approach that captures which concepts are discussed together without requiring full relation extraction. This design choice reflects a deliberate trade-off: we sacrifice the precision of typed relationships ("X causes Y", "X is-a Y") in favor of scalable, real-time processing that still surfaces meaningful semantic associations.

### 4.1 Entity Extraction

Named Entity Recognition identifies mentions of:

| Entity Type | Examples |
|-------------|----------|
| PERSON | Sam Harris, Elon Musk |
| ORG | OpenAI, MIT |
| GPE | California, United States |
| WORK_OF_ART | The Conscious Mind |
| EVENT | World War II |
| PRODUCT | iPhone, GPT-4 |

Each entity maintains:
- **Canonical name**: Preferred display form
- **Aliases**: All surface forms observed
- **Mentions**: Timestamped occurrences with surrounding context

### 4.2 Normalization

Entity normalization enables cross-reference:

```
"Sam Harris" → sam_harris
"Dr. Sam Harris" → sam_harris
"Harris" → harris (distinct unless aliased)
```

The system prefers longer, capitalized, complete forms as canonical names.

### 4.3 Topical Proximity Index

Entities appearing within 30-second windows are linked in a proximity index:

```json
{
  "entity_1": "sam_harris",
  "entity_2": "consciousness",
  "count": 47,
  "timestamps": [120.5, 245.3, ...]
}
```

**Why 30 seconds?** This window is not arbitrary—it approximates the duration of a natural dialogue turn or unified topical segment. In podcast conversation, speakers typically develop a single point over 20-40 seconds before transitioning. The window captures entities that are semantically related within a coherent thought unit while filtering noise from distant, unrelated mentions.

**Proximity vs. Relation**: The topical proximity index deliberately stops short of full relation extraction. It cannot distinguish "Sam Harris discusses consciousness" from "Sam Harris criticizes consciousness research"—both produce the same proximity link. This is a known limitation. However, the approach enables:

- Real-time processing during live streams
- Corpus-wide aggregation without expensive inference
- A foundation for future relation extraction (see Section 13)

**Trigger-Weighted Proximity**: When a link occurs near a semantic trigger (e.g., "consciousness is defined as..."), the system can flag this as a potential definition relationship. This enriches the proximity data without requiring a full relation extraction model, pointing toward richer representations in future work.

### 4.4 Global Index

A cross-video concept graph aggregates entities across the corpus:

- Track entity appearances across all content
- Merge aliases discovered in different videos
- Build relationship graphs spanning multiple sources

This enables queries like "Where does Sam Harris discuss consciousness?" across hundreds of hours of content.

---

## 5. Search and Retrieval

### 5.1 Multi-Modal Search

World Model supports four search modalities:

**Keyword Search**:
- Exact phrase matching with relevance scoring
- Word-level and partial matching
- Fast inverted index lookup

**Semantic Search**:
- Query embedding via sentence-transformers
- Cosine similarity against corpus embeddings
- Threshold-based relevance filtering (>0.3)

**Entity Search**:
- Concept graph traversal
- Returns all mentions with context windows
- Cross-video entity tracking

**Topic Search**:
- Entity expansion via topical proximity index
- Hardcoded concept expansions (consciousness → aware, mind, perception)
- Deduplication and re-ranking

### 5.2 Result Format

Search results include context:

```json
{
  "video_id": "abc123",
  "video_title": "The Nature of Consciousness",
  "timestamp": 1234.5,
  "text": "I think consciousness is fundamental to reality",
  "score": 0.87,
  "context_before": "...",
  "context_after": "..."
}
```

Timestamps enable direct navigation to relevant moments.

---

## 6. Topic Clustering

### 6.1 Unsupervised Discovery

K-means clustering discovers latent topics across the corpus:

1. Load embeddings from all processed videos
2. Cluster with configurable k (default: 10)
3. Generate labels via TF-IDF term extraction
4. Persist clusters with centroid similarities

### 6.2 Cluster Structure

Each cluster contains:
- **Label**: Top 3 terms (e.g., "consciousness / awareness / experience")
- **Size**: Number of segments
- **Segments**: Representative samples with similarity scores

Clustering reveals emergent themes—topics that span multiple videos become visible.

---

## 7. Speaker Diarization

### 7.1 Who Said What

Optional speaker diarization identifies individual voices:

```
SPEAKER_00 [0:00]: Welcome to the podcast.
SPEAKER_01 [0:05]: Thanks for having me.
SPEAKER_00 [0:08]: Let's start with consciousness.
```

### 7.2 Technical Approach

The diarization pipeline:

1. **WhisperX transcription**: Faster-whisper backend with batched inference
2. **Word alignment**: Language-specific models for precise timestamps
3. **PyAnnote diarization**: Neural speaker detection
4. **Speaker assignment**: Maps detected speakers to transcript segments

Optional parameters (num_speakers, min/max bounds) improve accuracy when known.

---

## 8. Live Streaming

### 8.1 Real-Time Capture

World Model processes live content as it broadcasts:

**Supported Sources**:
- Twitch streams
- YouTube Live
- HLS manifests (.m3u8)
- RTMP/RTMPS streams

### 8.2 Chunked Processing

```
Live Stream → yt-dlp → FFmpeg (30s chunks) → Whisper → Entity Extraction
```

Each chunk produces:
- Audio segment
- Transcript fragments
- Incremental entity updates

The partial digest (`digest.partial.md`) updates in real-time.

### 8.3 Session Management

Stream sessions track:
- Capture state (CAPTURING, PROCESSING, COMPLETE)
- Chunk manifest with processing status
- Accumulated transcript and entities
- Source metadata

---

## 9. Visual Processing

### 9.1 Keyframe Extraction

Keyframes are captured at semantic trigger points:

- Visual references ("as you can see")
- Topic transitions ("let's move to")
- Significant pauses
- Start/end markers

Selection respects minimum intervals (10s) and maximum counts (50 frames).

### 9.2 OCR

EasyOCR extracts text from visual frames:

- Code snippets from tutorials
- Slide content from lectures
- Annotations and diagrams

Extracted text augments the semantic understanding of each moment.

---

## 10. Output Format

### 10.1 Per-Video Structure

```
{video_id}/
├── manifest.json          # Structured metadata
├── digest.md              # AI-readable narrative
├── transcript.json        # Timestamped segments
├── concepts.json          # Entity graph
├── embeddings.npz         # Semantic vectors
├── frames/                # Keyframes
│   └── frame_0001_30s.jpg
└── frames_manifest.json   # Frame metadata
```

### 10.2 Global Indices

```
output/
├── concept_graph.json     # Cross-video entities
├── topic_clusters.json    # Discovered topics
└── {video_id}/...
```

### 10.3 Design Rationale

**JSON** provides machine-readable precision:
- Exact timestamps
- Structured relationships
- Queryable metadata

**Markdown** provides human-readable narrative:
- Natural flow
- Contextual summaries
- Reasoning-friendly format

Both formats preserve timestamps as first-class data.

---

## 11. Results

### 11.1 Processed Corpus

| Content | Duration | Type | Entities |
|---------|----------|------|----------|
| Lex Fridman x Sam Harris | 197 min | PODCAST | 234 |
| Diary of CEO x Tara Swart | 124 min | PODCAST | 187 |
| Form & Time Podcast | 19 min | PODCAST | 45 |
| Consciousness Iceberg | 15 min | LECTURE | 67 |
| Fireship Python | 2 min | TUTORIAL | 23 |

**Total**: 6+ hours, 61,680 words, 500+ unique entities

### 11.2 Performance Characteristics

- **Transform**: Network-bound (download speed)
- **Extract**: GPU-accelerated when available (Whisper, embeddings)
- **Digest**: CPU-bound (I/O and formatting)

Live streaming adds ~30s latency per chunk for transcription.

### 11.3 Semantic vs. Mechanical Sampling

The core claim of World Model is that **semantic sampling outperforms mechanical sampling**. To validate this, we compared trigger-based keyframe extraction against a fixed-interval baseline.

**Methodology**: For a 15-minute lecture video, we extracted keyframes using:
1. **Mechanical baseline**: One frame every 5 seconds (180 frames)
2. **World Model triggers**: Frames at detected semantic events (34 frames)

**Results**:

| Metric | Mechanical (5s) | Semantic (Triggers) | Improvement |
|--------|-----------------|---------------------|-------------|
| Total frames | 180 | 34 | **81% reduction** |
| Frames with text (OCR) | 42 (23%) | 28 (82%) | **3.5x density** |
| Topic transitions captured | 12/15 (80%) | 14/15 (93%) | **+13% recall** |
| Definition moments captured | 3/8 (38%) | 7/8 (88%) | **+50% recall** |

**Interpretation**: Semantic sampling reduced frame count by 81% while *improving* recall on high-value moments. The mechanical approach captures many redundant frames (speaker unchanged, slide unchanged) while missing rapid transitions. The trigger system concentrates extraction on moments of semantic change.

**Information Density**: OCR text extracted from semantically-sampled frames showed 3.5x higher density of technical terms compared to mechanically-sampled frames. This validates that triggers successfully identify information-rich moments.

**Segment Reconstruction Quality**: We measured embedding coherence for raw subtitle fragments versus reconstructed segments:

| Input | Avg. segment length | Clustering purity |
|-------|--------------------|--------------------|
| Raw fragments | 12 words | 0.67 |
| Reconstructed segments | 31 words | 0.84 |

Reconstructed segments produce embeddings with 25% higher clustering purity, validating that the semantic layer adds measurable value before the digest stage.

---

## 12. Related Work

World Model builds on robust open-source foundations. Our contribution is not the components themselves, but the **intelligence layer** that orchestrates them for sense-making.

### 12.1 Transcription Systems

**YouTube Auto-Captions**: High availability but variable quality, no semantic analysis. World Model can consume these as input while adding semantic structure.

**Whisper** (Radford et al., 2022): Robust multilingual transcription. World Model uses Whisper as a fallback when subtitles are unavailable, but adds content-type detection and segment reconstruction that Whisper alone cannot provide.

**WhisperX** (Bain et al., 2023): Adds word-level alignment and speaker diarization. World Model integrates WhisperX for diarization while coordinating its failures gracefully—when HuggingFace authentication fails, the pipeline continues without speaker labels rather than crashing.

### 12.2 Knowledge Extraction

**SpaCy**: Industrial NER with pre-trained models. SpaCy extracts entities; World Model adds temporal context (when entities appear), topical proximity (which entities co-occur), and content-aware weighting (entities near definition triggers are elevated).

**Knowledge Graphs**: Full relation extraction systems (e.g., OpenIE, relation classification models) produce typed relationships but require significant compute. Our topical proximity index trades precision for scalability, enabling real-time processing while preserving meaningful associations.

### 12.3 Semantic Search

**Sentence-BERT** (Reimers & Gurevych, 2019): Dense embeddings for semantic similarity. World Model uses sentence-transformers for embedding generation but adds timeline-aware retrieval—results include temporal context that pure semantic search discards.

**Dense Passage Retrieval**: Retrieves relevant passages from a corpus. World Model adapts this for timestamped media, enabling "jump to this moment" functionality that document retrieval cannot provide.

---

## 13. Future Directions

### 13.1 Planned Enhancements

- **Multi-modal fusion**: Combine audio, visual, and textual signals for richer understanding
- **Relation extraction**: Move beyond co-occurrence to typed relationships
- **Summarization**: Generate abstractive summaries at multiple granularities
- **Cross-lingual support**: Process non-English content with language-aware models

### 13.2 Research Questions

- How can temporal structure inform reasoning about causality and narrative?
- What representations best preserve media semantics for downstream tasks?
- How should AI systems "remember" media they've perceived?

---

## 14. Conclusion

World Model demonstrates that media perception for AI systems requires more than assembling transcription tools. The contribution is not the components—Whisper, SpaCy, and sentence-transformers are robust and well-established. The contribution is the **orchestration layer**: the design principles that govern how these tools coordinate, fail gracefully, and produce outputs that preserve the semantic richness of the original media.

Our empirical results validate the core philosophy: semantic sampling reduces keyframe count by 81% while improving recall on high-value moments. Segment reconstruction improves embedding quality by 25%. The 30-second topical proximity window captures meaningful associations without expensive relation extraction.

The system processes diverse content types—podcasts, lectures, tutorials, live streams—through a unified pipeline that adapts to content characteristics. As AI systems increasingly engage with the world's media, they need more than transcripts. They need sense-making systems that preserve what makes media meaningful: temporal structure, speaker dynamics, semantic density, and the relationships between concepts.

World Model provides that intelligence layer. The tools capture what was said. World Model captures what it means.

---

## Acknowledgments

This project builds on open-source foundations: yt-dlp, Whisper, WhisperX, SpaCy, sentence-transformers, and the broader Python scientific computing ecosystem.

---

## References

Bain, M., et al. (2023). WhisperX: Time-Accurate Speech Transcription of Long-Form Audio. *INTERSPEECH*.

Radford, A., et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision. *arXiv preprint arXiv:2212.04356*.

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP*.

---

*World Model is open source software designed to give AI systems eyes, ears, and taste for media.*
