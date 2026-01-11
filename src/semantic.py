"""
Semantic Sampling: Intelligent keyframe and segment extraction.

Instead of fixed-interval sampling, we extract frames and segments
based on semantic triggers in the content.
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class ContentType(Enum):
    """Type of content being analyzed."""
    TUTORIAL = "tutorial"       # How-to, educational, visual-heavy
    PODCAST = "podcast"         # Conversational, interview-style
    LECTURE = "lecture"         # Presentation, structured talk
    GENERAL = "general"         # Unknown/mixed


@dataclass
class MergedSegment:
    """A merged segment combining fragmented subtitles."""
    start: float
    end: float
    text: str
    original_segments: list = field(default_factory=list)


@dataclass
class SemanticTrigger:
    """A detected trigger for keyframe capture."""
    timestamp: float
    trigger_type: str
    confidence: float
    context: str


# =============================================================================
# SEGMENT MERGING - Combine fragmented subtitles into sentences
# =============================================================================

def merge_segments(transcript_segments, max_gap: float = 1.0, max_length: int = 200) -> list[MergedSegment]:
    """
    Merge fragmented subtitle segments into sentence-level chunks.

    YouTube auto-subs often split mid-sentence. This reconstructs
    natural sentence boundaries based on:
    - Punctuation (., !, ?)
    - Significant pauses (gaps > max_gap)
    - Maximum length limits

    Args:
        transcript_segments: Raw segments from extract stage
        max_gap: Maximum gap (seconds) before forcing a new segment
        max_length: Maximum characters before forcing a split

    Returns:
        List of MergedSegments with natural sentence boundaries
    """
    if not transcript_segments:
        return []

    merged = []
    current_text = []
    current_start = transcript_segments[0].start
    current_originals = []
    last_end = transcript_segments[0].start

    sentence_endings = re.compile(r'[.!?]\s*$')

    for seg in transcript_segments:
        gap = seg.start - last_end
        text_so_far = ' '.join(current_text)

        # Decide if we should start a new merged segment
        should_split = (
            gap > max_gap or                                    # Significant pause
            len(text_so_far) > max_length or                   # Too long
            sentence_endings.search(text_so_far)               # Sentence ended
        )

        if should_split and current_text:
            # Save current merged segment
            merged.append(MergedSegment(
                start=current_start,
                end=last_end,
                text=' '.join(current_text).strip(),
                original_segments=current_originals
            ))
            current_text = []
            current_start = seg.start
            current_originals = []

        current_text.append(seg.text)
        current_originals.append(seg)
        last_end = seg.end

    # Flush remaining
    if current_text:
        merged.append(MergedSegment(
            start=current_start,
            end=last_end,
            text=' '.join(current_text).strip(),
            original_segments=current_originals
        ))

    return merged


# =============================================================================
# PATTERN DEFINITIONS BY CONTENT TYPE
# =============================================================================

# Phrases that suggest visual content is being referenced
VISUAL_REFERENCE_PATTERNS = [
    r'\b(as you can see|look at|looking at)\b',
    r'\b(this (shows?|displays?|demonstrates?))\b',
    r'\b(here we (have|see))\b',
    r'\b(on the? (screen|slide|chart|graph|diagram))\b',
    r'\b(this (image|picture|photo|screenshot))\b',
    r'\b(let me show you)\b',
    r'\b(take a look)\b',
    r'\b(you\'ll notice)\b',
    r'\b(pointing (to|at))\b',
    r'\b(the (code|text) (here|above|below))\b',
]

# Phrases that suggest topic transitions (tutorials/lectures)
TOPIC_TRANSITION_PATTERNS = [
    r'\b(now let\'s (talk|move|look) (about|on|at))\b',
    r'\b(moving on to)\b',
    r'\b(the next (thing|topic|point))\b',
    r'\b(another (important|key))\b',
    r'\b(first|second|third|finally)\b',
    r'\b(in conclusion|to summarize|wrapping up)\b',
    r'\b(let\'s (start|begin) (with|by))\b',
]

# =============================================================================
# PODCAST-SPECIFIC PATTERNS
# =============================================================================

# Conversational topic shifts in podcasts/interviews
PODCAST_TOPIC_PATTERNS = [
    r'\b(so (tell me|what|how)|can you tell me)\b',
    r'\b(that\'s (interesting|fascinating|great|amazing))\b',
    r'\b(speaking of|talking about)\b',
    r'\b(i (have to|want to|need to) ask)\b',
    r'\b(let\'s (talk|go back|dive))\b',
    r'\b(what (was|is) (your|the))\b',
    r'\b(how did (you|that|it))\b',
    r'\b(i guess|so i guess)\b',
    r'\b(from your (point of view|perspective|experience))\b',
]

# Key insight moments in conversations
PODCAST_INSIGHT_PATTERNS = [
    r'\b(the (key|main|important) (thing|point|takeaway))\b',
    r'\b(i (think|feel|believe) (that |the )?(most|really))\b',
    r'\b(what i\'ve (learned|found|realized))\b',
    r'\b(the truth is|honestly|to be honest)\b',
    r'\b(i would (say|tell|advise))\b',
    r'\b(my advice|one piece of advice)\b',
    r'\b((super|really|very) (important|inspiring|interesting))\b',
    r'\b(that\'s (exactly|precisely)|exactly right)\b',
]

# Emotional/emphasis moments
PODCAST_EMPHASIS_PATTERNS = [
    r'\b(it (genuinely|literally|actually|really) (is|was|changed))\b',
    r'\b(i\'m not (going to|gonna) lie)\b',
    r'\b(do you know what i mean)\b',
    r'\b(like (genuinely|seriously|honestly))\b',
    r'\b(it\'s (hard|difficult|challenging|intense))\b',
    r'\b(it was (incredible|amazing|life-?changing))\b',
    r'\b(i (love|loved) (that|this|it))\b',
]

# Speaker turn indicators (useful for detecting conversation flow)
SPEAKER_TURN_PATTERNS = [
    r'\b(yeah|yes)[,.]?\s*(so|and|but|i)\b',
    r'\b(okay|ok)[,.]?\s*(so|and|i)\b',
    r'\b(right)[,.]?\s*(so|and|exactly)\b',
    r'\b(absolutely|definitely|exactly)\b',
    r'^(so|and|but|yeah)\s',
]

# =============================================================================
# LECTURE/PHILOSOPHY PATTERNS
# =============================================================================

# Definition and explanation markers
LECTURE_DEFINITION_PATTERNS = [
    r'\b(is (defined as|the state of|when))\b',
    r'\b(refers to|means that)\b',
    r'\bwhat is [\w\s]+\?\s*\w+ is\b',  # "What is X? X is..."
    r'\b(this is (called|known as|the))\b',
    r'\b(we (call|define|refer to) this)\b',
]

# Structural/navigational markers in lectures
LECTURE_STRUCTURE_PATTERNS = [
    r'\b(the (first|second|third|next|final) (layer|section|part|point))\b',
    r'\b(we\'ll (explain|explore|cover|discuss))\b',
    r'\b(let\'s (begin|start|move on) (with|to))\b',
    r'\b(this (leads|brings) us to)\b',
    r'\b(the (question|problem|issue) of)\b',
    r'\b(in (summary|conclusion|other words))\b',
]

# Conceptual depth markers
LECTURE_CONCEPT_PATTERNS = [
    r'\b(this (means|implies|suggests) that)\b',
    r'\b(the (key|central|fundamental) (idea|concept|point))\b',
    r'\b(in (philosophy|science|psychology))\b',
    r'\b(according to|as [\w\s]+ (argues|suggests|claims))\b',
    r'\b(the (relationship|connection) between)\b',
]

# =============================================================================
# TUTORIAL PATTERNS
# =============================================================================

# Technical definition patterns
TUTORIAL_DEFINITION_PATTERNS = [
    r'\b(a (high-?level|low-?level|compiled|interpreted))\b',
    r'\b(is (commonly|often|typically) used (to|for))\b',
    r'\b(allows you to|enables you to|lets you)\b',
    r'\b((programming|scripting) language)\b',
    r'\b(framework[s]? (like|such as))\b',
]

# Code/demonstration patterns
TUTORIAL_CODE_PATTERNS = [
    r'\b(create a|define a|declare a)\b',
    r'\b(by (setting|using|adding|calling))\b',
    r'\b(instead of|rather than)\b',
    r'\b(uses? (indentation|semicolons|brackets|syntax))\b',
    r'\b(the (output|result|return value) (is|will be))\b',
    r'\b(run (this|the|your))\b',
]

# Feature enumeration patterns (technical context only)
TUTORIAL_FEATURE_PATTERNS = [
    r'\b(it (also|additionally) (has|supports|includes|provides))\b',
    r'\b((also|additionally) (has|supports|comes with|provides))\b',
    r'\b(frameworks? (like|such as))\b',
    r'\b(libraries? (like|such as))\b',
    r'\b(tools? (like|such as))\b',
    r'\b(with (features|tools|libraries|packages) like)\b',
    r'\b(a huge (ecosystem|library|collection) of)\b',
    r'\b((built-?in|standard) (library|support|features))\b',
]


def detect_visual_references(transcript_segments) -> list[SemanticTrigger]:
    """
    Detect moments where the speaker references visual content.

    These are high-value moments for keyframe extraction.
    """
    triggers = []

    for seg in transcript_segments:
        text = seg.text.lower()

        for pattern in VISUAL_REFERENCE_PATTERNS:
            if re.search(pattern, text):
                triggers.append(SemanticTrigger(
                    timestamp=seg.start,
                    trigger_type='visual_reference',
                    confidence=0.8,
                    context=seg.text[:100]
                ))
                break  # One trigger per segment

    return triggers


def detect_topic_transitions(transcript_segments) -> list[SemanticTrigger]:
    """
    Detect topic transitions in the transcript.

    Useful for segmenting content and capturing transition frames.
    """
    triggers = []

    for seg in transcript_segments:
        text = seg.text.lower()

        for pattern in TOPIC_TRANSITION_PATTERNS:
            if re.search(pattern, text):
                triggers.append(SemanticTrigger(
                    timestamp=seg.start,
                    trigger_type='topic_transition',
                    confidence=0.7,
                    context=seg.text[:100]
                ))
                break

    return triggers


# =============================================================================
# PODCAST-SPECIFIC DETECTION
# =============================================================================

def detect_podcast_topics(segments) -> list[SemanticTrigger]:
    """Detect topic shifts in conversational content."""
    triggers = []

    for seg in segments:
        text = seg.text.lower() if hasattr(seg, 'text') else str(seg).lower()
        start = seg.start if hasattr(seg, 'start') else 0

        for pattern in PODCAST_TOPIC_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                triggers.append(SemanticTrigger(
                    timestamp=start,
                    trigger_type='podcast_topic',
                    confidence=0.7,
                    context=text[:80]
                ))
                break

    return triggers


def detect_podcast_insights(segments) -> list[SemanticTrigger]:
    """Detect key insight moments in conversations."""
    triggers = []

    for seg in segments:
        text = seg.text.lower() if hasattr(seg, 'text') else str(seg).lower()
        start = seg.start if hasattr(seg, 'start') else 0

        for pattern in PODCAST_INSIGHT_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                triggers.append(SemanticTrigger(
                    timestamp=start,
                    trigger_type='podcast_insight',
                    confidence=0.85,
                    context=text[:80]
                ))
                break

    return triggers


def detect_podcast_emphasis(segments) -> list[SemanticTrigger]:
    """Detect emotional/emphasis moments."""
    triggers = []

    for seg in segments:
        text = seg.text.lower() if hasattr(seg, 'text') else str(seg).lower()
        start = seg.start if hasattr(seg, 'start') else 0

        for pattern in PODCAST_EMPHASIS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                triggers.append(SemanticTrigger(
                    timestamp=start,
                    trigger_type='podcast_emphasis',
                    confidence=0.75,
                    context=text[:80]
                ))
                break

    return triggers


def analyze_podcast(transcript_segments) -> dict:
    """
    Full semantic analysis for podcast/conversational content.

    Returns a structured analysis with all detected triggers.
    """
    # First, merge fragmented segments
    merged = merge_segments(transcript_segments)

    # Run all podcast detectors on merged segments
    topics = detect_podcast_topics(merged)
    insights = detect_podcast_insights(merged)
    emphasis = detect_podcast_emphasis(merged)

    # Combine and sort by timestamp
    all_triggers = topics + insights + emphasis
    all_triggers.sort(key=lambda t: t.timestamp)

    return {
        'merged_segments': len(merged),
        'raw_segments': len(transcript_segments),
        'compression_ratio': len(transcript_segments) / len(merged) if merged else 0,
        'triggers': {
            'topics': topics,
            'insights': insights,
            'emphasis': emphasis,
        },
        'all_triggers': all_triggers,
        'trigger_counts': {
            'topics': len(topics),
            'insights': len(insights),
            'emphasis': len(emphasis),
            'total': len(all_triggers),
        }
    }


# =============================================================================
# LECTURE/PHILOSOPHY DETECTION
# =============================================================================

def detect_lecture_definitions(segments) -> list[SemanticTrigger]:
    """Detect definition moments in lecture content."""
    triggers = []
    for seg in segments:
        text = seg.text if hasattr(seg, 'text') else str(seg)
        start = seg.start if hasattr(seg, 'start') else 0
        for pattern in LECTURE_DEFINITION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                triggers.append(SemanticTrigger(
                    timestamp=start,
                    trigger_type='lecture_definition',
                    confidence=0.85,
                    context=text[:80]
                ))
                break
    return triggers


def detect_lecture_structure(segments) -> list[SemanticTrigger]:
    """Detect structural/navigational moments in lectures."""
    triggers = []
    for seg in segments:
        text = seg.text if hasattr(seg, 'text') else str(seg)
        start = seg.start if hasattr(seg, 'start') else 0
        for pattern in LECTURE_STRUCTURE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                triggers.append(SemanticTrigger(
                    timestamp=start,
                    trigger_type='lecture_structure',
                    confidence=0.8,
                    context=text[:80]
                ))
                break
    return triggers


def detect_lecture_concepts(segments) -> list[SemanticTrigger]:
    """Detect conceptual depth moments."""
    triggers = []
    for seg in segments:
        text = seg.text if hasattr(seg, 'text') else str(seg)
        start = seg.start if hasattr(seg, 'start') else 0
        for pattern in LECTURE_CONCEPT_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                triggers.append(SemanticTrigger(
                    timestamp=start,
                    trigger_type='lecture_concept',
                    confidence=0.75,
                    context=text[:80]
                ))
                break
    return triggers


def analyze_lecture(transcript_segments) -> dict:
    """Full semantic analysis for lecture/educational content."""
    merged = merge_segments(transcript_segments)

    definitions = detect_lecture_definitions(merged)
    structure = detect_lecture_structure(merged)
    concepts = detect_lecture_concepts(merged)

    all_triggers = definitions + structure + concepts
    all_triggers.sort(key=lambda t: t.timestamp)

    return {
        'merged_segments': len(merged),
        'raw_segments': len(transcript_segments),
        'compression_ratio': len(transcript_segments) / len(merged) if merged else 0,
        'triggers': {
            'definitions': definitions,
            'structure': structure,
            'concepts': concepts,
        },
        'all_triggers': all_triggers,
        'trigger_counts': {
            'definitions': len(definitions),
            'structure': len(structure),
            'concepts': len(concepts),
            'total': len(all_triggers),
        }
    }


# =============================================================================
# TUTORIAL DETECTION
# =============================================================================

def detect_tutorial_definitions(segments) -> list[SemanticTrigger]:
    """Detect technical definition moments."""
    triggers = []
    for seg in segments:
        text = seg.text if hasattr(seg, 'text') else str(seg)
        start = seg.start if hasattr(seg, 'start') else 0
        for pattern in TUTORIAL_DEFINITION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                triggers.append(SemanticTrigger(
                    timestamp=start,
                    trigger_type='tutorial_definition',
                    confidence=0.85,
                    context=text[:80]
                ))
                break
    return triggers


def detect_tutorial_code(segments) -> list[SemanticTrigger]:
    """Detect code demonstration moments."""
    triggers = []
    for seg in segments:
        text = seg.text if hasattr(seg, 'text') else str(seg)
        start = seg.start if hasattr(seg, 'start') else 0
        for pattern in TUTORIAL_CODE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                triggers.append(SemanticTrigger(
                    timestamp=start,
                    trigger_type='tutorial_code',
                    confidence=0.9,
                    context=text[:80]
                ))
                break
    return triggers


def detect_tutorial_features(segments) -> list[SemanticTrigger]:
    """Detect feature enumeration moments."""
    triggers = []
    for seg in segments:
        text = seg.text if hasattr(seg, 'text') else str(seg)
        start = seg.start if hasattr(seg, 'start') else 0
        for pattern in TUTORIAL_FEATURE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                triggers.append(SemanticTrigger(
                    timestamp=start,
                    trigger_type='tutorial_feature',
                    confidence=0.7,
                    context=text[:80]
                ))
                break
    return triggers


def analyze_tutorial(transcript_segments) -> dict:
    """Full semantic analysis for tutorial/technical content."""
    merged = merge_segments(transcript_segments)

    definitions = detect_tutorial_definitions(merged)
    code = detect_tutorial_code(merged)
    features = detect_tutorial_features(merged)

    all_triggers = definitions + code + features
    all_triggers.sort(key=lambda t: t.timestamp)

    return {
        'merged_segments': len(merged),
        'raw_segments': len(transcript_segments),
        'compression_ratio': len(transcript_segments) / len(merged) if merged else 0,
        'triggers': {
            'definitions': definitions,
            'code': code,
            'features': features,
        },
        'all_triggers': all_triggers,
        'trigger_counts': {
            'definitions': len(definitions),
            'code': len(code),
            'features': len(features),
            'total': len(all_triggers),
        }
    }


# =============================================================================
# UNIFIED ANALYSIS - Auto-detect content type
# =============================================================================

def analyze_content(transcript_segments, content_type: ContentType = None) -> dict:
    """
    Analyze content with appropriate patterns based on content type.

    If content_type is None, runs all analyzers and picks best match.
    """
    merged = merge_segments(transcript_segments)

    results = {
        'podcast': analyze_podcast(transcript_segments),
        'lecture': analyze_lecture(transcript_segments),
        'tutorial': analyze_tutorial(transcript_segments),
    }

    # Find which analyzer found the most triggers
    best_type = max(results.keys(), key=lambda k: results[k]['trigger_counts']['total'])

    return {
        'detected_type': best_type,
        'merged_segments': len(merged),
        'raw_segments': len(transcript_segments),
        'all_results': results,
        'best_result': results[best_type],
    }


def detect_pauses(transcript_segments, min_gap: float = 3.0) -> list[SemanticTrigger]:
    """
    Detect significant pauses in speech.

    Pauses often indicate transitions, emphasis, or demonstration moments.
    """
    triggers = []

    for i in range(1, len(transcript_segments)):
        prev = transcript_segments[i - 1]
        curr = transcript_segments[i]

        gap = curr.start - prev.end

        if gap >= min_gap:
            triggers.append(SemanticTrigger(
                timestamp=prev.end + (gap / 2),  # Middle of pause
                trigger_type='significant_pause',
                confidence=min(0.5 + (gap / 10), 0.9),  # Higher confidence for longer pauses
                context=f"Pause of {gap:.1f}s between segments"
            ))

    return triggers


def plan_keyframe_extraction(
    transcript_segments,
    duration: float,
    max_frames: int = 50,
    min_interval: float = 10.0
) -> list[SemanticTrigger]:
    """
    Plan which frames to extract based on semantic analysis.

    Combines multiple trigger types and ensures reasonable distribution.

    Args:
        transcript_segments: Transcript from extract stage
        duration: Total video duration in seconds
        max_frames: Maximum number of keyframes to extract
        min_interval: Minimum seconds between keyframes

    Returns:
        List of SemanticTriggers indicating when to capture frames
    """
    # Collect all triggers
    all_triggers = []
    all_triggers.extend(detect_visual_references(transcript_segments))
    all_triggers.extend(detect_topic_transitions(transcript_segments))
    all_triggers.extend(detect_pauses(transcript_segments))

    # Sort by timestamp
    all_triggers.sort(key=lambda t: t.timestamp)

    # Filter to respect min_interval
    selected = []
    last_time = -min_interval

    for trigger in all_triggers:
        if trigger.timestamp - last_time >= min_interval:
            selected.append(trigger)
            last_time = trigger.timestamp

            if len(selected) >= max_frames:
                break

    # Ensure we have at least start and end frames
    if selected and selected[0].timestamp > 5.0:
        selected.insert(0, SemanticTrigger(
            timestamp=1.0,
            trigger_type='start',
            confidence=1.0,
            context='Video start'
        ))

    if duration > 30 and (not selected or selected[-1].timestamp < duration - 30):
        selected.append(SemanticTrigger(
            timestamp=duration - 5,
            trigger_type='end',
            confidence=1.0,
            context='Video end'
        ))

    return selected


def analyze_semantic_density(transcript_segments, window_size: float = 60.0) -> list[dict]:
    """
    Analyze semantic density across the content.

    High-density regions might warrant more keyframes.
    Returns density scores per time window.
    """
    if not transcript_segments:
        return []

    # Get total duration
    duration = max(seg.end for seg in transcript_segments)

    windows = []
    current_start = 0

    while current_start < duration:
        window_end = current_start + window_size

        # Count words and triggers in this window
        words = 0
        visual_refs = 0
        topic_shifts = 0

        for seg in transcript_segments:
            if seg.start >= current_start and seg.start < window_end:
                words += len(seg.text.split())

                text = seg.text.lower()
                for pattern in VISUAL_REFERENCE_PATTERNS:
                    if re.search(pattern, text):
                        visual_refs += 1
                        break

                for pattern in TOPIC_TRANSITION_PATTERNS:
                    if re.search(pattern, text):
                        topic_shifts += 1
                        break

        windows.append({
            'start': current_start,
            'end': window_end,
            'word_count': words,
            'visual_references': visual_refs,
            'topic_transitions': topic_shifts,
            'density_score': words / window_size + visual_refs * 10 + topic_shifts * 5
        })

        current_start = window_end

    return windows


if __name__ == '__main__':
    # Test with sample transcript
    from extract import TranscriptSegment

    sample = [
        TranscriptSegment(0, 5, "Welcome to the show"),
        TranscriptSegment(5, 15, "As you can see on the screen, we have our first topic"),
        TranscriptSegment(20, 30, "Now let's move on to the next point"),
        TranscriptSegment(30, 45, "This is really interesting stuff"),
        TranscriptSegment(50, 60, "Look at this diagram here"),
    ]

    print("Visual references:")
    for t in detect_visual_references(sample):
        print(f"  [{t.timestamp:.1f}s] {t.context}")

    print("\nTopic transitions:")
    for t in detect_topic_transitions(sample):
        print(f"  [{t.timestamp:.1f}s] {t.context}")

    print("\nPlanned keyframes:")
    for t in plan_keyframe_extraction(sample, duration=60):
        print(f"  [{t.timestamp:.1f}s] {t.trigger_type}: {t.context}")
