"""
Real-time Event System

Provides publish/subscribe event bus for coordinating real-time updates
between stream capture, transcription, entity extraction, and display.

Usage:
    bus = EventBus()
    bus.subscribe("transcript_update", lambda e: print(e.payload))
    bus.publish(StreamEvent("transcript_update", payload={"text": "Hello"}))
"""

import uuid
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional, Any
from pathlib import Path

from .extract import TranscriptSegment


# Event types
EVENT_CHUNK_READY = "chunk_ready"
EVENT_TRANSCRIPT_UPDATE = "transcript_update"
EVENT_ENTITY_FOUND = "entity_found"
EVENT_ENTITY_UPDATE = "entity_update"
EVENT_COOCCURRENCE = "cooccurrence"
EVENT_DIGEST_UPDATE = "digest_update"
EVENT_STATUS_CHANGE = "status_change"
EVENT_ERROR = "error"


@dataclass
class StreamEvent:
    """An event emitted during stream processing."""
    event_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    payload: dict = field(default_factory=dict)
    source: Optional[str] = None  # Component that emitted the event

    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)


@dataclass
class Subscription:
    """A registered event subscription."""
    subscription_id: str
    event_type: str
    callback: Callable[[StreamEvent], None]
    priority: int = 0  # Higher priority = called first


class EventBus:
    """
    Thread-safe publish/subscribe event dispatcher.

    Coordinates real-time updates between stream components:
    - Stream capture -> chunk_ready events
    - Transcriber -> transcript_update events
    - Entity extractor -> entity_found, entity_update events
    - Digest generator -> digest_update events
    """

    def __init__(self):
        self._subscriptions: dict[str, list[Subscription]] = {}
        self._lock = threading.RLock()
        self._event_history: list[StreamEvent] = []
        self._max_history = 100

    def subscribe(
        self,
        event_type: str,
        callback: Callable[[StreamEvent], None],
        priority: int = 0
    ) -> str:
        """
        Subscribe to events of a given type.

        Args:
            event_type: Event type to subscribe to (e.g., "transcript_update")
            callback: Function to call when event occurs
            priority: Higher priority callbacks run first

        Returns:
            Subscription ID for unsubscribing
        """
        subscription_id = str(uuid.uuid4())
        subscription = Subscription(
            subscription_id=subscription_id,
            event_type=event_type,
            callback=callback,
            priority=priority
        )

        with self._lock:
            if event_type not in self._subscriptions:
                self._subscriptions[event_type] = []
            self._subscriptions[event_type].append(subscription)
            # Sort by priority (descending)
            self._subscriptions[event_type].sort(key=lambda s: -s.priority)

        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Remove a subscription by ID.

        Returns:
            True if subscription was found and removed
        """
        with self._lock:
            for event_type, subs in self._subscriptions.items():
                for sub in subs:
                    if sub.subscription_id == subscription_id:
                        subs.remove(sub)
                        return True
        return False

    def publish(self, event: StreamEvent) -> None:
        """
        Publish an event to all subscribers.

        Calls each subscriber's callback synchronously in priority order.
        Errors in callbacks are caught and logged but don't stop propagation.
        """
        with self._lock:
            # Store in history
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history = self._event_history[-self._max_history:]

            # Get subscribers
            subscribers = list(self._subscriptions.get(event.event_type, []))

        # Call subscribers outside lock
        for sub in subscribers:
            try:
                sub.callback(event)
            except Exception as e:
                print(f"Warning: Event callback error for {event.event_type}: {e}")

    def publish_async(self, event: StreamEvent) -> threading.Thread:
        """
        Publish an event asynchronously in a background thread.

        Returns:
            Thread object (already started)
        """
        thread = threading.Thread(target=self.publish, args=(event,), daemon=True)
        thread.start()
        return thread

    def get_history(self, event_type: Optional[str] = None) -> list[StreamEvent]:
        """Get recent event history, optionally filtered by type."""
        with self._lock:
            if event_type:
                return [e for e in self._event_history if e.event_type == event_type]
            return list(self._event_history)

    def clear_history(self) -> None:
        """Clear event history."""
        with self._lock:
            self._event_history.clear()


@dataclass
class StreamContext:
    """
    Shared state for a real-time streaming session.

    Holds accumulated data that multiple components need access to:
    - Transcript buffer (all segments so far)
    - Entity buffer (extracted entities)
    - Pending co-occurrence mentions (for sliding window)
    """

    session_id: str
    work_dir: Path
    event_bus: EventBus

    # Accumulated data
    transcript_buffer: list[TranscriptSegment] = field(default_factory=list)
    entity_buffer: dict[str, Any] = field(default_factory=dict)  # entity_id -> ExtractedEntity

    # For sliding window co-occurrence
    # List of (timestamp, entity_id) tuples from recent segments
    pending_mentions: list[tuple[float, str]] = field(default_factory=list)
    cooccurrence_window: float = 30.0  # seconds

    # Processing state
    chunks_processed: int = 0
    last_processed_timestamp: float = 0.0
    is_capturing: bool = False
    is_processing: bool = False

    # Stats
    total_words: int = 0
    total_entities: int = 0

    def add_transcript_segments(self, segments: list[TranscriptSegment]) -> None:
        """Add new transcript segments and update stats."""
        self.transcript_buffer.extend(segments)
        for seg in segments:
            self.total_words += len(seg.text.split())
            if seg.end > self.last_processed_timestamp:
                self.last_processed_timestamp = seg.end

        # Publish event
        self.event_bus.publish(StreamEvent(
            event_type=EVENT_TRANSCRIPT_UPDATE,
            payload={
                "new_segments": [{"start": s.start, "end": s.end, "text": s.text} for s in segments],
                "total_segments": len(self.transcript_buffer),
                "total_words": self.total_words
            },
            source="stream_context"
        ))

    def add_entity(self, entity_id: str, entity: Any) -> bool:
        """
        Add or update an entity in the buffer.

        Returns:
            True if this is a new entity, False if update
        """
        is_new = entity_id not in self.entity_buffer
        self.entity_buffer[entity_id] = entity

        if is_new:
            self.total_entities += 1

        # Publish event
        self.event_bus.publish(StreamEvent(
            event_type=EVENT_ENTITY_FOUND if is_new else EVENT_ENTITY_UPDATE,
            payload={
                "entity_id": entity_id,
                "entity_type": getattr(entity, 'entity_type', 'UNKNOWN'),
                "canonical_name": getattr(entity, 'canonical_name', entity_id),
                "is_new": is_new,
                "total_entities": self.total_entities
            },
            source="stream_context"
        ))

        return is_new

    def add_mention_for_cooccurrence(self, timestamp: float, entity_id: str) -> None:
        """
        Track a mention for sliding window co-occurrence calculation.

        Prunes mentions older than the window.
        """
        self.pending_mentions.append((timestamp, entity_id))

        # Prune old mentions
        cutoff = timestamp - self.cooccurrence_window
        self.pending_mentions = [
            (t, eid) for t, eid in self.pending_mentions
            if t >= cutoff
        ]

    def get_recent_cooccurrences(self) -> list[tuple[str, str]]:
        """
        Get entity pairs that co-occurred within the window.

        Returns:
            List of (entity_a_id, entity_b_id) tuples, normalized order
        """
        pairs = set()
        mentions = self.pending_mentions

        for i, (t1, e1) in enumerate(mentions):
            for t2, e2 in mentions[i+1:]:
                if e1 != e2 and abs(t2 - t1) <= self.cooccurrence_window:
                    # Normalize order for consistent counting
                    pair = tuple(sorted([e1, e2]))
                    pairs.add(pair)

        return list(pairs)

    def update_status(self, capturing: bool = None, processing: bool = None) -> None:
        """Update processing status and emit event."""
        if capturing is not None:
            self.is_capturing = capturing
        if processing is not None:
            self.is_processing = processing

        self.event_bus.publish(StreamEvent(
            event_type=EVENT_STATUS_CHANGE,
            payload={
                "is_capturing": self.is_capturing,
                "is_processing": self.is_processing,
                "chunks_processed": self.chunks_processed
            },
            source="stream_context"
        ))


def create_stream_context(
    session_id: str,
    work_dir: Path,
    event_bus: Optional[EventBus] = None
) -> StreamContext:
    """
    Create a new StreamContext for a streaming session.

    Args:
        session_id: Unique session identifier
        work_dir: Directory for session output
        event_bus: Optional existing EventBus, creates new if None

    Returns:
        Initialized StreamContext
    """
    if event_bus is None:
        event_bus = EventBus()

    return StreamContext(
        session_id=session_id,
        work_dir=work_dir,
        event_bus=event_bus
    )
