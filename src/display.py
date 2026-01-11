"""
Live Terminal Display

Rich terminal UI for real-time stream processing visualization.
Shows live transcript, extracted entities, and progress.

Usage:
    display = LiveDisplay()
    display.start()
    display.on_transcript_update(event)
    display.stop()
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.layout import Layout
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None

from .realtime import StreamEvent, EventBus, EVENT_TRANSCRIPT_UPDATE, EVENT_ENTITY_FOUND, EVENT_STATUS_CHANGE


@dataclass
class TranscriptLine:
    """A line in the transcript display."""
    timestamp: float
    text: str
    is_new: bool = False


@dataclass
class EntityDisplay:
    """An entity for display."""
    entity_id: str
    canonical_name: str
    entity_type: str
    mention_count: int
    is_new: bool = False
    last_seen: datetime = field(default_factory=datetime.now)


class LiveDisplay:
    """
    Rich terminal UI for real-time stream processing.

    Layout:
    ┌─────────────────────────────────────────────────────────┐
    │ Stream: [title]                            LIVE [time] │
    ├─────────────────────────────────────────────────────────┤
    │                                                         │
    │  TRANSCRIPT (scrolling)                                 │
    │  [00:01:23] This is the live transcript...             │
    │  [00:01:25] As words are spoken they appear...         │
    │  [00:01:28] New lines highlighted briefly...           │
    │                                                         │
    ├─────────────────────────────────────────────────────────┤
    │  ENTITIES              │  PROGRESS                     │
    │  PERSON: Sam Harris    │  Chunks: 5 / ~10              │
    │  ORG: OpenAI           │  Words: 1,234                 │
    │  GPE: San Francisco    │  Entities: 12                 │
    └─────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        console: Optional['Console'] = None,
        max_transcript_lines: int = 12,
        max_entities: int = 8
    ):
        if not RICH_AVAILABLE:
            raise ImportError("rich library required for LiveDisplay. Install with: pip install rich")

        self.console = console or Console()
        self.max_transcript_lines = max_transcript_lines
        self.max_entities = max_entities

        # State
        self.transcript_lines: list[TranscriptLine] = []
        self.entities: dict[str, EntityDisplay] = {}
        self.title: str = "Stream"
        self.start_time: Optional[datetime] = None
        self.chunks_processed: int = 0
        self.total_words: int = 0
        self.is_capturing: bool = False
        self.is_processing: bool = False

        # Rich Live display
        self._live: Optional[Live] = None

    def start(self, title: str = "Stream") -> None:
        """Start the live display."""
        self.title = title
        self.start_time = datetime.now()
        self.is_capturing = True

        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=4,
            transient=False
        )
        self._live.start()

    def stop(self) -> None:
        """Stop the live display."""
        self.is_capturing = False
        if self._live:
            self._live.stop()
            self._live = None

    def refresh(self) -> None:
        """Force a display refresh."""
        if self._live:
            self._live.update(self._render())

    def _render(self) -> Panel:
        """Render the complete display layout."""
        layout = Layout()

        # Header
        elapsed = ""
        if self.start_time:
            delta = datetime.now() - self.start_time
            elapsed = str(timedelta(seconds=int(delta.total_seconds())))

        status = "[bold red]LIVE[/]" if self.is_capturing else "[dim]STOPPED[/]"
        if self.is_processing:
            status = "[bold yellow]PROCESSING[/]"

        header = Text()
        header.append(f" {self.title[:50]}", style="bold cyan")
        header.append(f"  {status}  ", style="")
        header.append(elapsed, style="dim")

        # Transcript panel
        transcript_panel = self._render_transcript()

        # Bottom row: entities + progress
        entities_panel = self._render_entities()
        progress_panel = self._render_progress()

        # Combine into layout
        bottom_table = Table.grid(expand=True)
        bottom_table.add_column(ratio=1)
        bottom_table.add_column(ratio=1)
        bottom_table.add_row(entities_panel, progress_panel)

        content = Table.grid(expand=True)
        content.add_column()
        content.add_row(transcript_panel)
        content.add_row(bottom_table)

        return Panel(
            content,
            title=header,
            border_style="blue",
            box=box.ROUNDED
        )

    def _render_transcript(self) -> Panel:
        """Render the transcript panel."""
        text = Text()

        # Get recent lines
        lines = self.transcript_lines[-self.max_transcript_lines:]

        if not lines:
            text.append("Waiting for transcript...", style="dim italic")
        else:
            for i, line in enumerate(lines):
                # Timestamp
                mins = int(line.timestamp // 60)
                secs = int(line.timestamp % 60)
                ts_str = f"[{mins:02d}:{secs:02d}]"

                style = "bold green" if line.is_new else "dim"
                text.append(ts_str, style=style)
                text.append(" ")

                # Text content
                text_style = "white" if line.is_new else ""
                text.append(line.text[:80], style=text_style)

                if i < len(lines) - 1:
                    text.append("\n")

        return Panel(
            text,
            title="[bold]Transcript[/]",
            border_style="green",
            height=self.max_transcript_lines + 2
        )

    def _render_entities(self) -> Panel:
        """Render the entities panel."""
        table = Table(box=None, show_header=False, padding=(0, 1))
        table.add_column("Type", style="dim", width=8)
        table.add_column("Name", style="white")
        table.add_column("Count", style="dim", justify="right", width=4)

        # Sort by mention count, take top N
        sorted_entities = sorted(
            self.entities.values(),
            key=lambda e: e.mention_count,
            reverse=True
        )[:self.max_entities]

        if not sorted_entities:
            return Panel(
                Text("No entities yet...", style="dim italic"),
                title="[bold]Entities[/]",
                border_style="yellow"
            )

        for entity in sorted_entities:
            type_style = self._entity_type_style(entity.entity_type)
            name_style = "bold green" if entity.is_new else "white"

            table.add_row(
                Text(entity.entity_type[:7], style=type_style),
                Text(entity.canonical_name[:20], style=name_style),
                str(entity.mention_count)
            )

        return Panel(
            table,
            title="[bold]Entities[/]",
            border_style="yellow"
        )

    def _render_progress(self) -> Panel:
        """Render the progress panel."""
        table = Table(box=None, show_header=False, padding=(0, 1))
        table.add_column("Label", style="dim")
        table.add_column("Value", style="bold white", justify="right")

        table.add_row("Chunks", str(self.chunks_processed))
        table.add_row("Words", f"{self.total_words:,}")
        table.add_row("Entities", str(len(self.entities)))

        if self.start_time:
            elapsed = datetime.now() - self.start_time
            table.add_row("Elapsed", str(timedelta(seconds=int(elapsed.total_seconds()))))

        return Panel(
            table,
            title="[bold]Progress[/]",
            border_style="magenta"
        )

    def _entity_type_style(self, entity_type: str) -> str:
        """Get color style for entity type."""
        styles = {
            "PERSON": "cyan",
            "ORG": "yellow",
            "GPE": "green",
            "LOC": "green",
            "WORK_OF_ART": "magenta",
            "EVENT": "red",
            "PRODUCT": "blue",
        }
        return styles.get(entity_type, "dim")

    # Event handlers

    def on_transcript_update(self, event: StreamEvent) -> None:
        """Handle transcript update events."""
        segments = event.payload.get("new_segments", [])

        # Mark existing as not new
        for line in self.transcript_lines:
            line.is_new = False

        # Add new segments
        for seg in segments:
            self.transcript_lines.append(TranscriptLine(
                timestamp=seg.get("start", 0),
                text=seg.get("text", ""),
                is_new=True
            ))

        # Update stats
        self.total_words = event.payload.get("total_words", self.total_words)

        self.refresh()

    def on_entity_update(self, event: StreamEvent) -> None:
        """Handle entity found/update events."""
        entity_id = event.payload.get("entity_id")
        if not entity_id:
            return

        # Mark existing as not new
        for ent in self.entities.values():
            ent.is_new = False

        # Add or update entity
        is_new = event.payload.get("is_new", False)

        if entity_id in self.entities:
            self.entities[entity_id].mention_count += 1
            self.entities[entity_id].is_new = is_new
            self.entities[entity_id].last_seen = datetime.now()
        else:
            self.entities[entity_id] = EntityDisplay(
                entity_id=entity_id,
                canonical_name=event.payload.get("canonical_name", entity_id),
                entity_type=event.payload.get("entity_type", "UNKNOWN"),
                mention_count=1,
                is_new=True
            )

        self.refresh()

    def on_status_change(self, event: StreamEvent) -> None:
        """Handle status change events."""
        self.is_capturing = event.payload.get("is_capturing", self.is_capturing)
        self.is_processing = event.payload.get("is_processing", self.is_processing)
        self.chunks_processed = event.payload.get("chunks_processed", self.chunks_processed)
        self.refresh()

    def on_chunk_processed(self) -> None:
        """Called when a chunk finishes processing."""
        self.chunks_processed += 1
        self.refresh()

    def subscribe_to_events(self, event_bus: EventBus) -> list[str]:
        """
        Subscribe to all relevant events from an EventBus.

        Returns:
            List of subscription IDs
        """
        subscriptions = []

        subscriptions.append(
            event_bus.subscribe(EVENT_TRANSCRIPT_UPDATE, self.on_transcript_update)
        )
        subscriptions.append(
            event_bus.subscribe(EVENT_ENTITY_FOUND, self.on_entity_update)
        )
        subscriptions.append(
            event_bus.subscribe("entity_update", self.on_entity_update)
        )
        subscriptions.append(
            event_bus.subscribe(EVENT_STATUS_CHANGE, self.on_status_change)
        )

        return subscriptions


class SimpleDisplay:
    """
    Fallback display when rich is not available.

    Prints updates to stdout in a simple format.
    """

    def __init__(self):
        self.last_timestamp = 0.0

    def start(self, title: str = "Stream") -> None:
        print(f"\n=== {title} ===")
        print("Capturing stream... (Install 'rich' for better display)\n")

    def stop(self) -> None:
        print("\n=== Stream ended ===\n")

    def refresh(self) -> None:
        pass

    def on_transcript_update(self, event: StreamEvent) -> None:
        segments = event.payload.get("new_segments", [])
        for seg in segments:
            ts = seg.get("start", 0)
            if ts > self.last_timestamp:
                mins = int(ts // 60)
                secs = int(ts % 60)
                print(f"[{mins:02d}:{secs:02d}] {seg.get('text', '')}")
                self.last_timestamp = ts

    def on_entity_update(self, event: StreamEvent) -> None:
        if event.payload.get("is_new"):
            name = event.payload.get("canonical_name", "?")
            etype = event.payload.get("entity_type", "?")
            print(f"  -> Entity: {name} ({etype})")

    def on_status_change(self, event: StreamEvent) -> None:
        pass

    def on_chunk_processed(self) -> None:
        print(".", end="", flush=True)

    def subscribe_to_events(self, event_bus: EventBus) -> list[str]:
        subscriptions = []
        subscriptions.append(
            event_bus.subscribe(EVENT_TRANSCRIPT_UPDATE, self.on_transcript_update)
        )
        subscriptions.append(
            event_bus.subscribe(EVENT_ENTITY_FOUND, self.on_entity_update)
        )
        return subscriptions


def create_display(use_rich: bool = True) -> LiveDisplay | SimpleDisplay:
    """
    Create appropriate display based on availability.

    Args:
        use_rich: Prefer rich display if available

    Returns:
        LiveDisplay or SimpleDisplay instance
    """
    if use_rich and RICH_AVAILABLE:
        return LiveDisplay()
    return SimpleDisplay()
