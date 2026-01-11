"""
File Synchronization Engine

Handles concurrent file access for multi-device sync via cloud storage
(Dropbox, iCloud, network drives). Provides file locking, versioning,
and conflict detection.

Usage:
    engine = FileSyncEngine(Path("output/session"))
    data, version = engine.read_with_lock("transcript.json")
    success = engine.write_with_lock("transcript.json", new_data, version)
"""

import json
import hashlib
import socket
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Any


# Lock file timeout - auto-release stale locks after this duration
LOCK_TIMEOUT_SECONDS = 60.0
LOCK_POLL_INTERVAL = 0.1  # seconds


@dataclass
class LockInfo:
    """Information stored in a lock file."""
    pid: int
    hostname: str
    acquired: datetime
    owner_id: str  # Unique identifier for this process

    def to_dict(self) -> dict:
        return {
            "pid": self.pid,
            "hostname": self.hostname,
            "acquired": self.acquired.isoformat(),
            "owner_id": self.owner_id
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LockInfo":
        return cls(
            pid=data["pid"],
            hostname=data["hostname"],
            acquired=datetime.fromisoformat(data["acquired"]),
            owner_id=data.get("owner_id", f"{data['hostname']}:{data['pid']}")
        )

    def is_stale(self, timeout_seconds: float = LOCK_TIMEOUT_SECONDS) -> bool:
        """Check if lock has exceeded timeout."""
        age = datetime.now() - self.acquired
        return age.total_seconds() > timeout_seconds


@dataclass
class SyncableFile:
    """Metadata for a synchronized file."""
    path: Path
    version: int
    last_modified: datetime
    checksum: str

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "version": self.version,
            "last_modified": self.last_modified.isoformat(),
            "checksum": self.checksum
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SyncableFile":
        return cls(
            path=Path(data["path"]),
            version=data["version"],
            last_modified=datetime.fromisoformat(data["last_modified"]),
            checksum=data["checksum"]
        )


class LockManager:
    """
    Cross-process file locking using sidecar .lock files.

    Lock files contain JSON with:
    - pid: Process ID that holds the lock
    - hostname: Machine name
    - acquired: Timestamp when lock was acquired
    - owner_id: Unique identifier for the lock holder
    """

    def __init__(self, lock_timeout: float = LOCK_TIMEOUT_SECONDS):
        self.lock_timeout = lock_timeout
        self._owner_id = f"{socket.gethostname()}:{os.getpid()}:{id(self)}"
        self._held_locks: set[Path] = set()

    def _lock_path(self, file_path: Path) -> Path:
        """Get lock file path for a given file."""
        return file_path.with_suffix(file_path.suffix + ".lock")

    def acquire(self, file_path: Path, timeout: float = 5.0) -> bool:
        """
        Acquire a lock on a file.

        Args:
            file_path: Path to file to lock
            timeout: Maximum time to wait for lock

        Returns:
            True if lock acquired, False if timeout
        """
        lock_path = self._lock_path(file_path)
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Try to read existing lock
            existing_lock = self._read_lock(lock_path)

            if existing_lock is None:
                # No lock exists, try to create one
                if self._create_lock(lock_path):
                    self._held_locks.add(lock_path)
                    return True

            elif existing_lock.owner_id == self._owner_id:
                # We already hold this lock
                return True

            elif existing_lock.is_stale(self.lock_timeout):
                # Stale lock, remove and retry
                self._remove_lock(lock_path)
                continue

            # Lock held by another process, wait and retry
            time.sleep(LOCK_POLL_INTERVAL)

        return False

    def release(self, file_path: Path) -> bool:
        """
        Release a lock on a file.

        Returns:
            True if lock was released, False if we didn't hold it
        """
        lock_path = self._lock_path(file_path)

        existing_lock = self._read_lock(lock_path)
        if existing_lock is None:
            self._held_locks.discard(lock_path)
            return True

        if existing_lock.owner_id != self._owner_id:
            return False

        self._remove_lock(lock_path)
        self._held_locks.discard(lock_path)
        return True

    def is_locked(self, file_path: Path) -> bool:
        """Check if a file is currently locked (by anyone)."""
        lock_path = self._lock_path(file_path)
        existing_lock = self._read_lock(lock_path)

        if existing_lock is None:
            return False

        if existing_lock.is_stale(self.lock_timeout):
            return False

        return True

    def is_locked_by_us(self, file_path: Path) -> bool:
        """Check if we hold the lock on a file."""
        lock_path = self._lock_path(file_path)
        existing_lock = self._read_lock(lock_path)

        if existing_lock is None:
            return False

        return existing_lock.owner_id == self._owner_id

    def _read_lock(self, lock_path: Path) -> Optional[LockInfo]:
        """Read lock file, return None if doesn't exist or invalid."""
        try:
            if not lock_path.exists():
                return None
            with open(lock_path, 'r') as f:
                data = json.load(f)
                return LockInfo.from_dict(data)
        except (json.JSONDecodeError, KeyError, OSError):
            return None

    def _create_lock(self, lock_path: Path) -> bool:
        """
        Atomically create a lock file.

        Uses exclusive creation mode to prevent race conditions.
        """
        lock_info = LockInfo(
            pid=os.getpid(),
            hostname=socket.gethostname(),
            acquired=datetime.now(),
            owner_id=self._owner_id
        )

        try:
            # Use exclusive creation - fails if file exists
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, 'w') as f:
                json.dump(lock_info.to_dict(), f)
            return True
        except FileExistsError:
            return False
        except OSError:
            return False

    def _remove_lock(self, lock_path: Path) -> None:
        """Remove a lock file."""
        try:
            lock_path.unlink()
        except OSError:
            pass

    def release_all(self) -> None:
        """Release all locks held by this manager."""
        for lock_path in list(self._held_locks):
            file_path = lock_path.with_suffix('')  # Remove .lock suffix
            if lock_path.suffix == '.lock':
                # Handle double suffix like .json.lock
                file_path = Path(str(lock_path)[:-5])
            self.release(file_path)

    def __del__(self):
        """Clean up locks on deletion."""
        self.release_all()


class FileSyncEngine:
    """
    Manages file operations for multi-device synchronization.

    Provides:
    - Atomic read/write with locking
    - Version tracking for conflict detection
    - Append operations for incremental updates
    - Checksum computation for change detection
    """

    def __init__(self, work_dir: Path, lock_timeout: float = 5.0):
        self.work_dir = Path(work_dir)
        self.lock_manager = LockManager(lock_timeout=LOCK_TIMEOUT_SECONDS)
        self.lock_timeout = lock_timeout

        # Ensure work directory exists
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Version tracking file
        self._version_file = self.work_dir / ".sync_versions.json"
        self._versions: dict[str, int] = self._load_versions()

    def _load_versions(self) -> dict[str, int]:
        """Load version tracking from disk."""
        if self._version_file.exists():
            try:
                with open(self._version_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_versions(self) -> None:
        """Save version tracking to disk."""
        try:
            with open(self._version_file, 'w') as f:
                json.dump(self._versions, f)
        except OSError as e:
            print(f"Warning: Could not save version tracking: {e}")

    def _file_path(self, filename: str) -> Path:
        """Get full path for a filename."""
        return self.work_dir / filename

    def get_version(self, filename: str) -> int:
        """Get current version of a file."""
        return self._versions.get(filename, 0)

    def increment_version(self, filename: str) -> int:
        """Increment and return new version for a file."""
        self._versions[filename] = self._versions.get(filename, 0) + 1
        self._save_versions()
        return self._versions[filename]

    def compute_checksum(self, data: Any) -> str:
        """Compute SHA256 checksum of data."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def read_with_lock(
        self,
        filename: str,
        timeout: float = None
    ) -> tuple[Optional[dict], int]:
        """
        Read a file with locking.

        Args:
            filename: File to read (relative to work_dir)
            timeout: Lock timeout (uses default if None)

        Returns:
            Tuple of (data, version). Data is None if file doesn't exist.
        """
        file_path = self._file_path(filename)
        timeout = timeout or self.lock_timeout

        if not self.lock_manager.acquire(file_path, timeout):
            raise TimeoutError(f"Could not acquire lock on {filename}")

        try:
            if not file_path.exists():
                return None, 0

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            version = self.get_version(filename)
            return data, version

        finally:
            self.lock_manager.release(file_path)

    def write_with_lock(
        self,
        filename: str,
        data: dict,
        expected_version: int = None,
        timeout: float = None
    ) -> bool:
        """
        Write a file with locking and optional version check.

        Args:
            filename: File to write
            data: Data to write (must be JSON-serializable)
            expected_version: If set, only write if current version matches
            timeout: Lock timeout

        Returns:
            True if write succeeded, False if version mismatch
        """
        file_path = self._file_path(filename)
        timeout = timeout or self.lock_timeout

        if not self.lock_manager.acquire(file_path, timeout):
            raise TimeoutError(f"Could not acquire lock on {filename}")

        try:
            current_version = self.get_version(filename)

            # Version check for optimistic concurrency
            if expected_version is not None and current_version != expected_version:
                return False

            # Write atomically using temp file
            temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)

            # Rename is atomic on most systems
            temp_path.replace(file_path)

            # Increment version
            self.increment_version(filename)
            return True

        finally:
            self.lock_manager.release(file_path)

    def append_with_lock(
        self,
        filename: str,
        items: list,
        list_key: str = "segments",
        timeout: float = None
    ) -> bool:
        """
        Append items to a list within a JSON file.

        Args:
            filename: File to append to
            items: Items to append
            list_key: Key containing the list in the JSON structure
            timeout: Lock timeout

        Returns:
            True if append succeeded
        """
        file_path = self._file_path(filename)
        timeout = timeout or self.lock_timeout

        if not self.lock_manager.acquire(file_path, timeout):
            raise TimeoutError(f"Could not acquire lock on {filename}")

        try:
            # Read existing data
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = {list_key: [], "version": 2}

            # Append items
            if list_key not in data:
                data[list_key] = []
            data[list_key].extend(items)

            # Update sync metadata
            if "sync" not in data:
                data["sync"] = {}
            data["sync"]["last_modified"] = datetime.now().isoformat()
            data["sync"]["item_count"] = len(data[list_key])

            # Write back
            temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            temp_path.replace(file_path)

            self.increment_version(filename)
            return True

        finally:
            self.lock_manager.release(file_path)

    def get_sync_manifest(self) -> dict:
        """
        Get current sync manifest showing all tracked files.

        Returns:
            Dict with file versions and checksums
        """
        manifest = {
            "version": 1,
            "generated": datetime.now().isoformat(),
            "work_dir": str(self.work_dir),
            "files": {}
        }

        # Check common sync files
        sync_files = [
            "transcript.json",
            "concepts.json",
            "session_state.json",
            "digest.partial.md"
        ]

        for filename in sync_files:
            file_path = self._file_path(filename)
            if file_path.exists():
                stat = file_path.stat()
                manifest["files"][filename] = {
                    "version": self.get_version(filename),
                    "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "size": stat.st_size
                }

        return manifest

    def save_sync_manifest(self) -> Path:
        """Save sync manifest to disk."""
        manifest = self.get_sync_manifest()
        manifest_path = self.work_dir / "sync_manifest.json"

        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)

        return manifest_path

    def release_all_locks(self) -> None:
        """Release all locks held by this engine."""
        self.lock_manager.release_all()


def merge_transcripts(
    local: list[dict],
    remote: list[dict]
) -> list[dict]:
    """
    Merge two transcript segment lists.

    Uses timestamps to deduplicate and order segments.
    Prefers remote version for conflicts at same timestamp.

    Args:
        local: Local transcript segments
        remote: Remote transcript segments

    Returns:
        Merged and deduplicated segment list
    """
    # Index by start timestamp
    merged = {}

    for seg in local:
        key = (seg.get("start", 0), seg.get("text", "")[:50])
        merged[key] = seg

    for seg in remote:
        key = (seg.get("start", 0), seg.get("text", "")[:50])
        merged[key] = seg  # Remote wins on conflict

    # Sort by start time
    result = sorted(merged.values(), key=lambda s: s.get("start", 0))
    return result


def merge_entities(
    local: dict[str, dict],
    remote: dict[str, dict]
) -> dict[str, dict]:
    """
    Merge two entity dictionaries.

    Combines mention counts and aliases.
    Uses higher mention count as canonical.

    Args:
        local: Local entity dict
        remote: Remote entity dict

    Returns:
        Merged entity dict
    """
    merged = dict(local)

    for entity_id, remote_entity in remote.items():
        if entity_id not in merged:
            merged[entity_id] = remote_entity
        else:
            local_entity = merged[entity_id]

            # Merge aliases
            local_aliases = set(local_entity.get("aliases", []))
            remote_aliases = set(remote_entity.get("aliases", []))
            merged[entity_id]["aliases"] = list(local_aliases | remote_aliases)

            # Merge mentions (dedupe by timestamp)
            local_mentions = {m.get("start", 0): m for m in local_entity.get("mentions", [])}
            for m in remote_entity.get("mentions", []):
                local_mentions[m.get("start", 0)] = m
            merged[entity_id]["mentions"] = list(local_mentions.values())

            # Update count
            merged[entity_id]["mention_count"] = len(merged[entity_id]["mentions"])

    return merged
