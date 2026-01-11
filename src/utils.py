"""
Shared utilities for World Model pipeline.
"""

import subprocess
from pathlib import Path


def check_dependencies() -> dict[str, bool]:
    """Check if required external tools are available."""
    deps = {}

    # Check yt-dlp
    try:
        result = subprocess.run(['yt-dlp', '--version'], capture_output=True)
        deps['yt-dlp'] = result.returncode == 0
    except FileNotFoundError:
        deps['yt-dlp'] = False

    # Check ffmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True)
        deps['ffmpeg'] = result.returncode == 0
    except FileNotFoundError:
        deps['ffmpeg'] = False

    # Check whisper (Python package)
    try:
        import whisper
        deps['whisper'] = True
    except ImportError:
        deps['whisper'] = False

    return deps


def print_dependency_status():
    """Print status of all dependencies."""
    deps = check_dependencies()

    print("Dependency Status:")
    print("-" * 30)

    for name, available in deps.items():
        status = "OK" if available else "MISSING"
        print(f"  {name}: {status}")

    missing = [name for name, ok in deps.items() if not ok]
    if missing:
        print()
        print("To install missing dependencies:")
        if 'yt-dlp' in missing:
            print("  pip install yt-dlp")
        if 'ffmpeg' in missing:
            print("  # Install ffmpeg from https://ffmpeg.org/")
            print("  # Or: choco install ffmpeg (Windows)")
            print("  # Or: brew install ffmpeg (Mac)")
        if 'whisper' in missing:
            print("  pip install openai-whisper")

    return len(missing) == 0


def ensure_output_dir(output_dir: Path) -> Path:
    """Ensure output directory exists."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def sanitize_filename(name: str) -> str:
    """Sanitize string for use as filename."""
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, '_')
    return name[:200]  # Limit length


if __name__ == '__main__':
    print_dependency_status()
