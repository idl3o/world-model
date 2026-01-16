"""
OCR Module: Extract text from keyframe images.

Uses EasyOCR for scene text recognition - handles slides, code,
whiteboards, and other visual text in video frames.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TextBlock:
    """A single detected text region."""
    text: str
    confidence: float
    bbox: tuple[tuple[int, int], ...]  # Bounding box corners


@dataclass
class OCRResult:
    """Complete OCR output for an image."""
    text: str  # Full extracted text (all blocks joined)
    blocks: list[TextBlock] = field(default_factory=list)
    confidence: float = 0.0  # Average confidence


# =============================================================================
# EasyOCR Model Cache
# =============================================================================

_ocr_reader = None
_ocr_languages = None


def get_ocr_reader(languages: list[str] = None):
    """
    Get cached EasyOCR reader instance.

    Args:
        languages: List of language codes (default: ['en'])
                   Common codes: 'en', 'ch_sim', 'ja', 'ko', 'fr', 'de', 'es'

    Returns:
        EasyOCR Reader instance, or None if import fails
    """
    global _ocr_reader, _ocr_languages

    if languages is None:
        languages = ['en']

    # Return cached if same languages requested
    if _ocr_reader is not None and _ocr_languages == languages:
        return _ocr_reader

    try:
        import easyocr
    except ImportError:
        print("  Warning: EasyOCR not installed. Run: pip install easyocr")
        return None

    print(f"  Loading OCR model for languages: {languages}...")
    _ocr_reader = easyocr.Reader(languages, gpu=_check_gpu_available())
    _ocr_languages = languages
    print(f"  OCR model loaded.")

    return _ocr_reader


def _check_gpu_available() -> bool:
    """Check if CUDA GPU is available for acceleration."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def unload_ocr_model():
    """Unload cached OCR model to free memory."""
    global _ocr_reader, _ocr_languages
    _ocr_reader = None
    _ocr_languages = None


def extract_text(
    image_path: Path,
    languages: list[str] = None,
    min_confidence: float = 0.3,
    paragraph: bool = True
) -> OCRResult:
    """
    Extract text from an image using OCR.

    Args:
        image_path: Path to image file (jpg, png, etc.)
        languages: Language codes for detection
        min_confidence: Minimum confidence threshold (0-1)
        paragraph: Whether to group text into paragraphs

    Returns:
        OCRResult with extracted text and metadata
    """
    reader = get_ocr_reader(languages)
    if reader is None:
        return OCRResult(text="", confidence=0.0)

    try:
        # Run OCR
        results = reader.readtext(
            str(image_path),
            paragraph=paragraph,
            min_size=10,
            text_threshold=0.7,
            low_text=0.4
        )
    except Exception as e:
        print(f"  Warning: OCR failed for {image_path}: {e}")
        return OCRResult(text="", confidence=0.0)

    if not results:
        return OCRResult(text="", confidence=0.0)

    # Parse results
    blocks = []
    texts = []
    confidences = []

    for result in results:
        # EasyOCR returns: (bbox, text, confidence)
        if len(result) >= 3:
            bbox, text, confidence = result[0], result[1], result[2]
        elif len(result) == 2:
            bbox, text = result[0], result[1]
            confidence = 1.0
        else:
            continue

        # Filter by confidence
        if confidence < min_confidence:
            continue

        # Clean text
        text = text.strip()
        if not text:
            continue

        blocks.append(TextBlock(
            text=text,
            confidence=confidence,
            bbox=tuple(tuple(p) for p in bbox) if bbox else ()
        ))
        texts.append(text)
        confidences.append(confidence)

    # Combine text
    full_text = " ".join(texts)
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    return OCRResult(
        text=full_text,
        blocks=blocks,
        confidence=avg_confidence
    )


def extract_text_batch(
    image_paths: list[Path],
    languages: list[str] = None,
    min_confidence: float = 0.3
) -> dict[Path, OCRResult]:
    """
    Extract text from multiple images.

    More efficient than calling extract_text() repeatedly
    as it ensures the model is loaded once.

    Args:
        image_paths: List of image paths
        languages: Language codes
        min_confidence: Minimum confidence threshold

    Returns:
        Dict mapping image paths to OCR results
    """
    # Pre-load model
    reader = get_ocr_reader(languages)
    if reader is None:
        return {p: OCRResult(text="", confidence=0.0) for p in image_paths}

    results = {}
    for path in image_paths:
        results[path] = extract_text(
            path,
            languages=languages,
            min_confidence=min_confidence
        )

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.ocr <image_file>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Error: File not found: {image_path}")
        sys.exit(1)

    print(f"Running OCR on: {image_path}")
    result = extract_text(image_path)

    print(f"\nResults:")
    print(f"  Confidence: {result.confidence:.2%}")
    print(f"  Text blocks: {len(result.blocks)}")
    print(f"\nExtracted text:")
    print(result.text or "(no text detected)")
