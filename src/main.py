"""
World Model: Main Pipeline Entry Point

Usage:
    python -m src.main <youtube_url>
    python -m src.main <youtube_url> --skip-download  # Use existing assets
    python -m src.main <stream_url>  # Auto-detects live streams
"""

import argparse
from pathlib import Path

from .transform import transform, transform_local, extract_video_id, is_local_file
from .extract import extract, load_transcript, ExtractedContent
from .digest import digest
from .stream import detect_stream_type, run_stream_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Transform media into AI-native format"
    )
    parser.add_argument(
        'url',
        help="YouTube URL, video ID, or local file path"
    )
    parser.add_argument(
        '--title', '-t',
        type=str,
        default=None,
        help="Custom title (for local files)"
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path(__file__).parent.parent / 'output',
        help="Output directory (default: ./output)"
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help="Skip download, use existing assets"
    )
    parser.add_argument(
        '--skip-transcribe',
        action='store_true',
        help="Skip transcription, use existing transcript"
    )
    parser.add_argument(
        '--skip-concepts',
        action='store_true',
        help="Skip concept/entity extraction"
    )
    parser.add_argument(
        '--diarize',
        action='store_true',
        help="Enable speaker diarization (requires HF_TOKEN)"
    )
    parser.add_argument(
        '--hf-token',
        type=str,
        default=None,
        help="HuggingFace token for diarization (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        '--num-speakers',
        type=int,
        default=None,
        help="Number of speakers if known (helps diarization accuracy)"
    )

    args = parser.parse_args()

    print(f"World Model Pipeline")
    print(f"{'=' * 40}")
    print(f"Input: {args.url}")
    print()

    # Check if this is a local file
    if is_local_file(args.url):
        print(f"Detected local file")
        print(f"[1/3] Transform: Processing local file...")
        assets = transform_local(args.url, args.output, title=args.title)
        print(f"      Audio: {assets.audio_path}")
        print(f"      Duration: {assets.duration:.0f}s")
        video_id = assets.video_id
        work_dir = args.output / video_id
    else:
        # Check if this is a live stream
        stream_type = detect_stream_type(args.url)
        if stream_type is not None:
            print(f"Detected live stream: {stream_type.value}")
            print("Routing to stream pipeline...")
            print()
            session = run_stream_pipeline(args.url, args.output)
            return session.work_dir

        video_id = extract_video_id(args.url)
        work_dir = args.output / video_id

        # Stage 1: Transform
        if args.skip_download and work_dir.exists():
            print(f"[1/3] Transform: Using existing assets in {work_dir}")
            assets = _load_assets(work_dir)
        else:
            print(f"[1/3] Transform: Downloading media...")
            assets = transform(args.url, args.output)
            print(f"      Audio: {assets.audio_path}")
            print(f"      Duration: {assets.duration:.0f}s")

    # Stage 2: Extract
    if args.skip_transcribe and (work_dir / 'transcript.json').exists():
        print(f"[2/3] Extract: Using existing transcript")
        content = ExtractedContent(video_id=video_id)
        content.transcript = load_transcript(work_dir)
        # Still extract concepts if not skipped
        if not args.skip_concepts:
            try:
                from .concepts import extract_entities, compute_cooccurrences, save_video_concepts
                print(f"      Extracting concepts...")
                concept_data = extract_entities(content.transcript)
                concept_data.video_id = video_id
                concept_data.cooccurrences = compute_cooccurrences(concept_data.entities)
                save_video_concepts(concept_data, work_dir)
                content.concepts = concept_data
                print(f"      Concepts: {len(concept_data.entities)} entities")
            except Exception as e:
                print(f"      Skipping concepts: {e}")
    else:
        print(f"[2/3] Extract: Processing content...")
        content = extract(
            assets,
            work_dir,
            extract_concepts=not args.skip_concepts,
            enable_diarization=args.diarize,
            hf_token=args.hf_token,
            num_speakers=args.num_speakers
        )

    print(f"      Transcript: {len(content.transcript)} segments")

    # Stage 3: Digest
    print(f"[3/3] Digest: Generating output...")
    digest_path = digest(assets, content, work_dir)

    # Update global concept graph
    if content.concepts:
        try:
            from .concepts import update_global_graph
            update_global_graph(content.concepts, assets.title, args.output)
            print(f"      Updated global concept graph")
        except Exception as e:
            print(f"      Warning: Failed to update global graph: {e}")

    print()
    print(f"{'=' * 40}")
    print(f"Complete! Output: {work_dir}")
    print(f"  - manifest.json   (structured data)")
    print(f"  - digest.md       (AI-readable document)")
    print(f"  - transcript.json (timestamped transcript)")
    if content.concepts:
        print(f"  - concepts.json   (entity graph)")

    return work_dir


def _load_assets(work_dir: Path):
    """Load assets from existing transform metadata."""
    import json

    meta_path = work_dir / 'transform_meta.json'
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata found at {meta_path}")

    with open(meta_path) as f:
        meta = json.load(f)

    # Create a simple object with the metadata
    class Assets:
        pass

    assets = Assets()
    for key, value in meta.items():
        if key.endswith('_path') and value:
            setattr(assets, key, Path(value))
        else:
            setattr(assets, key, value)

    return assets


if __name__ == '__main__':
    main()
