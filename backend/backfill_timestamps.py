#!/usr/bin/env python3
"""
Backfill word timestamps for existing audio files
Runs Whisper on audio files that don't have metadata with word timestamps
"""

import json
import logging
from pathlib import Path
import tts_generator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

AUDIO_DIR = Path(__file__).parent.parent / "audio"


def needs_backfill(audio_file: Path) -> bool:
    """Check if an audio file needs timestamp backfill"""
    article_id = audio_file.stem  # Get filename without extension
    meta_file = AUDIO_DIR / f"{article_id}.meta.json"

    # No metadata file at all
    if not meta_file.exists():
        return True

    # Metadata exists, check if it has word_timestamps
    try:
        with open(meta_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Check if word_timestamps exists and is not empty
        word_timestamps = metadata.get('word_timestamps', [])
        if not word_timestamps or len(word_timestamps) == 0:
            logger.info(f"Article {article_id}: metadata exists but no word timestamps")
            return True

        logger.debug(f"Article {article_id}: already has {len(word_timestamps)} word timestamps")
        return False

    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Article {article_id}: corrupt metadata, needs backfill: {e}")
        return True


def backfill_timestamps(audio_file: Path):
    """Extract and save word timestamps for an audio file"""
    article_id = audio_file.stem

    logger.info(f"Article {article_id}: Extracting word timestamps from {audio_file.name}")

    try:
        # Extract word timestamps
        word_timestamps = tts_generator.extract_word_timestamps(audio_file)

        if not word_timestamps:
            logger.warning(f"Article {article_id}: No timestamps extracted")
            return False

        # Calculate duration
        with open(audio_file, 'rb') as f:
            audio_bytes = f.read()
        duration = tts_generator.estimate_audio_duration(audio_bytes)
        size = len(audio_bytes)

        # Create or update metadata
        meta_file = AUDIO_DIR / f"{article_id}.meta.json"

        metadata = {
            "version": 1,
            "article_id": int(article_id),
            "backfilled_at": tts_generator.datetime.now().isoformat(),
            "duration": duration,
            "size": size,
            "voice": "af_heart",  # Assumed
            "word_timestamps": word_timestamps,
            "word_count": len(word_timestamps)
        }

        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Article {article_id}: ✓ Saved metadata with {len(word_timestamps)} word timestamps")
        return True

    except Exception as e:
        logger.error(f"Article {article_id}: Failed to backfill timestamps: {e}")
        return False


def backfill_all():
    """Backfill timestamps for all audio files that need it"""

    # Ensure audio directory exists
    if not AUDIO_DIR.exists():
        logger.error(f"Audio directory not found: {AUDIO_DIR}")
        return

    # Find all WAV files
    audio_files = list(AUDIO_DIR.glob("*.wav"))

    if not audio_files:
        logger.info("No audio files found")
        return

    logger.info(f"Found {len(audio_files)} audio files")

    # Check which need backfill
    files_needing_backfill = [f for f in audio_files if needs_backfill(f)]

    if not files_needing_backfill:
        logger.info("All audio files already have word timestamps!")
        return

    logger.info(f"Found {len(files_needing_backfill)} files needing timestamp backfill")

    # Backfill each file
    success_count = 0
    for i, audio_file in enumerate(files_needing_backfill, 1):
        logger.info(f"\n[{i}/{len(files_needing_backfill)}] Processing {audio_file.name}...")

        if backfill_timestamps(audio_file):
            success_count += 1

    logger.info(f"\n✓ Backfill complete: {success_count}/{len(files_needing_backfill)} successful")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Backfill specific article ID
        article_id = sys.argv[1]
        audio_file = AUDIO_DIR / f"{article_id}.wav"

        if not audio_file.exists():
            logger.error(f"Audio file not found: {audio_file}")
            sys.exit(1)

        if needs_backfill(audio_file):
            logger.info(f"Backfilling timestamps for article {article_id}")
            success = backfill_timestamps(audio_file)
            sys.exit(0 if success else 1)
        else:
            logger.info(f"Article {article_id} already has word timestamps")
            sys.exit(0)
    else:
        # Backfill all files
        backfill_all()
