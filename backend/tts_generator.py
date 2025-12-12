#!/usr/bin/env python3
"""
Text-to-Speech Generator using Kokoro via Chutes AI
Converts article content to natural-sounding audio
"""

import requests
import os
import time
import logging
from pathlib import Path
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# API Configuration
CHUTES_API_TOKEN = os.getenv("CHUTES_API_KEY", "")
TTS_ENDPOINT = "https://chutes-kokoro.chutes.ai/speak"
DEFAULT_VOICE = "af_heart"  # American Female
REQUEST_TIMEOUT = 300  # 5 minutes for TTS generation

# Audio storage
AUDIO_DIR = Path(__file__).parent.parent / "audio"


class TTSError(Exception):
    """Raised when TTS generation fails"""
    pass


def ensure_audio_directory():
    """Ensure the audio directory exists"""
    AUDIO_DIR.mkdir(exist_ok=True)
    logger.info(f"Audio directory: {AUDIO_DIR}")


def extract_text_from_html(html_content: str) -> str:
    """
    Extract plain text from HTML content for TTS

    Args:
        html_content: HTML string from parser

    Returns:
        Plain text suitable for TTS
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove script and style elements
    for element in soup(['script', 'style', 'nav', 'footer', 'aside']):
        element.decompose()

    # Get text with proper spacing
    text = soup.get_text(separator=' ', strip=True)

    # Clean up whitespace
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = ' '.join(lines)

    # Limit length (Kokoro handles ~1000 chars well based on tests)
    # For longer content, we'll need to chunk
    MAX_CHARS = 50000  # Conservative limit
    if len(text) > MAX_CHARS:
        logger.warning(f"Text truncated from {len(text)} to {MAX_CHARS} chars")
        text = text[:MAX_CHARS]

    return text


def call_tts_api(text: str, voice: str = DEFAULT_VOICE, speed: float = 1.0) -> bytes:
    """
    Call Kokoro TTS API to generate audio

    Args:
        text: Text to convert to speech
        voice: Voice ID (default: af_heart)
        speed: Speech speed multiplier (default: 1.0)

    Returns:
        WAV audio bytes

    Raises:
        TTSError: If API call fails
    """
    if not CHUTES_API_TOKEN:
        raise TTSError("CHUTES_API_KEY environment variable not set")

    headers = {
        "Authorization": f"Bearer {CHUTES_API_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "text": text,
        "voice": voice,
        "speed": speed
    }

    logger.info(f"Generating TTS: {len(text)} chars, voice={voice}, speed={speed}")

    try:
        response = requests.post(
            TTS_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT
        )

        if response.status_code == 200:
            audio_bytes = response.content
            logger.info(f"TTS generated: {len(audio_bytes):,} bytes")
            return audio_bytes
        else:
            error_msg = f"TTS API returned {response.status_code}: {response.text[:200]}"
            logger.error(error_msg)
            raise TTSError(error_msg)

    except requests.exceptions.Timeout:
        raise TTSError("TTS generation timed out (>5 minutes)")
    except requests.exceptions.RequestException as e:
        raise TTSError(f"TTS API request failed: {e}")


def estimate_audio_duration(audio_bytes: bytes) -> float:
    """
    Estimate audio duration from WAV file bytes

    Args:
        audio_bytes: WAV file bytes

    Returns:
        Duration in seconds
    """
    # WAV format: 44 byte header, then 24kHz 16-bit mono = 48000 bytes/sec
    audio_data_size = len(audio_bytes) - 44
    if audio_data_size <= 0:
        return 0.0

    duration = audio_data_size / 48000.0
    return duration


def generate_audio_for_article(article_id: int, html_content: str,
                               voice: str = DEFAULT_VOICE) -> dict:
    """
    Generate TTS audio for an article

    Args:
        article_id: Article database ID
        html_content: HTML content from readability parser
        voice: Voice to use (default: af_heart)

    Returns:
        dict with audio metadata: {
            'success': bool,
            'file_path': str,
            'duration': float,
            'size': int,
            'text_length': int,
            'voice': str,
            'error': str (if failed)
        }
    """
    ensure_audio_directory()

    result = {
        'success': False,
        'voice': voice
    }

    try:
        # Extract text from HTML
        text = extract_text_from_html(html_content)

        if not text:
            result['error'] = "No text extracted from HTML"
            logger.warning(f"Article {article_id}: No text to convert")
            return result

        result['text_length'] = len(text)
        logger.info(f"Article {article_id}: Extracted {len(text)} chars for TTS")

        # Generate audio
        audio_bytes = call_tts_api(text, voice=voice)

        # Save to file
        audio_filename = f"{article_id}.wav"
        audio_path = AUDIO_DIR / audio_filename

        with open(audio_path, 'wb') as f:
            f.write(audio_bytes)

        # Calculate metadata
        duration = estimate_audio_duration(audio_bytes)
        size = len(audio_bytes)

        result.update({
            'success': True,
            'file_path': str(audio_path),
            'relative_path': f"audio/{audio_filename}",
            'duration': duration,
            'size': size
        })

        logger.info(f"Article {article_id}: Audio saved to {audio_path} "
                   f"({size:,} bytes, {duration:.1f}s)")

        return result

    except TTSError as e:
        result['error'] = str(e)
        logger.error(f"Article {article_id}: TTS failed - {e}")
        return result
    except Exception as e:
        result['error'] = f"Unexpected error: {e}"
        logger.error(f"Article {article_id}: Unexpected error - {e}")
        return result


def get_audio_path(article_id: int) -> Path:
    """Get the audio file path for an article"""
    return AUDIO_DIR / f"{article_id}.wav"


def audio_exists(article_id: int) -> bool:
    """Check if audio file exists for an article"""
    return get_audio_path(article_id).exists()


def delete_audio(article_id: int) -> bool:
    """Delete audio file for an article"""
    audio_path = get_audio_path(article_id)
    if audio_path.exists():
        audio_path.unlink()
        logger.info(f"Deleted audio for article {article_id}")
        return True
    return False


# For testing
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python tts_generator.py <html_file>")
        sys.exit(1)

    html_file = sys.argv[1]
    with open(html_file, 'r', encoding='utf-8') as f:
        html = f.read()

    print(f"Generating TTS for {html_file}...")
    result = generate_audio_for_article(999, html)

    if result['success']:
        print(f"✅ Success!")
        print(f"   File: {result['file_path']}")
        print(f"   Duration: {result['duration']:.1f}s")
        print(f"   Size: {result['size']:,} bytes")
    else:
        print(f"❌ Failed: {result.get('error')}")
