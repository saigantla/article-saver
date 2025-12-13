#!/usr/bin/env python3
"""
Text-to-Speech Generator using Kokoro via Chutes AI
Converts article content to natural-sounding audio
"""

import requests
import os
import time
import logging
import struct
import re
from pathlib import Path
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# API Configuration
CHUTES_API_TOKEN = os.getenv("CHUTES_API_KEY", "")
TTS_ENDPOINT = "https://chutes-kokoro.chutes.ai/speak"
DEFAULT_VOICE = "af_heart"  # American Female
REQUEST_TIMEOUT = 60  # 1 minute timeout per chunk
MAX_RETRIES = 3  # Retry up to 3 times on failure
CHUNK_SIZE = 8000  # Characters per API call to avoid 502 errors

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

    return text


def call_tts_api(text: str, voice: str = DEFAULT_VOICE, speed: float = 1.0) -> bytes:
    """
    Call Kokoro TTS API to generate audio with retry logic

    Args:
        text: Text to convert to speech
        voice: Voice ID (default: af_heart)
        speed: Speech speed multiplier (default: 1.0)

    Returns:
        WAV audio bytes

    Raises:
        TTSError: If API call fails after all retries
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

    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            if attempt > 0:
                # Exponential backoff: 2s, 4s, 8s
                backoff = 2 ** attempt
                logger.info(f"Retry attempt {attempt + 1}/{MAX_RETRIES} after {backoff}s backoff")
                time.sleep(backoff)

            logger.info(f"Generating TTS: {len(text)} chars, voice={voice}, speed={speed} (attempt {attempt + 1}/{MAX_RETRIES})")

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
                logger.warning(f"Attempt {attempt + 1} failed: {error_msg}")
                last_error = TTSError(error_msg)

                # Don't retry on client errors (4xx), only server errors (5xx) and timeouts
                if 400 <= response.status_code < 500:
                    raise last_error

        except requests.exceptions.Timeout:
            error_msg = f"TTS generation timed out (>{REQUEST_TIMEOUT}s)"
            logger.warning(f"Attempt {attempt + 1} failed: {error_msg}")
            last_error = TTSError(error_msg)

        except requests.exceptions.RequestException as e:
            error_msg = f"TTS API request failed: {e}"
            logger.warning(f"Attempt {attempt + 1} failed: {error_msg}")
            last_error = TTSError(error_msg)

    # All retries exhausted
    final_error = f"TTS failed after {MAX_RETRIES} attempts: {last_error}"
    logger.error(final_error)
    raise TTSError(final_error)


def chunk_text(text: str, max_chunk_size: int = CHUNK_SIZE) -> list[str]:
    """
    Split text into chunks at sentence boundaries to avoid cutting mid-sentence

    Args:
        text: Text to chunk
        max_chunk_size: Maximum characters per chunk

    Returns:
        List of text chunks
    """
    # If text is small enough, return as-is
    if len(text) <= max_chunk_size:
        return [text]

    chunks = []
    # Split on sentence boundaries (., !, ?) followed by space
    sentences = re.split(r'([.!?]+\s+)', text)

    current_chunk = ""
    for i in range(0, len(sentences), 2):
        sentence = sentences[i]
        # Include the punctuation/space if it exists
        if i + 1 < len(sentences):
            sentence += sentences[i + 1]

        # If adding this sentence would exceed chunk size and we have content, save chunk
        if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += sentence

    # Add remaining chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    logger.info(f"Split {len(text)} chars into {len(chunks)} chunks")
    return chunks


def concatenate_wav_files(wav_bytes_list: list[bytes]) -> bytes:
    """
    Concatenate multiple WAV files into a single WAV file

    Args:
        wav_bytes_list: List of WAV file bytes to concatenate

    Returns:
        Single WAV file bytes
    """
    if len(wav_bytes_list) == 1:
        return wav_bytes_list[0]

    # Extract audio data from each WAV file (skip 44-byte header)
    audio_data_chunks = []
    for wav_bytes in wav_bytes_list:
        if len(wav_bytes) > 44:
            audio_data_chunks.append(wav_bytes[44:])

    # Concatenate all audio data
    combined_audio_data = b''.join(audio_data_chunks)
    total_audio_size = len(combined_audio_data)

    # Build new WAV header
    # WAV format: RIFF header + fmt chunk + data chunk
    wav_header = bytearray(44)

    # RIFF header
    wav_header[0:4] = b'RIFF'
    wav_header[4:8] = struct.pack('<I', 36 + total_audio_size)  # File size - 8
    wav_header[8:12] = b'WAVE'

    # fmt chunk
    wav_header[12:16] = b'fmt '
    wav_header[16:20] = struct.pack('<I', 16)  # fmt chunk size
    wav_header[20:22] = struct.pack('<H', 1)   # Audio format (1 = PCM)
    wav_header[22:24] = struct.pack('<H', 1)   # Channels (1 = mono)
    wav_header[24:28] = struct.pack('<I', 24000)  # Sample rate (24kHz)
    wav_header[28:32] = struct.pack('<I', 48000)  # Byte rate (24000 * 2)
    wav_header[32:34] = struct.pack('<H', 2)   # Block align
    wav_header[34:36] = struct.pack('<H', 16)  # Bits per sample

    # data chunk
    wav_header[36:40] = b'data'
    wav_header[40:44] = struct.pack('<I', total_audio_size)

    # Combine header and audio data
    return bytes(wav_header) + combined_audio_data


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

        # Chunk text if needed to avoid API 502 errors
        text_chunks = chunk_text(text)

        # Generate audio for each chunk in parallel
        def generate_chunk(index: int, chunk_text: str):
            """Helper to generate audio for a single chunk"""
            logger.info(f"Article {article_id}: Generating chunk {index+1}/{len(text_chunks)} ({len(chunk_text)} chars)")
            audio = call_tts_api(chunk_text, voice=voice)
            return (index, audio)

        # Use ThreadPoolExecutor for parallel requests (max 4 concurrent to avoid overwhelming API)
        max_workers = min(4, len(text_chunks))
        audio_chunks_dict = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all chunks
            futures = {
                executor.submit(generate_chunk, i, chunk): i
                for i, chunk in enumerate(text_chunks)
            }

            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    index, audio = future.result()
                    audio_chunks_dict[index] = audio
                    logger.info(f"Article {article_id}: Chunk {index+1}/{len(text_chunks)} completed")
                except Exception as e:
                    chunk_idx = futures[future]
                    logger.error(f"Article {article_id}: Chunk {chunk_idx+1} failed - {e}")
                    raise

        # Sort chunks by index to maintain correct order
        audio_chunks = [audio_chunks_dict[i] for i in sorted(audio_chunks_dict.keys())]

        # Concatenate audio chunks if multiple
        if len(audio_chunks) > 1:
            logger.info(f"Article {article_id}: Concatenating {len(audio_chunks)} audio chunks")
            audio_bytes = concatenate_wav_files(audio_chunks)
        else:
            audio_bytes = audio_chunks[0]

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
            'size': size,
            'chunks': len(text_chunks)
        })

        logger.info(f"Article {article_id}: Audio saved to {audio_path} "
                   f"({size:,} bytes, {duration:.1f}s, {len(text_chunks)} chunks)")

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
