#!/usr/bin/env python3
"""
LLM-based HTML Parser using Chutes API (DeepSeek V3.2)
Processes articles asynchronously with smart DOM extraction and fallback strategies
"""

import requests
import re
import json
import time
import os
from datetime import date
from bs4 import BeautifulSoup, Comment

# API Configuration
CHUTES_ENDPOINT = "https://llm.chutes.ai/v1/chat/completions"
CHUTES_API_KEY = os.getenv("CHUTES_API_KEY", "")
CHUTES_MODEL = "deepseek-ai/DeepSeek-V3.2"  # Standard model, not Speciale
MAX_TOKENS_INPUT = 150000  # Token limit (leave room for output)
MAX_CHARS = 450000  # ~150K tokens at 3 chars/token
REQUEST_TIMEOUT = 180  # 3 minutes

# Rate limiting
DAILY_LIMIT = 300
requests_today = 0
last_reset = date.today()


class RateLimitError(Exception):
    """Raised when daily API limit is reached"""
    pass


class APIError(Exception):
    """Raised when API request fails"""
    pass


def check_rate_limit():
    """Check and update rate limit counter"""
    global requests_today, last_reset

    if date.today() != last_reset:
        requests_today = 0
        last_reset = date.today()

    if requests_today >= DAILY_LIMIT:
        raise RateLimitError(f"Daily API limit reached ({DAILY_LIMIT} requests)")

    requests_today += 1


def handle_streaming_response(response, chunk_callback=None):
    """
    Handle streaming SSE response from Chutes API

    Args:
        response: requests Response object with streaming enabled
        chunk_callback: Optional callback function(chunk_text) called for each chunk

    Returns:
        Complete content string assembled from streaming chunks
    """
    content = ""

    for line in response.iter_lines():
        if not line:
            continue

        line = line.decode('utf-8')

        # SSE format: "data: {...}"
        if line.startswith('data: '):
            data_str = line[6:]  # Remove "data: " prefix

            # Check for end marker
            if data_str == '[DONE]':
                break

            try:
                data = json.loads(data_str)
                if 'choices' in data and len(data['choices']) > 0:
                    delta = data['choices'][0].get('delta', {})
                    chunk = delta.get('content', '')
                    if chunk:  # Only append if chunk is not None
                        content += chunk
                        # Call callback with chunk if provided
                        if chunk_callback:
                            chunk_callback(chunk)
            except json.JSONDecodeError:
                continue

    return content


def preprocess_html(html: str) -> str:
    """
    Remove high-token, low-value elements from HTML
    - SVG graphics (logos, decorations)
    - Scripts and styles (non-content)
    - Comments
    """
    # Remove SVG elements (often logos/icons with many coordinates)
    html = re.sub(r'<svg[^>]*>.*?</svg>', '', html, flags=re.DOTALL | re.IGNORECASE)

    # Remove script tags
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)

    # Remove style tags
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)

    # Remove HTML comments
    html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)

    return html


def extract_article_content(html: str) -> tuple[str, str]:
    """
    Extract article content using fallback strategy:
    1. <main> tag (most modern sites)
    2. <article> tag
    3. Common article class names
    4. <body> with aggressive preprocessing
    5. If still too large: returns 'chunking_needed'

    Returns: (extracted_html, method_used)
    """
    soup = BeautifulSoup(html, 'html.parser')

    # Strategy 1: Try <main> tag
    main_tag = soup.find('main')
    if main_tag:
        content = str(main_tag)
        content = preprocess_html(content)
        if len(content) < MAX_CHARS:
            return (content, 'main_tag')
        print(f"  <main> tag found but too large ({len(content):,} chars), trying alternatives...")

    # Strategy 2: Try <article> tag
    article_tag = soup.find('article')
    if article_tag:
        content = str(article_tag)
        content = preprocess_html(content)
        if len(content) < MAX_CHARS:
            return (content, 'article_tag')
        print(f"  <article> tag found but too large ({len(content):,} chars), trying alternatives...")

    # Strategy 3: Try common article class selectors
    class_selectors = [
        'article-body', 'post-content', 'story-body', 'entry-content',
        'article-content', 'post-body', 'content-body', 'main-content'
    ]

    for selector in class_selectors:
        element = soup.find(class_=selector) or soup.find(class_=re.compile(selector, re.I))
        if element:
            content = str(element)
            content = preprocess_html(content)
            if len(content) < MAX_CHARS:
                return (content, f'class_{selector}')

    # Strategy 4: Fall back to <body> with aggressive preprocessing
    body = soup.find('body')
    if body:
        content = str(body)
        content = preprocess_html(content)

        # Additional aggressive preprocessing for body
        # Remove nav, header, footer, aside elements
        soup_body = BeautifulSoup(content, 'html.parser')
        for tag in soup_body.find_all(['nav', 'header', 'footer', 'aside']):
            tag.decompose()
        content = str(soup_body)

        if len(content) < MAX_CHARS:
            return (content, 'body_preprocessed')

    # Strategy 5: Still too large - needs chunking
    return (content if 'content' in locals() else html, 'chunking_needed')


def create_dom_chunks(html: str, max_chunk_chars=MAX_CHARS) -> list[str]:
    """
    Split HTML into chunks at DOM element boundaries
    Each chunk is valid HTML that fits within token limit
    Handles nested large elements by recursively splitting them
    """
    soup = BeautifulSoup(html, 'html.parser')

    # Find container (body or root)
    container = soup.find('body') or soup

    chunks = []
    current_chunk = []
    current_size = 0

    # Iterate through top-level children
    for child in container.children:
        if isinstance(child, str):
            # Text node
            child_size = len(child)
            child_html = child
        else:
            # Element node
            child_html = str(child)
            child_size = len(child_html)

        # If single child is too large, split it recursively
        if child_size > max_chunk_chars:
            # Flush current chunk first
            if current_chunk:
                chunks.append(''.join(current_chunk))
                current_chunk = []
                current_size = 0

            # Recursively chunk this large element
            sub_chunks = _chunk_large_element(child, max_chunk_chars)
            chunks.extend(sub_chunks)
            continue

        # If adding this child would exceed limit and we have content, start new chunk
        if current_size + child_size > max_chunk_chars and current_chunk:
            chunks.append(''.join(current_chunk))
            current_chunk = [child_html]
            current_size = child_size
        else:
            current_chunk.append(child_html)
            current_size += child_size

    # Add final chunk
    if current_chunk:
        chunks.append(''.join(current_chunk))

    return chunks


def _chunk_large_element(element, max_chunk_chars):
    """
    Recursively split a large element into smaller chunks
    Tries to split at child boundaries, falls back to paragraph splitting
    """
    if isinstance(element, str):
        # Plain text - chunk by character (worst case)
        return [element[i:i+max_chunk_chars] for i in range(0, len(element), max_chunk_chars)]

    # Get all direct children
    children = list(element.children)

    if not children:
        # No children - chunk the string representation
        element_str = str(element)
        return [element_str[i:i+max_chunk_chars] for i in range(0, len(element_str), max_chunk_chars)]

    # Try to build chunks from children
    chunks = []
    current_chunk = []
    current_size = 0

    for child in children:
        child_html = str(child) if not isinstance(child, str) else child
        child_size = len(child_html)

        # If single child is too large, recursively split it
        if child_size > max_chunk_chars:
            # Flush current
            if current_chunk:
                chunks.append(''.join(current_chunk))
                current_chunk = []
                current_size = 0

            # Recursive split
            sub_chunks = _chunk_large_element(child, max_chunk_chars)
            chunks.extend(sub_chunks)
            continue

        # Normal chunking logic
        if current_size + child_size > max_chunk_chars and current_chunk:
            chunks.append(''.join(current_chunk))
            current_chunk = [child_html]
            current_size = child_size
        else:
            current_chunk.append(child_html)
            current_size += child_size

    if current_chunk:
        chunks.append(''.join(current_chunk))

    return chunks


def process_chunk(chunk: str, chunk_num: int, total_chunks: int, retries=2) -> str:
    """Process a single chunk with appropriate prompt and retry logic"""
    prompt = f"""Extract and clean article content from this HTML fragment.

CONTEXT: This is PART {chunk_num} of {total_chunks} of a larger article.
The HTML may start/end abruptly (mid-paragraph, mid-table, etc.) - this is expected.

INSTRUCTIONS:
1. Extract ANY article content found (text, headings, images, tables, lists)
2. DO NOT discard content because it's incomplete - extract what's there
3. Remove: navigation, ads, archive wrappers, scripts, non-article junk
4. Preserve structure of extracted content
5. Return clean HTML that can be concatenated with other chunks
6. DO NOT add <html>, <head>, or <body> wrappers
7. If a paragraph/section is cut off, keep the partial content

HTML Fragment:
{chunk}"""

    if not CHUTES_API_KEY:
        raise APIError("CHUTES_API_KEY environment variable not set")

    payload = {
        "model": CHUTES_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16000,
        "temperature": 0.1,
        "stream": True
    }

    check_rate_limit()

    for attempt in range(retries):
        try:
            print(f"  Processing chunk {chunk_num}/{total_chunks} ({len(chunk):,} chars){'...' if attempt == 0 else f' (retry {attempt}/{retries-1})...'}")

            response = requests.post(
                CHUTES_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {CHUTES_API_KEY}",
                    "Content-Type": "application/json"
                },
                json=payload,
                stream=True,
                timeout=REQUEST_TIMEOUT
            )

            if response.status_code == 200:
                content = handle_streaming_response(response)

                if content:
                    # Remove markdown code blocks
                    if content.startswith('```html'):
                        content = content.split('```html\n', 1)[1].rsplit('```', 1)[0]
                    elif content.startswith('```'):
                        content = content.split('```\n', 1)[1].rsplit('```', 1)[0]

                    print(f"    ✓ Chunk {chunk_num} processed ({len(content):,} chars)")
                    return content
                else:
                    raise APIError(f"No content in streaming response for chunk {chunk_num}")
            elif response.status_code == 429:
                print(f"    Rate limited on chunk {chunk_num}, waiting 60s...")
                if attempt < retries - 1:
                    time.sleep(60)
                    continue
                raise APIError(f"Chunk {chunk_num} rate limited after retries")
            else:
                raise APIError(f"Chunk {chunk_num} API error: {response.status_code}")

        except requests.exceptions.Timeout:
            print(f"    Chunk {chunk_num} timed out")
            if attempt < retries - 1:
                time.sleep(5)
                continue
            raise APIError(f"Chunk {chunk_num} timeout after {retries} attempts")
        except requests.exceptions.RequestException as e:
            print(f"    Chunk {chunk_num} network error: {e}")
            if attempt < retries - 1:
                time.sleep(5)
                continue
            raise APIError(f"Chunk {chunk_num} network error: {e}")

    raise APIError(f"Chunk {chunk_num} failed after {retries} attempts")


def call_chutes_api(content: str, retries=3) -> str:
    """
    Send content to Chutes API (DeepSeek V3.2) with retry logic

    Args:
        content: HTML content to process
        retries: Number of retry attempts

    Returns:
        Cleaned HTML content from LLM

    Raises:
        RateLimitError: Daily limit reached
        APIError: API request failed
    """
    if not CHUTES_API_KEY:
        raise APIError("CHUTES_API_KEY environment variable not set")

    check_rate_limit()

    prompt = f"""Extract and clean the main article content from this HTML.

Remove: navigation, ads, archive.ph wrappers, scripts, non-article elements
Keep: headline, author, date, article text, images with captions, tables
Return: Clean HTML only (no markdown code blocks)

HTML:
{content}"""

    payload = {
        "model": CHUTES_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16000,
        "temperature": 0.1,
        "stream": True
    }

    for attempt in range(retries):
        try:
            print(f"  Sending request to Chutes API (attempt {attempt + 1}/{retries})...")
            response = requests.post(
                CHUTES_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {CHUTES_API_KEY}",
                    "Content-Type": "application/json"
                },
                json=payload,
                stream=True,
                timeout=REQUEST_TIMEOUT
            )

            if response.status_code == 200:
                content = handle_streaming_response(response)

                if content:
                    # Remove markdown code blocks if present
                    if content.startswith('```html'):
                        content = content.split('```html\n', 1)[1].rsplit('```', 1)[0]
                    elif content.startswith('```'):
                        content = content.split('```\n', 1)[1].rsplit('```', 1)[0]

                    print(f"  ✓ API request successful ({len(content):,} chars returned)")
                    return content
                else:
                    raise APIError("No content in streaming API response")

            elif response.status_code == 429:
                # Rate limited by API
                print(f"  Rate limited by API, waiting 60s...")
                if attempt < retries - 1:
                    time.sleep(60)
                    continue
                raise APIError("API rate limit exceeded after retries")

            else:
                error_msg = f"API returned status {response.status_code}: {response.text[:500]}"
                print(f"  {error_msg}")
                if attempt < retries - 1:
                    time.sleep(5)
                    continue
                raise APIError(error_msg)

        except requests.exceptions.Timeout:
            print(f"  Request timed out after {REQUEST_TIMEOUT}s")
            if attempt < retries - 1:
                continue
            raise APIError(f"Request timeout after {retries} attempts")

        except requests.exceptions.RequestException as e:
            print(f"  Network error: {e}")
            if attempt < retries - 1:
                time.sleep(5)
                continue
            raise APIError(f"Network error: {e}")

    raise APIError("Max retries exceeded")


def process_article(html: str) -> dict:
    """
    Main entry point for LLM article processing

    Args:
        html: Raw HTML content from article

    Returns:
        dict with keys:
            - success: bool
            - content: str (cleaned HTML)
            - method: str (extraction method used)
            - error: str (if failed)
    """
    try:
        print(f"Processing article ({len(html):,} chars)...")

        # Extract article content with fallback strategy
        extracted_html, method = extract_article_content(html)
        print(f"  Extraction method: {method} ({len(extracted_html):,} chars)")

        # Check if chunking is needed
        if method == 'chunking_needed':
            print(f"  Content too large ({len(extracted_html):,} chars), using chunking strategy...")

            # Create chunks at DOM boundaries
            chunks = create_dom_chunks(extracted_html)
            print(f"  Split into {len(chunks)} chunks")

            # Process each chunk
            processed_chunks = []
            for i, chunk in enumerate(chunks, 1):
                try:
                    processed = process_chunk(chunk, i, len(chunks))
                    processed_chunks.append(processed)
                except Exception as e:
                    print(f"  Failed to process chunk {i}: {e}")
                    return {
                        'success': False,
                        'error': f'Chunking failed at chunk {i}/{len(chunks)}: {e}',
                        'method': 'chunked'
                    }

            # Stitch chunks together
            cleaned_html = '\n\n'.join(processed_chunks)

            return {
                'success': True,
                'content': cleaned_html,
                'method': 'chunked',
                'chunks_processed': len(chunks),
                'input_size': len(extracted_html),
                'output_size': len(cleaned_html)
            }

        # Check if content is too small (likely not an article)
        if len(extracted_html) < 500:
            return {
                'success': False,
                'error': 'Extracted content too small (<500 chars), likely not an article',
                'method': method
            }

        # Send to LLM API (single request)
        cleaned_html = call_chutes_api(extracted_html)

        return {
            'success': True,
            'content': cleaned_html,
            'method': method,
            'input_size': len(extracted_html),
            'output_size': len(cleaned_html)
        }

    except RateLimitError as e:
        print(f"  Rate limit error: {e}")
        return {
            'success': False,
            'error': str(e),
            'rate_limited': True
        }

    except APIError as e:
        print(f"  API error: {e}")
        return {
            'success': False,
            'error': str(e)
        }

    except Exception as e:
        print(f"  Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': f"Unexpected error: {e}"
        }


def process_article_streaming_generator(html: str):
    """
    Generator version for real-time SSE streaming

    Args:
        html: Raw HTML content

    Yields:
        Tuples of (event_type, data) where:
            event_type: 'status', 'chunk', 'done', 'error'
            data: Event-specific data dict
    """
    try:
        yield ('status', {'message': 'Starting extraction...', 'stage': 'extract'})

        # Extract article content
        extracted_html, method = extract_article_content(html)
        yield ('status', {
            'message': f'Extracted content ({len(extracted_html):,} chars)',
            'stage': 'extract',
            'method': method
        })

        # Check if chunking needed
        if method == 'chunking_needed':
            yield ('status', {
                'message': f'Content too large, splitting into chunks...',
                'stage': 'chunking'
            })

            chunks = create_dom_chunks(extracted_html)
            yield ('status', {
                'message': f'Processing {len(chunks)} chunks...',
                'stage': 'processing',
                'total_chunks': len(chunks)
            })

            processed_chunks = []
            for i, chunk in enumerate(chunks, 1):
                try:
                    yield ('status', {
                        'message': f'Processing chunk {i}/{len(chunks)}...',
                        'stage': 'processing',
                        'current_chunk': i,
                        'total_chunks': len(chunks)
                    })

                    # Stream each chunk in real-time
                    chunk_content = ''
                    for chunk_text in process_chunk_streaming_generator(chunk, i, len(chunks)):
                        chunk_content += chunk_text
                        # Yield each character/token as it arrives
                        yield ('chunk', {'text': chunk_text, 'chunk_num': i})

                    processed_chunks.append(chunk_content)

                except Exception as e:
                    yield ('error', {'message': f'Chunk {i} failed: {str(e)}'})
                    return

            cleaned_html = '\n\n'.join(processed_chunks)

            yield ('done', {
                'content': cleaned_html,
                'method': 'chunked',
                'chunks_processed': len(chunks),
                'result': {
                    'success': True,
                    'content': cleaned_html,
                    'method': 'chunked',
                    'chunks_processed': len(chunks),
                    'input_size': len(extracted_html),
                    'output_size': len(cleaned_html)
                }
            })
            return

        if len(extracted_html) < 500:
            yield ('error', {'message': 'Content too small (<500 chars)'})
            return

        yield ('status', {'message': 'Cleaning HTML with LLM...', 'stage': 'llm'})

        # Stream LLM processing - yield chunks as they arrive from API
        accumulated_content = ''

        for chunk_text in call_chutes_api_streaming_generator(extracted_html):
            accumulated_content += chunk_text
            yield ('chunk', {'text': chunk_text})

        cleaned_html = accumulated_content

        yield ('done', {
            'content': cleaned_html,
            'method': method,
            'result': {
                'success': True,
                'content': cleaned_html,
                'method': method,
                'input_size': len(extracted_html),
                'output_size': len(cleaned_html)
            }
        })

    except RateLimitError as e:
        yield ('error', {'message': str(e), 'rate_limited': True})
    except APIError as e:
        yield ('error', {'message': str(e)})
    except Exception as e:
        yield ('error', {'message': f'Unexpected error: {str(e)}'})


def process_article_streaming(html: str, progress_callback):
    """
    Stream article processing with real-time progress updates

    Args:
        html: Raw HTML content
        progress_callback: Function(event_type, data) called with progress updates
            event_type: 'status', 'chunk', 'done', 'error'
            data: Event-specific data

    Returns:
        dict with processing results (same as process_article)
    """
    try:
        progress_callback('status', {'message': 'Starting extraction...', 'stage': 'extract'})

        # Extract article content
        extracted_html, method = extract_article_content(html)
        progress_callback('status', {
            'message': f'Extracted content ({len(extracted_html):,} chars)',
            'stage': 'extract',
            'method': method
        })

        # Check if chunking needed
        if method == 'chunking_needed':
            progress_callback('status', {
                'message': f'Content too large, splitting into chunks...',
                'stage': 'chunking'
            })

            chunks = create_dom_chunks(extracted_html)
            progress_callback('status', {
                'message': f'Processing {len(chunks)} chunks...',
                'stage': 'processing',
                'total_chunks': len(chunks)
            })

            processed_chunks = []
            for i, chunk in enumerate(chunks, 1):
                try:
                    progress_callback('status', {
                        'message': f'Processing chunk {i}/{len(chunks)}...',
                        'stage': 'processing',
                        'current_chunk': i,
                        'total_chunks': len(chunks)
                    })

                    # Process chunk with streaming
                    def chunk_content_callback(text):
                        progress_callback('chunk', {'text': text, 'chunk_num': i})

                    processed = process_chunk_streaming(chunk, i, len(chunks), chunk_content_callback)
                    processed_chunks.append(processed)

                except Exception as e:
                    progress_callback('error', {'message': f'Chunk {i} failed: {str(e)}'})
                    return {
                        'success': False,
                        'error': f'Chunking failed at chunk {i}/{len(chunks)}: {e}',
                        'method': 'chunked'
                    }

            cleaned_html = '\n\n'.join(processed_chunks)

            progress_callback('done', {
                'content': cleaned_html,
                'method': 'chunked',
                'chunks_processed': len(chunks)
            })

            return {
                'success': True,
                'content': cleaned_html,
                'method': 'chunked',
                'chunks_processed': len(chunks),
                'input_size': len(extracted_html),
                'output_size': len(cleaned_html)
            }

        if len(extracted_html) < 500:
            progress_callback('error', {'message': 'Content too small (<500 chars)'})
            return {
                'success': False,
                'error': 'Extracted content too small (<500 chars), likely not an article',
                'method': method
            }

        progress_callback('status', {'message': 'Cleaning HTML with LLM...', 'stage': 'llm'})

        # Stream LLM processing
        def llm_chunk_callback(text):
            progress_callback('chunk', {'text': text})

        cleaned_html = call_chutes_api_streaming(extracted_html, llm_chunk_callback)

        progress_callback('done', {
            'content': cleaned_html,
            'method': method
        })

        return {
            'success': True,
            'content': cleaned_html,
            'method': method,
            'input_size': len(extracted_html),
            'output_size': len(cleaned_html)
        }

    except RateLimitError as e:
        progress_callback('error', {'message': str(e), 'rate_limited': True})
        return {'success': False, 'error': str(e), 'rate_limited': True}
    except APIError as e:
        progress_callback('error', {'message': str(e)})
        return {'success': False, 'error': str(e)}
    except Exception as e:
        progress_callback('error', {'message': f'Unexpected error: {str(e)}'})
        return {'success': False, 'error': f'Unexpected error: {e}'}


def process_chunk_streaming_generator(chunk: str, chunk_num: int, total_chunks: int):
    """
    Generator version that yields chunks in real-time for chunked articles

    Yields:
        str: Each text chunk as it arrives from the streaming API
    """
    prompt = f"""Extract and clean article content from this HTML fragment.

CONTEXT: This is PART {chunk_num} of {total_chunks} of a larger article.
The HTML may start/end abruptly (mid-paragraph, mid-table, etc.) - this is expected.

INSTRUCTIONS:
1. Extract ANY article content found (text, headings, images, tables, lists)
2. DO NOT discard content because it's incomplete - extract what's there
3. Remove: navigation, ads, archive wrappers, scripts, non-article junk
4. Preserve structure of extracted content
5. Return clean HTML that can be concatenated with other chunks
6. DO NOT add <html>, <head>, or <body> wrappers
7. If a paragraph/section is cut off, keep the partial content

HTML Fragment:
{chunk}"""

    if not CHUTES_API_KEY:
        raise APIError("CHUTES_API_KEY environment variable not set")

    payload = {
        "model": CHUTES_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16000,
        "temperature": 0.1,
        "stream": True
    }

    check_rate_limit()

    for attempt in range(2):
        try:
            response = requests.post(
                CHUTES_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {CHUTES_API_KEY}",
                    "Content-Type": "application/json"
                },
                json=payload,
                stream=True,
                timeout=REQUEST_TIMEOUT
            )

            if response.status_code == 200:
                # Stream chunks in real-time
                in_code_block = False
                accumulated = ""

                for line in response.iter_lines():
                    if not line:
                        continue

                    line = line.decode('utf-8')

                    if line.startswith('data: '):
                        data_str = line[6:]

                        if data_str == '[DONE]':
                            break

                        try:
                            data = json.loads(data_str)
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                text_chunk = delta.get('content', '')
                                if text_chunk:
                                    accumulated += text_chunk

                                    # Handle markdown code block removal on the fly
                                    if not in_code_block and accumulated.endswith('```html\n'):
                                        accumulated = accumulated[:-8]
                                        in_code_block = True
                                        continue
                                    elif not in_code_block and accumulated.endswith('```\n'):
                                        accumulated = accumulated[:-4]
                                        in_code_block = True
                                        continue

                                    # Yield the chunk in real-time
                                    if not (text_chunk.startswith('```') or text_chunk == '\n' and len(accumulated) < 10):
                                        yield text_chunk

                        except json.JSONDecodeError:
                            continue

                return  # Successfully completed

            elif response.status_code == 429:
                if attempt < 1:
                    time.sleep(60)
                    continue
                raise APIError(f"Chunk {chunk_num} rate limited after retries")
            else:
                raise APIError(f"Chunk {chunk_num} API error: {response.status_code}")

        except requests.exceptions.Timeout:
            if attempt < 1:
                time.sleep(5)
                continue
            raise APIError(f"Chunk {chunk_num} timeout after retries")
        except requests.exceptions.RequestException as e:
            if attempt < 1:
                time.sleep(5)
                continue
            raise APIError(f"Chunk {chunk_num} network error: {e}")

    raise APIError(f"Chunk {chunk_num} failed after all retries")


def process_chunk_streaming(chunk: str, chunk_num: int, total_chunks: int, chunk_callback):
    """Streaming version of process_chunk (callback-based for backward compatibility)"""
    accumulated = ''
    for text_chunk in process_chunk_streaming_generator(chunk, chunk_num, total_chunks):
        accumulated += text_chunk
        if chunk_callback:
            chunk_callback(text_chunk)
    return accumulated


def call_chutes_api_streaming_generator(content: str):
    """
    Generator version that yields chunks in real-time as they arrive from Chutes API

    Yields:
        str: Each text chunk as it arrives from the streaming API
    """
    if not CHUTES_API_KEY:
        raise APIError("CHUTES_API_KEY environment variable not set")

    check_rate_limit()

    prompt = f"""Extract and clean the main article content from this HTML.

Remove: navigation, ads, archive.ph wrappers, scripts, non-article elements
Keep: headline, author, date, article text, images with captions, tables
Return: Clean HTML only (no markdown code blocks)

HTML:
{content}"""

    payload = {
        "model": CHUTES_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16000,
        "temperature": 0.1,
        "stream": True
    }

    for attempt in range(3):
        try:
            response = requests.post(
                CHUTES_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {CHUTES_API_KEY}",
                    "Content-Type": "application/json"
                },
                json=payload,
                stream=True,
                timeout=REQUEST_TIMEOUT
            )

            if response.status_code == 200:
                # Stream chunks in real-time as they arrive
                in_code_block = False
                accumulated = ""

                for line in response.iter_lines():
                    if not line:
                        continue

                    line = line.decode('utf-8')

                    # SSE format: "data: {...}"
                    if line.startswith('data: '):
                        data_str = line[6:]

                        if data_str == '[DONE]':
                            break

                        try:
                            data = json.loads(data_str)
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                chunk = delta.get('content', '')
                                if chunk:
                                    accumulated += chunk

                                    # Handle markdown code block removal on the fly
                                    # If we see the start of ```html, mark it and skip
                                    if not in_code_block and accumulated.endswith('```html\n'):
                                        # Remove the ```html\n we just added
                                        accumulated = accumulated[:-8]
                                        in_code_block = True
                                        continue
                                    elif not in_code_block and accumulated.endswith('```\n'):
                                        # Remove the ```\n we just added
                                        accumulated = accumulated[:-4]
                                        in_code_block = True
                                        continue

                                    # Yield the chunk (unless it's markdown syntax)
                                    if not (chunk.startswith('```') or chunk == '\n' and len(accumulated) < 10):
                                        yield chunk

                        except json.JSONDecodeError:
                            continue

                # Clean up trailing markdown if present
                if accumulated.endswith('```'):
                    # Don't need to yield the closing ```, already accumulated
                    pass

                return  # Successfully completed

            elif response.status_code == 429:
                if attempt < 2:
                    time.sleep(60)
                    continue
                raise APIError("API rate limit exceeded after retries")

            else:
                error_msg = f"API returned status {response.status_code}: {response.text[:500]}"
                if attempt < 2:
                    time.sleep(5)
                    continue
                raise APIError(error_msg)

        except requests.exceptions.Timeout:
            if attempt < 2:
                time.sleep(10)
                continue
            raise APIError("API request timeout after retries")
        except requests.exceptions.RequestException as e:
            if attempt < 2:
                time.sleep(5)
                continue
            raise APIError(f"Network error: {e}")

    raise APIError("API request failed after all retries")


def call_chutes_api_streaming(content: str, chunk_callback):
    """Streaming version of call_chutes_api (callback-based for backward compatibility)"""
    accumulated = ''
    for chunk in call_chutes_api_streaming_generator(content):
        accumulated += chunk
        if chunk_callback:
            chunk_callback(chunk)
    return accumulated


# For testing
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        html_file = sys.argv[1]
        with open(html_file, 'r', encoding='utf-8') as f:
            html = f.read()

        result = process_article(html)
        print("\n" + "="*60)
        print("Result:", json.dumps(result, indent=2))

        if result['success']:
            output_file = 'llm_output.html'
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result['content'])
            print(f"Saved output to: {output_file}")
