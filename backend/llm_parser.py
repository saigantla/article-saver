#!/usr/bin/env python3
"""
LLM-based HTML Parser using Chutes API (DeepSeek V3.2)
Processes articles asynchronously with smart DOM extraction and fallback strategies
"""

import requests
import re
import json
import time
from datetime import date
from bs4 import BeautifulSoup, Comment

# API Configuration
CHUTES_ENDPOINT = "https://llm.chutes.ai/v1/chat/completions"
CHUTES_API_KEY = "cpk_557af5e073424e18873db302b324e5e9.814bf52d333f5742aa78f6904ce33ba9.JP9a4vwS7YcsElLgUNef2jGjkgaKPgo7"
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

    payload = {
        "model": CHUTES_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16000,
        "temperature": 0.1,
        "stream": False
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
                timeout=REQUEST_TIMEOUT
            )

            if response.status_code == 200:
                data = response.json()
                if 'choices' in data and len(data['choices']) > 0:
                    content = data['choices'][0]['message']['content']

                    # Remove markdown code blocks
                    if content.startswith('```html'):
                        content = content.split('```html\n', 1)[1].rsplit('```', 1)[0]
                    elif content.startswith('```'):
                        content = content.split('```\n', 1)[1].rsplit('```', 1)[0]

                    print(f"    ✓ Chunk {chunk_num} processed ({len(content):,} chars)")
                    return content
                else:
                    raise APIError(f"No choices in response for chunk {chunk_num}")
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
        "stream": False
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
                timeout=REQUEST_TIMEOUT
            )

            if response.status_code == 200:
                data = response.json()
                if 'choices' in data and len(data['choices']) > 0:
                    content = data['choices'][0]['message']['content']

                    # Remove markdown code blocks if present
                    if content.startswith('```html'):
                        content = content.split('```html\n', 1)[1].rsplit('```', 1)[0]
                    elif content.startswith('```'):
                        content = content.split('```\n', 1)[1].rsplit('```', 1)[0]

                    print(f"  ✓ API request successful ({len(content):,} chars returned)")
                    return content
                else:
                    raise APIError("No choices in API response")

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
