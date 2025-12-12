#!/usr/bin/env python3
"""
Test streaming mode for Chutes LLM API
Validates the handle_streaming_response function works correctly
"""

import sys
import os
import json
import requests
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

# Import the streaming handler from llm_parser
from llm_parser import handle_streaming_response, CHUTES_ENDPOINT, CHUTES_MODEL

def test_streaming():
    print("="*80)
    print("STREAMING API TEST")
    print("="*80)

    # Get API key
    api_key = os.getenv("CHUTES_API_KEY")
    if not api_key:
        # Try loading from .env file
        env_file = Path(__file__).parent / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if line.startswith("CHUTES_API_KEY="):
                        api_key = line.split("=", 1)[1].strip().strip('"\'')
                        break

    if not api_key:
        print("ERROR: CHUTES_API_KEY not found in environment or .env file")
        return False

    print(f"\n[1/3] Configuration")
    print(f"   Endpoint: {CHUTES_ENDPOINT}")
    print(f"   Model: {CHUTES_MODEL}")
    print(f"   API Key: {api_key[:10]}...{api_key[-4:]}")
    print(f"   Stream: True")
    print(f"   Max tokens: 16000")

    # Simple test prompt
    test_html = """<html>
<head><title>Test Article</title></head>
<body>
    <nav>Navigation menu</nav>
    <article>
        <h1>Test Article Title</h1>
        <p>This is a test paragraph.</p>
        <p>Another paragraph with some content.</p>
    </article>
    <footer>Footer content</footer>
</body>
</html>"""

    prompt = f"""Extract and clean the main article content from this HTML.

Remove: navigation, footer
Keep: article content (title, paragraphs)
Return: Clean HTML only

HTML:
{test_html}"""

    payload = {
        "model": CHUTES_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16000,
        "temperature": 0.1,
        "stream": True
    }

    print(f"\n[2/3] Sending streaming request...")
    print(f"   Input size: {len(test_html)} chars")

    try:
        response = requests.post(
            CHUTES_ENDPOINT,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json=payload,
            stream=True,
            timeout=60
        )

        print(f"   Status code: {response.status_code}")

        if response.status_code != 200:
            print(f"   ERROR: {response.text[:500]}")
            return False

        print(f"\n[3/3] Processing streaming response...")
        content = handle_streaming_response(response)

        print(f"   ✓ Received {len(content)} chars")
        print(f"\n   Preview (first 300 chars):")
        print(f"   {'-'*76}")
        print(f"   {content[:300]}")
        print(f"   {'-'*76}")

        if content:
            print(f"\n✅ SUCCESS! Streaming mode working correctly.")
            return True
        else:
            print(f"\n❌ FAILED: No content received")
            return False

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        print(f"\n{'='*80}\n")

if __name__ == "__main__":
    success = test_streaming()
    sys.exit(0 if success else 1)
