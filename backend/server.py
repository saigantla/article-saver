#!/usr/bin/env python3
"""
Article Saver - Backend Server
- Receives HTML from bookmarklet
- Extracts with Readability.js
- Saves to database (SQLite) and files
- Serves frontend and API
"""

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import subprocess
import json
import sqlite3
import os
import threading
from datetime import datetime
from pathlib import Path
import re
from urllib.parse import urlparse
import llm_parser

app = Flask(__name__)
CORS(app)

# Configuration
BASE_DIR = Path(__file__).parent.parent
BACKEND_DIR = BASE_DIR / "backend"
FRONTEND_DIR = BASE_DIR / "frontend"
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "articles.db"
HTML_DIR = DATA_DIR / "html"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
HTML_DIR.mkdir(exist_ok=True)

# Initialize database
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            author TEXT,
            source_domain TEXT,
            archive_url TEXT NOT NULL,
            original_url TEXT,
            text_content TEXT,
            html_content TEXT,
            snippet TEXT,
            text_length INTEGER,
            read_time TEXT,
            html_file TEXT,
            captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processed BOOLEAN DEFAULT TRUE,
            parser_results TEXT,
            llm_content TEXT,
            llm_status TEXT DEFAULT 'pending'
        )
    ''')

    # Migration: Add llm columns if they don't exist
    try:
        c.execute("SELECT llm_content FROM articles LIMIT 1")
    except sqlite3.OperationalError:
        print("Migrating database: Adding llm_content and llm_status columns...")
        c.execute("ALTER TABLE articles ADD COLUMN llm_content TEXT")
        c.execute("ALTER TABLE articles ADD COLUMN llm_status TEXT DEFAULT 'pending'")
        print("Migration complete!")

    conn.commit()
    conn.close()

init_db()

def slugify(text):
    """Convert text to filename-safe slug"""
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    text = text.strip('_')
    return text[:50]  # Max 50 chars

def extract_domain(url):
    """Extract domain from URL"""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.replace('www.', '')
        return domain
    except:
        return 'unknown'

def calculate_read_time(text_length):
    """Estimate reading time (200 words per minute, avg 5 chars per word)"""
    words = text_length / 5
    minutes = max(1, round(words / 200))
    return f"{minutes}m"

def generate_snippet(text, max_length=300):
    """Generate preview snippet from text"""
    text = text.strip()
    if len(text) <= max_length:
        return text
    # Try to cut at sentence boundary
    snippet = text[:max_length]
    last_period = snippet.rfind('.')
    if last_period > max_length * 0.7:  # If period is not too early
        return snippet[:last_period + 1]
    return snippet + "..."

# LLM Processing helpers
def update_llm_status(article_id, status, error=None):
    """Update LLM processing status in database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if error:
        c.execute('UPDATE articles SET llm_status = ?, llm_content = ? WHERE id = ?',
                  (status, f"Error: {error}", article_id))
    else:
        c.execute('UPDATE articles SET llm_status = ? WHERE id = ?', (status, article_id))
    conn.commit()
    conn.close()

def store_llm_content(article_id, content):
    """Store LLM-processed content in database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('UPDATE articles SET llm_content = ?, llm_status = ? WHERE id = ?',
              (content, 'completed', article_id))
    conn.commit()
    conn.close()

def process_article_llm(article_id, html):
    """Background task: Process article with LLM"""
    try:
        print(f"\n🤖 Starting LLM processing for article {article_id}...")
        update_llm_status(article_id, 'processing')

        # Process with LLM
        result = llm_parser.process_article(html)

        if result['success']:
            print(f"✅ LLM processing complete for article {article_id}")
            print(f"   Method: {result['method']}")
            print(f"   Input size: {result.get('input_size', 'N/A'):,} chars")
            print(f"   Output size: {result.get('output_size', 'N/A'):,} chars")
            store_llm_content(article_id, result['content'])
        elif result.get('rate_limited'):
            print(f"⏸️  Rate limited, article {article_id} will retry later")
            update_llm_status(article_id, 'pending')
        else:
            error = result.get('error', 'Unknown error')
            print(f"❌ LLM processing failed for article {article_id}: {error}")
            update_llm_status(article_id, 'failed', error)

    except Exception as e:
        print(f"❌ Exception in LLM processor for article {article_id}: {e}")
        import traceback
        traceback.print_exc()
        update_llm_status(article_id, 'failed', str(e))

@app.route('/')
def index():
    """Serve frontend"""
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/sw.js')
def service_worker():
    """Serve service worker with proper MIME type"""
    response = send_from_directory(FRONTEND_DIR, 'sw.js')
    response.headers['Content-Type'] = 'application/javascript'
    response.headers['Service-Worker-Allowed'] = '/'
    return response

@app.route('/db.js')
def database_helper():
    """Serve IndexedDB helper with proper MIME type"""
    response = send_from_directory(FRONTEND_DIR, 'db.js')
    response.headers['Content-Type'] = 'application/javascript'
    return response

def process_article_parsers(article_id, html, html_file):
    """Background task: Run all parsers and update database"""
    try:
        print(f"🔄 Starting parser processing for article {article_id}...")

        # Run all parsers via parser-manager (10 minute timeout since this is async)
        result = subprocess.run(
            ['node', str(BACKEND_DIR / 'parser-manager.js')],
            input=html,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(BASE_DIR)
        )

        if result.returncode != 0:
            error_msg = result.stderr or "Extraction failed"
            print(f"❌ Parser manager failed for article {article_id}: {error_msg}")
            return

        all_results = json.loads(result.stdout)
        extracted = all_results['results'].get(all_results['default'], {})
        parser_results = all_results['results']

        if not extracted.get('success'):
            print(f"❌ Parser extraction failed for article {article_id}")
            return

        # Extract parsed data
        title = extracted.get('title', 'Untitled')
        author = extracted.get('byline', None)
        text_content = extracted.get('textContent', '')
        html_content = extracted.get('htmlContent', '')
        text_length = extracted.get('fullTextLength', 0)
        read_time = calculate_read_time(text_length)
        snippet = generate_snippet(text_content)

        # Update database with parsed results
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            UPDATE articles
            SET title = ?, author = ?, text_content = ?, html_content = ?,
                snippet = ?, text_length = ?, read_time = ?, parser_results = ?
            WHERE id = ?
        ''', (title, author, text_content, html_content, snippet,
              text_length, read_time, json.dumps(parser_results), article_id))
        conn.commit()
        conn.close()

        print(f"✅ Parser processing complete for article {article_id}")
        print(f"   Title: {title}")
        print(f"   Author: {author or 'N/A'}")
        print(f"   Text length: {text_length:,} characters")
        print(f"   Read time: {read_time}")

    except Exception as e:
        print(f"❌ Exception in parser processor for article {article_id}: {e}")
        import traceback
        traceback.print_exc()


@app.route('/save', methods=['POST'])
def save_article():
    """
    Main endpoint: receives HTML, saves immediately, processes parsers async
    """
    data = request.json
    html = data.get('html', '')
    archive_url = data.get('archive_url', '')
    original_url = data.get('original_url', archive_url)
    force_overwrite = data.get('force_overwrite', False)

    print(f"\n{'='*80}")
    print(f"📝 Saving article from: {archive_url}")
    print(f"   HTML size: {len(html):,} characters")
    print(f"   Force overwrite: {force_overwrite}")
    print(f"{'='*80}")

    # Check for duplicate URL
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT id, title FROM articles WHERE archive_url = ?', (archive_url,))
    existing = c.fetchone()
    conn.close()

    if existing and not force_overwrite:
        print(f"⚠️  Duplicate URL found! Existing article ID: {existing['id']}")
        return jsonify({
            "success": False,
            "duplicate": True,
            "existing_id": existing['id'],
            "existing_title": existing['title']
        }), 409

    try:
        # Quick extraction of title from HTML for immediate response
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        title_tag = soup.find('title')
        title = title_tag.get_text().strip() if title_tag else 'Untitled'

        # Calculate basic fields
        source_domain = extract_domain(original_url or archive_url)

        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        slug = slugify(title)
        filename = f"{timestamp}_{slug}.html"
        filepath = HTML_DIR / filename

        # Save HTML file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)

        # Save to database with minimal data (parsers will update later)
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        if existing and force_overwrite:
            # Update existing article
            article_id = existing['id']
            print(f"🔄 Overwriting existing article ID: {article_id}")

            # Delete old HTML file
            old_html_file = c.execute('SELECT html_file FROM articles WHERE id = ?', (article_id,)).fetchone()[0]
            if old_html_file:
                old_filepath = HTML_DIR / old_html_file
                if old_filepath.exists():
                    old_filepath.unlink()

            c.execute('''
                UPDATE articles
                SET title = ?, source_domain = ?, original_url = ?, html_file = ?,
                    captured_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (title, source_domain, original_url, filename, article_id))
        else:
            # Insert new article with minimal data
            c.execute('''
                INSERT INTO articles
                (title, source_domain, archive_url, original_url, html_file)
                VALUES (?, ?, ?, ?, ?)
            ''', (title, source_domain, archive_url, original_url, filename))
            article_id = c.lastrowid

        conn.commit()
        conn.close()

        print(f"✅ Article saved!")
        print(f"   ID: {article_id}")
        print(f"   Title: {title}")
        print(f"   Domain: {source_domain}")
        print(f"   File: {filename}")
        print(f"{'='*80}\n")

        # Queue async parser processing in background
        threading.Thread(
            target=process_article_parsers,
            args=(article_id, html, filename),
            daemon=True,
            name=f"Parser-{article_id}"
        ).start()
        print(f"🚀 Queued parser processing for article {article_id}")

        # Queue async LLM processing in background
        threading.Thread(
            target=process_article_llm,
            args=(article_id, html),
            daemon=True,
            name=f"LLM-{article_id}"
        ).start()
        print(f"🚀 Queued LLM processing for article {article_id}")

        return jsonify({
            "success": True,
            "id": article_id,
            "title": title,
            "message": "Article saved! Processing in background..."
        })
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/articles', methods=['GET'])
def list_articles():
    """List all saved articles with metadata for library view"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('''
        SELECT id, title, author, source_domain, archive_url, original_url,
               snippet, text_length, read_time, captured_at
        FROM articles
        ORDER BY captured_at DESC
    ''')
    rows = c.fetchall()
    conn.close()

    articles = []
    for row in rows:
        articles.append({
            'id': row['id'],
            'title': row['title'],
            'author': row['author'],
            'source': row['original_url'] or row['archive_url'],
            'source_domain': row['source_domain'],
            'archive_url': row['archive_url'],
            'snippet': row['snippet'],
            'text_length': row['text_length'],
            'readTime': row['read_time'],
            'captured_at': row['captured_at']
        })

    return jsonify(articles)

@app.route('/articles/<int:article_id>', methods=['GET'])
def get_article(article_id):
    """Get full article content for reader view"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM articles WHERE id = ?', (article_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return jsonify({"error": "Article not found"}), 404

    # Parse parser_results from JSON if available
    parser_results = None
    if row['parser_results']:
        try:
            parser_results = json.loads(row['parser_results'])
        except:
            parser_results = None

    return jsonify({
        'id': row['id'],
        'title': row['title'],
        'author': row['author'],
        'source': row['original_url'] or row['archive_url'],
        'source_domain': row['source_domain'],
        'archive_url': row['archive_url'],
        'content': row['html_content'],
        'text_length': row['text_length'],
        'readTime': row['read_time'],
        'captured_at': row['captured_at'],
        'parser_results': parser_results,
        'llm_content': row['llm_content'],
        'llm_status': row['llm_status']
    })

@app.route('/articles/<int:article_id>/html', methods=['GET'])
def get_article_html(article_id):
    """Serve original HTML file"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT html_file FROM articles WHERE id = ?', (article_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return jsonify({"error": "Article not found"}), 404

    filepath = HTML_DIR / row[0]
    if not filepath.exists():
        return jsonify({"error": "HTML file not found"}), 404

    return send_file(filepath, mimetype='text/html')

@app.route('/articles/<int:article_id>', methods=['DELETE'])
def delete_article(article_id):
    """Delete article and its HTML file"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Get HTML filename
    c.execute('SELECT html_file FROM articles WHERE id = ?', (article_id,))
    row = c.fetchone()

    if row:
        # Delete HTML file
        filepath = HTML_DIR / row[0]
        if filepath.exists():
            filepath.unlink()

        # Delete database entry
        c.execute('DELETE FROM articles WHERE id = ?', (article_id,))
        conn.commit()

    conn.close()
    return jsonify({"success": True})

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM articles')
    count = c.fetchone()[0]
    conn.close()

    return jsonify({
        "status": "ok",
        "message": "Article Saver running",
        "articles_count": count
    })

@app.route('/articles/<int:article_id>/reprocess-llm', methods=['POST'])
def reprocess_llm(article_id):
    """Manually trigger LLM processing for an article (useful for failed/pending articles)"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Get article HTML file
    c.execute('SELECT html_file FROM articles WHERE id = ?', (article_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return jsonify({"success": False, "error": "Article not found"}), 404

    # Read HTML file
    filepath = HTML_DIR / row[0]
    if not filepath.exists():
        return jsonify({"success": False, "error": "HTML file not found"}), 404

    with open(filepath, 'r', encoding='utf-8') as f:
        html = f.read()

    # Queue LLM processing
    threading.Thread(
        target=process_article_llm,
        args=(article_id, html),
        daemon=True,
        name=f"LLM-Reprocess-{article_id}"
    ).start()

    return jsonify({"success": True, "message": f"LLM processing queued for article {article_id}"})

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 Article Saver - Backend Server")
    print("="*80)
    print("Server running at: http://localhost:3000")
    print("Frontend: http://localhost:3000/")
    print("Health check: http://localhost:3000/health")
    print("\nAPI Endpoints:")
    print("  POST   /save              - Save article from bookmarklet")
    print("  GET    /articles          - List all articles")
    print("  GET    /articles/:id      - Get article content")
    print("  GET    /articles/:id/html - View original HTML")
    print("  DELETE /articles/:id      - Delete article")
    print("\nDatabase:", DB_PATH)
    print("HTML files:", HTML_DIR)
    print("="*80 + "\n")

    app.run(host='0.0.0.0', port=3000, debug=True)
