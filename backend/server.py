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
import logging
from datetime import datetime
from pathlib import Path
import re
from urllib.parse import urlparse
from contextlib import contextmanager
from typing import Optional, Dict, Any, Tuple
import llm_parser

# =============================================================================
# Configuration
# =============================================================================

app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
BACKEND_DIR = BASE_DIR / "backend"
FRONTEND_DIR = BASE_DIR / "frontend"
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "articles.db"
HTML_DIR = DATA_DIR / "html"
MIGRATIONS_FLAG = DATA_DIR / ".migrations_applied"

# Constants
PARSER_TIMEOUT = 600  # 10 minutes for full parsing
SINGLE_PARSER_TIMEOUT = 120  # 2 minutes for single parser
VALID_PARSERS = ['readability', 'defuddle', 'postlight', 'llm']
VALID_STATUS_VALUES = ['read', 'unread']
MAX_HTML_SIZE = 50 * 1024 * 1024  # 50MB
MAX_TITLE_LENGTH = 500
MAX_URL_LENGTH = 2000

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
HTML_DIR.mkdir(exist_ok=True)

# =============================================================================
# Database Utilities
# =============================================================================

@contextmanager
def get_db_connection():
    """Context manager for database connections with proper error handling"""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        yield conn
        conn.commit()
    except sqlite3.Error as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            conn.close()


def init_db():
    """Initialize database with schema"""
    with get_db_connection() as conn:
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
                llm_status TEXT DEFAULT 'pending',
                status TEXT DEFAULT 'unread',
                topics TEXT DEFAULT '[]'
            )
        ''')
        logger.info("Database initialized")


def run_migrations():
    """Run database migrations only once"""
    if MIGRATIONS_FLAG.exists():
        logger.info("Migrations already applied, skipping")
        return

    logger.info("Running database migrations...")

    with get_db_connection() as conn:
        c = conn.cursor()

        # Migration: Add llm columns if they don't exist
        try:
            c.execute("SELECT llm_content FROM articles LIMIT 1")
        except sqlite3.OperationalError:
            logger.info("Adding llm_content and llm_status columns")
            c.execute("ALTER TABLE articles ADD COLUMN llm_content TEXT")
            c.execute("ALTER TABLE articles ADD COLUMN llm_status TEXT DEFAULT 'pending'")

        # Migration: Add status and topics columns
        try:
            c.execute("SELECT status FROM articles LIMIT 1")
        except sqlite3.OperationalError:
            logger.info("Adding status and topics columns")
            c.execute("ALTER TABLE articles ADD COLUMN status TEXT DEFAULT 'unread'")
            c.execute("ALTER TABLE articles ADD COLUMN topics TEXT DEFAULT '[]'")

    # Mark migrations as complete
    MIGRATIONS_FLAG.touch()
    logger.info("Migrations complete")


# =============================================================================
# Utility Functions
# =============================================================================

def slugify(text: str) -> str:
    """Convert text to filename-safe slug"""
    if not text:
        return "untitled"
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    text = text.strip('_')
    return text[:50]  # Max 50 chars


def extract_domain(url: str) -> str:
    """Extract domain from URL"""
    if not url:
        return 'unknown'
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.replace('www.', '')
        return domain if domain else 'unknown'
    except Exception as e:
        logger.warning(f"Failed to parse URL '{url}': {e}")
        return 'unknown'


def calculate_read_time(text_length: int) -> str:
    """Estimate reading time (200 words per minute, avg 5 chars per word)"""
    if not text_length or text_length <= 0:
        return "1m"
    words = text_length / 5
    minutes = max(1, round(words / 200))
    return f"{minutes}m"


def generate_snippet(text: str, max_length: int = 300) -> str:
    """Generate preview snippet from text"""
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_length:
        return text
    # Try to cut at sentence boundary
    snippet = text[:max_length]
    last_period = snippet.rfind('.')
    if last_period > max_length * 0.7:  # If period is not too early
        return snippet[:last_period + 1]
    return snippet + "..."


# =============================================================================
# Input Validation
# =============================================================================

def validate_html_input(html: str) -> Tuple[bool, Optional[str]]:
    """Validate HTML input"""
    if not html:
        return False, "HTML content is required"
    if not isinstance(html, str):
        return False, "HTML must be a string"
    if len(html) > MAX_HTML_SIZE:
        return False, f"HTML size exceeds maximum of {MAX_HTML_SIZE} bytes"
    return True, None


def validate_url_input(url: str) -> Tuple[bool, Optional[str]]:
    """Validate URL input"""
    if not url:
        return False, "URL is required"
    if not isinstance(url, str):
        return False, "URL must be a string"
    if len(url) > MAX_URL_LENGTH:
        return False, f"URL exceeds maximum length of {MAX_URL_LENGTH}"
    # Basic URL validation
    if not url.startswith(('http://', 'https://')):
        return False, "URL must start with http:// or https://"
    return True, None


def validate_status(status: str) -> Tuple[bool, Optional[str]]:
    """Validate article status"""
    if status not in VALID_STATUS_VALUES:
        return False, f"Status must be one of {VALID_STATUS_VALUES}"
    return True, None


def validate_topics(topics: list) -> Tuple[bool, Optional[str]]:
    """Validate topics list"""
    if not isinstance(topics, list):
        return False, "Topics must be an array"
    if len(topics) > 50:
        return False, "Maximum 50 topics allowed"
    for topic in topics:
        if not isinstance(topic, str):
            return False, "All topics must be strings"
        if len(topic) > 100:
            return False, "Topic length cannot exceed 100 characters"
    return True, None


def validate_parser(parser: str) -> Tuple[bool, Optional[str]]:
    """Validate parser name"""
    if parser not in VALID_PARSERS:
        return False, f"Parser must be one of {VALID_PARSERS}"
    return True, None


# =============================================================================
# LLM Processing
# =============================================================================

def update_llm_status(article_id: int, status: str, error: Optional[str] = None):
    """Update LLM processing status in database"""
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            if error:
                c.execute(
                    'UPDATE articles SET llm_status = ?, llm_content = ? WHERE id = ?',
                    (status, f"Error: {error}", article_id)
                )
            else:
                c.execute(
                    'UPDATE articles SET llm_status = ? WHERE id = ?',
                    (status, article_id)
                )
    except Exception as e:
        logger.error(f"Failed to update LLM status for article {article_id}: {e}")


def store_llm_content(article_id: int, content: str):
    """Store LLM-processed content in database"""
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute(
                'UPDATE articles SET llm_content = ?, llm_status = ? WHERE id = ?',
                (content, 'completed', article_id)
            )
    except Exception as e:
        logger.error(f"Failed to store LLM content for article {article_id}: {e}")


def process_article_llm(article_id: int, html: str):
    """Background task: Process article with LLM"""
    try:
        logger.info(f"Starting LLM processing for article {article_id}")
        update_llm_status(article_id, 'processing')

        # Process with LLM
        result = llm_parser.process_article(html)

        if result['success']:
            logger.info(f"LLM processing complete for article {article_id}: "
                       f"method={result['method']}, "
                       f"input_size={result.get('input_size', 'N/A')}, "
                       f"output_size={result.get('output_size', 'N/A')}")
            store_llm_content(article_id, result['content'])
        elif result.get('rate_limited'):
            logger.warning(f"Rate limited for article {article_id}, will retry later")
            update_llm_status(article_id, 'pending')
        else:
            error = result.get('error', 'Unknown error')
            logger.error(f"LLM processing failed for article {article_id}: {error}")
            update_llm_status(article_id, 'failed', error)

    except Exception as e:
        logger.exception(f"Exception in LLM processor for article {article_id}")
        update_llm_status(article_id, 'failed', str(e))


# =============================================================================
# Parser Processing
# =============================================================================

def process_article_parsers(article_id: int, html: str, html_file: str):
    """Background task: Run all parsers and update database"""
    try:
        logger.info(f"Starting parser processing for article {article_id}")

        # Run all parsers via parser-manager
        result = subprocess.run(
            ['node', str(BACKEND_DIR / 'parser-manager.js')],
            input=html,
            capture_output=True,
            text=True,
            timeout=PARSER_TIMEOUT,
            cwd=str(BASE_DIR)
        )

        if result.returncode != 0:
            error_msg = result.stderr or "Extraction failed"
            logger.error(f"Parser manager failed for article {article_id}: {error_msg}")
            return

        all_results = json.loads(result.stdout)
        extracted = all_results['results'].get(all_results['default'], {})
        parser_results = all_results['results']

        if not extracted.get('success'):
            logger.error(f"Parser extraction failed for article {article_id}")
            return

        # Extract parsed data
        title = extracted.get('title', 'Untitled')[:MAX_TITLE_LENGTH]
        author = extracted.get('byline')
        if author:
            author = author[:200]  # Limit author length
        text_content = extracted.get('textContent', '')
        html_content = extracted.get('htmlContent', '')
        text_length = extracted.get('fullTextLength', 0)
        read_time = calculate_read_time(text_length)
        snippet = generate_snippet(text_content)

        # Update database with parsed results
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('''
                UPDATE articles
                SET title = ?, author = ?, text_content = ?, html_content = ?,
                    snippet = ?, text_length = ?, read_time = ?, parser_results = ?
                WHERE id = ?
            ''', (title, author, text_content, html_content, snippet,
                  text_length, read_time, json.dumps(parser_results), article_id))

        logger.info(f"Parser processing complete for article {article_id}: "
                   f"title={title}, author={author or 'N/A'}, "
                   f"text_length={text_length}, read_time={read_time}")

    except subprocess.TimeoutExpired:
        logger.error(f"Parser processing timed out for article {article_id}")
    except Exception as e:
        logger.exception(f"Exception in parser processor for article {article_id}")


def process_single_parser(article_id: int, html: str, parser_name: str):
    """Background task: Run a single parser and update specific result in database"""
    try:
        logger.info(f"Re-driving parser '{parser_name}' for article {article_id}")

        # Run specific parser
        result = subprocess.run(
            ['node', str(BACKEND_DIR / 'parser-manager.js'), parser_name],
            input=html,
            capture_output=True,
            text=True,
            timeout=SINGLE_PARSER_TIMEOUT,
            cwd=str(BASE_DIR)
        )

        new_result = None

        if result.returncode != 0:
            error_msg = result.stderr or "Extraction failed"
            logger.error(f"Parser '{parser_name}' failed for article {article_id}: {error_msg}")
            new_result = {"success": False, "error": error_msg}
        else:
            try:
                output = json.loads(result.stdout)
                new_result = output['results'].get(parser_name)
            except Exception as e:
                logger.error(f"Failed to parse JSON output: {e}")
                new_result = {"success": False, "error": f"JSON parse error: {e}"}

        if not new_result:
            logger.error(f"No result returned for parser '{parser_name}'")
            new_result = {"success": False, "error": "No result returned"}

        # Update database
        with get_db_connection() as conn:
            c = conn.cursor()

            # Get existing results
            c.execute('SELECT parser_results FROM articles WHERE id = ?', (article_id,))
            row = c.fetchone()
            existing_json = row[0] if row and row[0] else '{}'

            try:
                results = json.loads(existing_json)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in parser_results for article {article_id}, resetting")
                results = {}

            # Update specific parser result
            results[parser_name] = new_result

            c.execute('UPDATE articles SET parser_results = ? WHERE id = ?',
                      (json.dumps(results), article_id))

        logger.info(f"Re-drive of '{parser_name}' complete for article {article_id}")

    except subprocess.TimeoutExpired:
        logger.error(f"Single parser '{parser_name}' timed out for article {article_id}")
    except Exception as e:
        logger.exception(f"Exception in single parser processor for article {article_id}")


def read_article_html(article_id: int) -> Tuple[Optional[str], Optional[str]]:
    """Read HTML file for an article, returns (html, error_message)"""
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('SELECT html_file FROM articles WHERE id = ?', (article_id,))
            row = c.fetchone()

        if not row:
            return None, "Article not found"

        filepath = HTML_DIR / row[0]
        if not filepath.exists():
            return None, "HTML file not found"

        with open(filepath, 'r', encoding='utf-8') as f:
            html = f.read()

        return html, None

    except Exception as e:
        logger.exception(f"Failed to read HTML for article {article_id}")
        return None, str(e)


# =============================================================================
# Routes - Static Files
# =============================================================================

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


# =============================================================================
# Routes - API Endpoints
# =============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('SELECT COUNT(*) FROM articles')
            count = c.fetchone()[0]

        return jsonify({
            "status": "ok",
            "message": "Article Saver running",
            "articles_count": count
        })
    except Exception as e:
        logger.exception("Health check failed")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/save', methods=['POST'])
def save_article():
    """
    Main endpoint: receives HTML, saves immediately, processes parsers async
    """
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400

        html = data.get('html', '')
        archive_url = data.get('archive_url', '')
        original_url = data.get('original_url', archive_url)
        force_overwrite = data.get('force_overwrite', False)
        topics = data.get('topics', [])

        # Validate inputs
        valid, error = validate_html_input(html)
        if not valid:
            return jsonify({"success": False, "error": error}), 400

        valid, error = validate_url_input(archive_url)
        if not valid:
            return jsonify({"success": False, "error": error}), 400

        if original_url:
            valid, error = validate_url_input(original_url)
            if not valid:
                return jsonify({"success": False, "error": error}), 400

        valid, error = validate_topics(topics)
        if not valid:
            return jsonify({"success": False, "error": error}), 400

        topics_json = json.dumps(topics)

        logger.info(f"Saving article from: {archive_url}, size: {len(html)}, overwrite: {force_overwrite}")

        # Check for duplicate URL
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('SELECT id, title FROM articles WHERE archive_url = ?', (archive_url,))
            existing = c.fetchone()

        if existing and not force_overwrite:
            logger.warning(f"Duplicate URL found! Existing article ID: {existing['id']}")
            return jsonify({
                "success": False,
                "duplicate": True,
                "existing_id": existing['id'],
                "existing_title": existing['title']
            }), 409

        # Quick extraction of title from HTML for immediate response
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        title_tag = soup.find('title')
        title = title_tag.get_text().strip() if title_tag else 'Untitled'
        title = title[:MAX_TITLE_LENGTH]

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

        # Save to database
        with get_db_connection() as conn:
            c = conn.cursor()

            if existing and force_overwrite:
                # Update existing article
                article_id = existing['id']
                logger.info(f"Overwriting existing article ID: {article_id}")

                # Delete old HTML file
                c.execute('SELECT html_file FROM articles WHERE id = ?', (article_id,))
                old_row = c.fetchone()
                if old_row and old_row[0]:
                    old_filepath = HTML_DIR / old_row[0]
                    if old_filepath.exists():
                        old_filepath.unlink()

                c.execute('''
                    UPDATE articles
                    SET title = ?, source_domain = ?, original_url = ?, html_file = ?,
                        captured_at = CURRENT_TIMESTAMP, topics = ?
                    WHERE id = ?
                ''', (title, source_domain, original_url, filename, topics_json, article_id))
            else:
                # Insert new article
                c.execute('''
                    INSERT INTO articles
                    (title, source_domain, archive_url, original_url, html_file, topics)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (title, source_domain, archive_url, original_url, filename, topics_json))
                article_id = c.lastrowid

        logger.info(f"Article saved: id={article_id}, title={title}, domain={source_domain}")

        # Queue async processing
        threading.Thread(
            target=process_article_parsers,
            args=(article_id, html, filename),
            daemon=True,
            name=f"Parser-{article_id}"
        ).start()

        threading.Thread(
            target=process_article_llm,
            args=(article_id, html),
            daemon=True,
            name=f"LLM-{article_id}"
        ).start()

        logger.info(f"Queued background processing for article {article_id}")

        return jsonify({
            "success": True,
            "id": article_id,
            "title": title,
            "message": "Article saved! Processing in background..."
        })

    except Exception as e:
        logger.exception("Error saving article")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/articles', methods=['GET'])
def list_articles():
    """List all saved articles with metadata for library view"""
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('''
                SELECT id, title, author, source_domain, archive_url, original_url,
                       snippet, text_length, read_time, captured_at, status, topics
                FROM articles
                ORDER BY captured_at DESC
            ''')
            rows = c.fetchall()

        articles = []
        for row in rows:
            try:
                topics = json.loads(row['topics']) if row['topics'] else []
            except json.JSONDecodeError:
                logger.warning(f"Invalid topics JSON for article {row['id']}")
                topics = []

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
                'captured_at': row['captured_at'],
                'status': row['status'] or 'unread',
                'topics': topics
            })

        return jsonify(articles)

    except Exception as e:
        logger.exception("Error listing articles")
        return jsonify({"error": "Failed to list articles"}), 500


@app.route('/articles/<int:article_id>', methods=['GET'])
def get_article(article_id):
    """Get full article content for reader view"""
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('SELECT * FROM articles WHERE id = ?', (article_id,))
            row = c.fetchone()

        if not row:
            return jsonify({"error": "Article not found"}), 404

        # Parse parser_results from JSON if available
        parser_results = None
        if row['parser_results']:
            try:
                parser_results = json.loads(row['parser_results'])
            except json.JSONDecodeError:
                logger.warning(f"Invalid parser_results JSON for article {article_id}")
                parser_results = None

        # Parse topics
        try:
            topics = json.loads(row['topics']) if row['topics'] else []
        except json.JSONDecodeError:
            logger.warning(f"Invalid topics JSON for article {article_id}")
            topics = []

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
            'llm_status': row['llm_status'],
            'status': row['status'] or 'unread',
            'topics': topics
        })

    except Exception as e:
        logger.exception(f"Error getting article {article_id}")
        return jsonify({"error": "Failed to get article"}), 500


@app.route('/articles/<int:article_id>', methods=['PATCH'])
def update_article(article_id):
    """Update article status or topics"""
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400

        updates = []
        values = []

        if 'status' in data:
            valid, error = validate_status(data['status'])
            if not valid:
                return jsonify({"success": False, "error": error}), 400
            updates.append("status = ?")
            values.append(data['status'])

        if 'topics' in data:
            valid, error = validate_topics(data['topics'])
            if not valid:
                return jsonify({"success": False, "error": error}), 400
            updates.append("topics = ?")
            values.append(json.dumps(data['topics']))

        if not updates:
            return jsonify({"success": False, "error": "No fields to update"}), 400

        values.append(article_id)

        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute(f'UPDATE articles SET {", ".join(updates)} WHERE id = ?', tuple(values))
            if c.rowcount == 0:
                return jsonify({"success": False, "error": "Article not found"}), 404

        logger.info(f"Updated article {article_id}: {updates}")
        return jsonify({"success": True})

    except Exception as e:
        logger.exception(f"Error updating article {article_id}")
        return jsonify({"success": False, "error": "Failed to update article"}), 500


@app.route('/articles/<int:article_id>/html', methods=['GET'])
def get_article_html(article_id):
    """Serve original HTML file"""
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('SELECT html_file FROM articles WHERE id = ?', (article_id,))
            row = c.fetchone()

        if not row:
            return jsonify({"error": "Article not found"}), 404

        filepath = HTML_DIR / row[0]
        if not filepath.exists():
            return jsonify({"error": "HTML file not found"}), 404

        return send_file(filepath, mimetype='text/html')

    except Exception as e:
        logger.exception(f"Error serving HTML for article {article_id}")
        return jsonify({"error": "Failed to serve HTML"}), 500


@app.route('/articles/<int:article_id>', methods=['DELETE'])
def delete_article(article_id):
    """Delete article and its HTML file"""
    try:
        with get_db_connection() as conn:
            c = conn.cursor()

            # Get HTML filename
            c.execute('SELECT html_file FROM articles WHERE id = ?', (article_id,))
            row = c.fetchone()

            if not row:
                return jsonify({"success": False, "error": "Article not found"}), 404

            # Delete HTML file
            if row[0]:
                filepath = HTML_DIR / row[0]
                if filepath.exists():
                    filepath.unlink()

            # Delete database entry
            c.execute('DELETE FROM articles WHERE id = ?', (article_id,))

        logger.info(f"Deleted article {article_id}")
        return jsonify({"success": True})

    except Exception as e:
        logger.exception(f"Error deleting article {article_id}")
        return jsonify({"success": False, "error": "Failed to delete article"}), 500


@app.route('/articles/<int:article_id>/reprocess', methods=['POST'])
def reprocess_article(article_id):
    """Trigger re-processing for a specific parser"""
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400

        parser = data.get('parser')
        if not parser:
            return jsonify({"success": False, "error": "Parser name required"}), 400

        valid, error = validate_parser(parser)
        if not valid:
            return jsonify({"success": False, "error": error}), 400

        # Read article HTML
        html, error = read_article_html(article_id)
        if error:
            return jsonify({"success": False, "error": error}), 404

        # Queue reprocessing
        if parser == 'llm':
            threading.Thread(
                target=process_article_llm,
                args=(article_id, html),
                daemon=True,
                name=f"LLM-Reprocess-{article_id}"
            ).start()
        else:
            threading.Thread(
                target=process_single_parser,
                args=(article_id, html, parser),
                daemon=True,
                name=f"Parser-{parser}-{article_id}"
            ).start()

        logger.info(f"Queued re-processing of '{parser}' for article {article_id}")
        return jsonify({"success": True, "message": f"Queued re-processing for {parser}"})

    except Exception as e:
        logger.exception(f"Error reprocessing article {article_id}")
        return jsonify({"success": False, "error": "Failed to queue reprocessing"}), 500


# Deprecated endpoint for backward compatibility
@app.route('/articles/<int:article_id>/reprocess-llm', methods=['POST'])
def reprocess_llm(article_id):
    """Legacy endpoint: Use /articles/<id>/reprocess with parser='llm' instead"""
    return reprocess_article(article_id)


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 Article Saver - Backend Server")
    print("="*80)
    print("Server running at: http://localhost:3000")
    print("Frontend: http://localhost:3000/")
    print("Health check: http://localhost:3000/health")
    print("\nAPI Endpoints:")
    print("  POST   /save                      - Save article from bookmarklet")
    print("  GET    /articles                  - List all articles")
    print("  GET    /articles/:id              - Get article content")
    print("  PATCH  /articles/:id              - Update article status/topics")
    print("  GET    /articles/:id/html         - View original HTML")
    print("  DELETE /articles/:id              - Delete article")
    print("  POST   /articles/:id/reprocess    - Reprocess article with parser")
    print("\nDatabase:", DB_PATH)
    print("HTML files:", HTML_DIR)
    print("="*80 + "\n")

    # Initialize database and run migrations
    init_db()
    run_migrations()

    app.run(host='0.0.0.0', port=3000, debug=True)
