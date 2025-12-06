# Article Saver

Lightweight self-hosted article archiving system with a cyber-minimalist interface.

## Features

- Save archived articles from archive.is via bookmarklet
- Clean reader view with progress tracking
- URL-based deduplication with overwrite confirmation
- SQLite database for metadata
- Full-text search support
- Readability.js for clean content extraction

## Architecture

- **Backend**: Flask (Python) + Node.js for Readability extraction
- **Frontend**: Single-page app with Tailwind CSS
- **Database**: SQLite
- **Storage**: HTML files on disk

## Local Development

```bash
# Install Python dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt

# Install Node dependencies
npm install

# Start server
python3 backend/server.py
```

Server runs at http://localhost:3000

## Helios Deployment

### Initial Setup

```bash
# On Helios
cd ~
git clone <your-github-repo-url> article-saver
cd article-saver

# Install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
npm install

# Create data directory
mkdir -p data/html

# Create systemd service
sudo cp article-saver.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable article-saver
sudo systemctl start article-saver

# Configure Tailscale Serve for HTTPS
tailscale serve https / http://localhost:3000
```

### Updating

```bash
# On Helios
cd ~/article-saver
git pull
sudo systemctl restart article-saver
```

## Usage

1. Visit your server (local: http://localhost:3000, Helios: https://helios.ts.net)
2. Click "SAVE_URL" and drag the bookmarklet to your bookmarks bar
3. Visit any article on archive.is
4. Click the bookmarklet to save it
5. View your library and read saved articles

## API Endpoints

- `POST /save` - Save article from bookmarklet
- `GET /articles` - List all articles
- `GET /articles/:id` - Get article content
- `GET /articles/:id/html` - View original HTML
- `DELETE /articles/:id` - Delete article
- `GET /health` - Health check
