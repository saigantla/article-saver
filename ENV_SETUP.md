# Environment Setup

## API Keys Configuration

This project uses environment variables to store sensitive API keys.

### Setup Steps

1. **Copy the example file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` and add your API key:**
   ```bash
   CHUTES_API_KEY=your_actual_api_key_here
   ```

3. **Get your API key from:** https://chutes.ai

### Running the Server

The server needs the environment variable loaded. Use one of these methods:

**Method 1: Load env vars inline (Mac/Linux)**
```bash
export $(cat .env | xargs) && python3 backend/server.py
```

**Method 2: Source before running**
```bash
source .env
python3 backend/server.py
```

**Method 3: Create a start script**
```bash
#!/bin/bash
export $(cat .env | xargs)
python3 backend/server.py
```

### For Remote Server Deployment (systemd)

If you're using systemd to manage the service:

1. **SSH into your server and navigate to the project directory:**
   ```bash
   cd /home/gantl/article-saver
   ```

2. **Create `.env` file with your API key:**
   ```bash
   nano .env
   ```
   Add this line:
   ```
   CHUTES_API_KEY=your_actual_api_key_here
   ```

3. **The systemd service file is already configured** to load `.env`:
   ```ini
   EnvironmentFile=/home/gantl/article-saver/.env
   ```

4. **Copy the updated service file and reload systemd:**
   ```bash
   sudo cp article-saver.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl restart article-saver
   ```

5. **Verify it's working:**
   ```bash
   sudo systemctl status article-saver
   ```

**IMPORTANT:**
- Never commit the `.env` file to git. It's already in `.gitignore`.
- Make sure the `.env` file has proper permissions (readable by the service user).
