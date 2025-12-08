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

### For Remote Server Deployment

1. SSH into your server
2. Navigate to the project directory
3. Create `.env` file with your API key
4. Run using one of the methods above

**IMPORTANT:** Never commit the `.env` file to git. It's already in `.gitignore`.
