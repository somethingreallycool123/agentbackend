# Agent Backend

This repo contains a FastAPI server built with LangChain and a simple Tauri front-end.

## Setup

1. Copy `.env.example` to `.env` and fill in the required API keys.
2. Install Python dependencies (e.g. `pip install fastapi langchain[all] python-dotenv uvicorn`).
3. Run the server:

```bash
python app_langchain.py
```

Endpoints require an API key supplied via the `X-API-Key` header. The key is
configured via the `API_KEY` variable in the environment.

## Front-end

The Tauri front-end sources are located in `index.html`, `main.js`, `lib.rs`,
`main.rs` and `tauri.conf.json`. It provides a basic chat interface that sends
messages to the `/chat` endpoint.

Build the front-end with the standard Tauri toolchain.
