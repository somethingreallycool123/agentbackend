# agentbackend

This repository contains a prototype LangChain agent and a Tauri-based UI.

## Backend

`app_langchain.py` exposes a FastAPI service. The agent uses two separate vector databases:

- `./chroma_text_db` for natural language documents
- `./chroma_code_db` for code snippets

Embeddings are created with OpenAI models that can be configured via environment variables:

```
TEXT_EMBED_MODEL  # default: text-embedding-3-small
CODE_EMBED_MODEL  # default: text-embedding-ada-002
```

Supported LLM providers are OpenAI, Anthropic and Gemini. The following API keys must be available as environment variables when starting the server:

```
OPENAI_API_KEY
ANTHROPIC_API_KEY
GOOGLE_API_KEY
```

The optional `SQL_URI` variable points to the database used by the SQL tool (defaults to `sqlite:///data.db`).

Start the backend with:

```bash
python app_langchain.py
```

### Tools

The agent exposes several tools:

- `terminal` – execute a shell command
- `read_file` – read a file from disk
- `write_file` – write a file (`path|content` format)
- `list_dir` – list directory contents
- `web_get` – fetch a web page
- `vector_search_text` / `vector_search_code` – search the text or code vector DB
- `vector_add_text` / `vector_add_code` – add documents to the vector DBs
- `sql_query` – execute SQL against `SQL_URI`

## Frontend

The Tauri frontend (`index.html`, `main.js`, `main.rs`) provides a floating chat UI but does not currently call the backend.
