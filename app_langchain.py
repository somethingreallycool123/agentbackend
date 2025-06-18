import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatAnthropic
from langchain.chat_models import ChatGoogleGenerativeAI
from langchain.agents import Tool, AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.utilities import SQLDatabase
from langchain.chains import SQLDatabaseChain

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


class ChatRequest(BaseModel):
    provider: str
    prompt: str
    history: Optional[List[str]] = None


app = FastAPI()

# Global memory and vector stores
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

TEXT_EMBED_MODEL = os.getenv("TEXT_EMBED_MODEL", "text-embedding-3-small")
CODE_EMBED_MODEL = os.getenv("CODE_EMBED_MODEL", "text-embedding-ada-002")

text_embeddings = OpenAIEmbeddings(model=TEXT_EMBED_MODEL)
code_embeddings = OpenAIEmbeddings(model=CODE_EMBED_MODEL)

text_vector_db = Chroma(persist_directory="./chroma_text_db", embedding_function=text_embeddings)
code_vector_db = Chroma(persist_directory="./chroma_code_db", embedding_function=code_embeddings)

# Configure SQL database
SQL_URI = os.getenv("SQL_URI", "sqlite:///data.db")
sql_database = SQLDatabase.from_uri(SQL_URI)


def get_llm(provider: str):
    if provider == "openai":
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")
        return ChatOpenAI(model_name="gpt-4o", temperature=0.6)
    if provider == "claude":
        if not ANTHROPIC_API_KEY:
            raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY is not set")
        return ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.6)
    if provider == "gemini":
        if not GOOGLE_API_KEY:
            raise HTTPException(status_code=500, detail="GOOGLE_API_KEY is not set")
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")


def run_terminal(cmd: str) -> str:
    import subprocess

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            return result.stdout
        return f"Error (code {result.returncode}): {result.stderr}"
    except Exception as e:
        return f"Exception: {str(e)}"


def read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file {path}: {str(e)}"


def write_file(spec: str) -> str:
    try:
        path, content = spec.split("|", 1)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return "ok"
    except Exception as e:
        return f"Error writing file: {str(e)}"


def web_get(url: str) -> str:
    import requests

    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.text[:5000]
    except Exception as e:
        return f"Error: {str(e)}"


def vector_search_text(query: str) -> str:
    docs = text_vector_db.similarity_search(query, k=4)
    return "\n".join(d.page_content for d in docs)


def vector_search_code(query: str) -> str:
    docs = code_vector_db.similarity_search(query, k=4)
    return "\n".join(d.page_content for d in docs)


def vector_add_text(content: str) -> str:
    text_vector_db.add_texts([content])
    return "ok"


def vector_add_code(content: str) -> str:
    code_vector_db.add_texts([content])
    return "ok"


def list_dir(path: str) -> str:
    try:
        files = os.listdir(path)
        return "\n".join(files)
    except Exception as e:
        return f"Error listing directory {path}: {str(e)}"


def sql_query(q: str) -> str:
    llm = ChatOpenAI(temperature=0)
    chain = SQLDatabaseChain.from_llm(llm, sql_database)
    return chain.run(q)


def build_agent(provider: str):
    llm = get_llm(provider)
    tools = [
        Tool(name="terminal", func=run_terminal, description="Execute a shell command"),
        Tool(name="read_file", func=read_file, description="Read a file"),
        Tool(name="write_file", func=write_file, description="Write content to a file. Format: path|content"),
        Tool(name="list_dir", func=list_dir, description="List files in a directory"),
        Tool(name="web_get", func=web_get, description="Fetch a web page via HTTP GET"),
        Tool(name="vector_search_text", func=vector_search_text, description="Search the text vector database"),
        Tool(name="vector_search_code", func=vector_search_code, description="Search the code vector database"),
        Tool(name="vector_add_text", func=vector_add_text, description="Add a text document to the text vector DB"),
        Tool(name="vector_add_code", func=vector_add_code, description="Add a code snippet to the code vector DB"),
        Tool(name="sql_query", func=sql_query, description="Run an SQL query"),
    ]
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        memory=memory,
        verbose=False,
    )
    return agent


@app.post("/chat")
def chat(req: ChatRequest):
    agent = build_agent(req.provider)
    if req.history:
        for msg in req.history:
            memory.chat_memory.add_user_message(msg)
    result = agent.run(req.prompt)
    return {"response": result}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
