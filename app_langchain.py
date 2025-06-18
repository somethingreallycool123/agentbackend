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


class ChatRequest(BaseModel):
    provider: str
    prompt: str
    history: Optional[List[str]] = None


app = FastAPI()

# Global memory and vector store
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
embeddings = OpenAIEmbeddings()
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Configure SQL database
SQL_URI = os.getenv("SQL_URI", "sqlite:///data.db")
sql_database = SQLDatabase.from_uri(SQL_URI)


def get_llm(provider: str):
    if provider == "openai":
        return ChatOpenAI(model_name="gpt-4o", temperature=0.6)
    if provider == "claude":
        return ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.6)
    if provider == "gemini":
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


def vector_search(query: str) -> str:
    docs = vector_db.similarity_search(query, k=4)
    return "\n".join(d.page_content for d in docs)


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
        Tool(name="web_get", func=web_get, description="Fetch a web page via HTTP GET"),
        Tool(name="vector_search", func=vector_search, description="Search the vector database"),
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
