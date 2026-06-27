from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    anthropic_api_key: str
    llm_model: str = "claude-sonnet-4-6"
    # Cap per research sub-question to prevent runaway tool loops.
    max_tool_turns: int = 5
    # Cap on critic → researcher loops before forcing synthesis.
    max_research_iterations: int = 2

    # Reranking
    cohere_api_key: str

    # Tracing
    langsmith_api_key: str = ""
    langsmith_project: str = "ai-research-agent"
    langchain_tracing_v2: bool = True

    # Web search
    tavily_api_key: str

    # ChromaDB
    # chroma_use_http=False → PersistentClient (in-process, local dev)
    # chroma_use_http=True  → HttpClient (connects to chromadb Docker service)
    chroma_use_http: bool = False
    chroma_host: str = "chromadb"   # service name in docker-compose
    chroma_port: int = 8000          # chromadb container's internal port
    chroma_persist_dir: str = "./data/chroma"

    # LangGraph
    langgraph_checkpoint_db: str = "./data/checkpoints.db"

    # Retrieval
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    retrieval_top_k: int = 20
    rerank_top_n: int = 5

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000


# Module-level singleton — import this everywhere instead of re-parsing .env.
settings = Settings()
