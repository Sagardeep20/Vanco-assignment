from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "AI Digital Asset Manager"
    database_url: str = (
        "postgresql+psycopg2://postgres:postgres@localhost:5432/assets"
    )
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    frontend_url: str = "http://localhost:5173"
    dataset_path: str = "./data"
    # Local-only AI by default. Set AI_PROVIDER=ollama + OLLAMA_MODEL to use
    # a model already present in a local Ollama instance. The app NEVER
    # pulls/downloads models; missing/unreachable Ollama falls back to local.
    ai_provider: str = "local"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = ""
    # Embedding provider: auto | local | deterministic (default auto).
    # auto: local 512-d CLIP model when already present, else fallback.
    # local: local model or a clear error. deterministic: hash fallback.
    # The app NEVER downloads models; local loading is offline-only.
    embedding_provider: str = "auto"
    local_embedding_model: str = "clip-ViT-B-32"
    local_embedding_model_path: str = ""

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
