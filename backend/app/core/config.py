from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str
    API_VERSION: str

    HOST: str
    PORT: int

    DEBUG: bool

    DATABASE_URL: str

    SECRET_KEY: str
    ALGORITHM: str

    ACCESS_TOKEN_EXPIRE_MINUTES: int
    REFRESH_TOKEN_EXPIRE_DAYS: int

    UPLOAD_DIR: str
    CHROMA_DB_DIR: str

    OLLAMA_BASE_URL: str
    OLLAMA_MODEL: str

    EMBEDDING_MODEL: str

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
    )


settings = Settings()