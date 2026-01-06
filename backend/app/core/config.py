"""
Application configuration settings.
"""
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import validator


class Settings(BaseSettings):
    """Application settings."""
    
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "DataPrep AI Platform"
    
    # Database Settings
    DATABASE_URL: str = "postgresql://dataprep_user:dataprep_password@localhost:5432/dataprep_db"
    
    # Redis Settings
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Celery Settings
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"
    
    # MinIO/S3 Settings
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin123"
    MINIO_BUCKET_NAME: str = "dataprep-files"
    MINIO_SECURE: bool = False
    
    # File Upload Settings
    MAX_FILE_SIZE: int = 1024 * 1024 * 1024  # 1GB in bytes
    ALLOWED_FILE_EXTENSIONS: List[str] = [".csv"]
    UPLOAD_DIR: str = "uploads"
    
    # Security Settings
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS Settings
    ALLOWED_HOSTS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    # Processing Settings
    MAX_ROWS_PREVIEW: int = 1000
    ANALYSIS_TIMEOUT: int = 300  # 5 minutes
    
    @validator("ALLOWED_HOSTS", pre=True)
    def assemble_cors_origins(cls, v):
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()