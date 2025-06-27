from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    # Servidor
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    DEBUG: bool = False
    
    # CLIP Configuration
    USE_GPU: bool = True
    MODEL_NAME: str = "ViT-L/14"
    
    # Paths
    DATABASE_PATH: str = "./data/sneaker_database"
    DATASET_PATH: str = "./data/dataset"
    
    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000"
    ]
    
    # Límites
    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024  # 10MB
    DEFAULT_TOP_K: int = 5
    
    # Auto-detectar device
    @property
    def device(self) -> str:
        if self.USE_GPU:
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return "cpu"
    
    class Config:
        env_file = ".env"

settings = Settings()