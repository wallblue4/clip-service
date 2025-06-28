# app/core/config.py - Optimizada para Render
from pydantic_settings import BaseSettings
from typing import List
import os
from pathlib import Path

class Settings(BaseSettings):
    # Servidor - Render específico
    HOST: str = "0.0.0.0"
    PORT: int = int(os.getenv("PORT", "10000"))  # Render asigna puerto
    DEBUG: bool = False
    
    # CLIP Configuration - Optimizado para free tier
    USE_GPU: bool = False  # Render gratuita no tiene GPU
    MODEL_NAME: str = "ViT-B/32"  # Modelo más pequeño para memoria limitada
    
    # Paths - Render compatible
    DATABASE_PATH: str = "./data/sneaker_database"
    DATASET_PATH: str = "./data/dataset"
    
    # CORS - Permitir tu dominio de TuStockYa
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "https://tu-app-principal.onrender.com",  # Tu app principal
        "*"  # Para desarrollo
    ]
    
    # Límites - Ajustados para Render
    MAX_IMAGE_SIZE: int = 5 * 1024 * 1024  # 5MB (reducido)
    DEFAULT_TOP_K: int = 3  # Reducido para menos procesamiento
    
    # Render optimizations
    BATCH_SIZE: int = 1  # Procesar una imagen a la vez
    MAX_WORKERS: int = 1  # Un solo worker
    
    @property
    def device(self) -> str:
        return "cpu"  # Forzar CPU en Render
    
    @property
    def database_path_absolute(self) -> str:
        return str(Path(self.DATABASE_PATH).resolve())
    
    @property
    def database_exists(self) -> bool:
        required_files = ["embeddings.npy", "faiss_index.idx", "sneakers.db"]
        db_path = Path(self.DATABASE_PATH)
        
        for file in required_files:
            if not (db_path / file).exists():
                return False
        return True
    
    class Config:
        env_file = ".env"

settings = Settings()