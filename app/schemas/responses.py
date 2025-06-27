from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class ClassificationResult(BaseModel):
    """Resultado individual de clasificación"""
    rank: int
    similarity_score: float
    confidence_percentage: float
    confidence_level: str
    model_name: str
    brand: str
    color: str
    size: Optional[str] = None
    price: Optional[float] = None
    description: Optional[str] = None
    image_path: Optional[str] = None

class ImageInfo(BaseModel):
    """Información de la imagen procesada"""
    filename: str
    size_bytes: int
    content_type: str
    width: Optional[int] = None
    height: Optional[int] = None

class ModelInfo(BaseModel):
    """Información del modelo CLIP"""
    model_name: str
    device: str
    embedding_dimension: int
    total_products: int

class ClassificationResponse(BaseModel):
    """Respuesta completa de clasificación"""
    success: bool
    processing_time_ms: float
    timestamp: datetime
    results: List[ClassificationResult]
    query_image_info: ImageInfo
    model_info: ModelInfo
    total_matches_found: int

class HealthResponse(BaseModel):
    """Respuesta de health check"""
    status: str
    classifier_loaded: bool
    device: str
    model_name: str
    gpu_available: bool
    total_products: int
    memory_usage: Optional[Dict[str, float]] = None

class StatsResponse(BaseModel):
    """Estadísticas de la base de datos"""
    total_images: int
    total_models: int
    total_brands: int
    top_models: List[tuple]
    brands_stats: List[tuple]

class ErrorResponse(BaseModel):
    """Respuesta de error"""
    success: bool = False
    error: str
    detail: Optional[str] = None
    timestamp: datetime