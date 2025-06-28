# app/main.py - Optimizada para Render
from contextlib import asynccontextmanager
from fastapi import FastAPI
import logging
import os

from app.core.config import settings

# Configurar logging para Render
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Variable global
classifier = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events - reemplaza deprecated on_event"""
    global classifier
    
    # Startup
    logger.info("🚀 Iniciando CLIP Service en Render...")
    
    try:
        if settings.database_exists:
            from app.models.classification import SneakerClassificationSystem
            
            classifier = SneakerClassificationSystem(
                database_path=settings.database_path_absolute,
                device="cpu"  # Forzar CPU
            )
            
            # Carga optimizada para memoria limitada
            classifier.load_database()
            
            stats = classifier.get_model_statistics()
            logger.info(f"✅ Cargado: {stats['total_images']} imágenes")
            
        else:
            logger.warning("⚠️ BD no encontrada - servicio en modo limitado")
            
    except Exception as e:
        logger.error(f"❌ Error startup: {e}")
        classifier = None
    
    yield
    
    # Shutdown
    logger.info("🔄 Cerrando servicio...")

# Crear app con lifespan
app = FastAPI(
    title="CLIP Sneaker Classification Service",
    description="Microservicio de clasificación de tenis",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "service": "CLIP Sneaker Classification",
        "version": "1.0.0",
        "status": "running" if classifier else "limited",
        "device": "cpu",
        "platform": "render",
        "model": settings.MODEL_NAME
    }

@app.get("/health")
async def health():
    """Health check para Render"""
    status = "healthy" if classifier else "limited"
    
    return {
        "status": status,
        "classifier_loaded": classifier is not None,
        "device": "cpu",
        "platform": "render",
        "memory_optimized": True
    }

# Importar rutas
from app.api.routes import router
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        log_level="info"
    )