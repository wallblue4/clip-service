from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
import logging
from datetime import datetime

from app.core.config import settings
from app.schemas.responses import HealthResponse, ErrorResponse

# Configurar logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Crear aplicación FastAPI
app = FastAPI(
    title="CLIP Sneaker Classification Service",
    description="Microservicio especializado en clasificación de tenis usando CLIP",
    version="1.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variable global para el clasificador
classifier = None

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Manejador global de excepciones"""
    logger.error(f"Error no controlado: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if settings.DEBUG else "An unexpected error occurred",
            timestamp=datetime.now()
        ).dict()
    )

@app.on_event("startup")
async def startup_event():
    """Inicializar servicio al arrancar"""
    global classifier
    
    try:
        logger.info("🚀 Iniciando CLIP Classification Service...")
        logger.info(f"📍 Dispositivo configurado: {settings.device}")
        logger.info(f"🎯 Modelo CLIP: {settings.MODEL_NAME}")
        
        # Importar aquí para evitar errores en startup si falta alguna dependencia
        from app.models.classification import SneakerClassificationSystem
        
        classifier = SneakerClassificationSystem(
            database_path=settings.DATABASE_PATH,
            device=settings.device
        )
        
        # Intentar cargar base de datos existente
        try:
            classifier.load_database()
            logger.info("✅ Base de datos CLIP cargada exitosamente")
        except FileNotFoundError:
            logger.warning("⚠️ Base de datos no encontrada. Usar /rebuild-database para crear.")
        except Exception as e:
            logger.error(f"❌ Error cargando base de datos: {e}")
            classifier = None
        
        logger.info("✅ CLIP Classification Service iniciado correctamente")
        
    except Exception as e:
        logger.error(f"❌ Error fatal en startup: {e}", exc_info=True)
        classifier = None

@app.on_event("shutdown")
async def shutdown_event():
    """Limpiar recursos al cerrar"""
    global classifier
    logger.info("🔄 Cerrando CLIP Classification Service...")
    
    if classifier and hasattr(classifier, 'device') and classifier.device == "cuda":
        try:
            import torch
            torch.cuda.empty_cache()
            logger.info("🧹 Cache GPU limpiado")
        except:
            pass
    
    logger.info("✅ Servicio cerrado correctamente")

@app.get("/")
async def root():
    """Endpoint raíz con información del servicio"""
    return {
        "service": "CLIP Sneaker Classification Microservice",
        "version": "1.0.0",
        "status": "running" if classifier else "initializing",
        "model": settings.MODEL_NAME,
        "device": settings.device,
        "gpu_available": settings.device == "cuda",
        "docs_url": "/docs" if settings.DEBUG else "disabled",
        "endpoints": {
            "classify": "POST /classify",
            "health": "GET /health", 
            "stats": "GET /stats",
            "rebuild": "POST /rebuild-database"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check detallado del servicio"""
    
    try:
        # Información básica
        health_data = {
            "status": "healthy" if classifier else "unhealthy",
            "classifier_loaded": classifier is not None,
            "device": settings.device,
            "model_name": settings.MODEL_NAME,
            "gpu_available": settings.device == "cuda",
            "total_products": 0
        }
        
        # Si el clasificador está cargado, obtener más info
        if classifier:
            try:
                stats = classifier.get_model_statistics()
                health_data["total_products"] = stats.get("total_images", 0)
                
                # Información de memoria GPU si está disponible
                if settings.device == "cuda":
                    try:
                        import torch
                        health_data["memory_usage"] = {
                            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
                            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
                            "total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9
                        }
                    except:
                        pass
                        
            except Exception as e:
                logger.warning(f"Error obteniendo estadísticas: {e}")
        
        return HealthResponse(**health_data)
        
    except Exception as e:
        logger.error(f"Error en health check: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

# Importar rutas de la API
from app.api.routes import router
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )