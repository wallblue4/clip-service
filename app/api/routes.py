from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
import tempfile
import time
import os
from datetime import datetime
from typing import Optional
import logging

from app.core.config import settings
from app.schemas.responses import ClassificationResponse, StatsResponse, ErrorResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# Esta variable se actualizará desde main.py
classifier = None

def get_classifier():
    """Obtener instancia del clasificador"""
    from app.main import classifier as global_classifier
    return global_classifier

@router.post("/classify", response_model=ClassificationResponse)
async def classify_sneaker(
    image: UploadFile = File(...),
    top_k: int = Query(default=settings.DEFAULT_TOP_K, ge=1, le=20)
):
    """
    Clasificar imagen de tenis usando CLIP
    
    - **image**: Imagen del tenis a clasificar (JPG, PNG, etc.)
    - **top_k**: Número de resultados a retornar (1-20)
    """
    
    classifier = get_classifier()
    if not classifier:
        raise HTTPException(
            status_code=503, 
            detail="Classifier not available. Service may be starting up."
        )
    
    # Validar imagen
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="El archivo debe ser una imagen (JPG, PNG, etc.)"
        )
    
    try:
        # Leer imagen
        content = await image.read()
        if len(content) > settings.MAX_IMAGE_SIZE:
            raise HTTPException(
                status_code=413, 
                detail=f"Imagen muy grande. Máximo {settings.MAX_IMAGE_SIZE // (1024*1024)}MB"
            )
        
        # Procesar imagen
        start_time = time.time()
        
        # Guardar imagen temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Clasificar
            results = classifier.classify_sneaker(tmp_path, top_k=top_k)
            processing_time = (time.time() - start_time) * 1000
            
            # Obtener dimensiones de imagen
            try:
                from PIL import Image as PILImage
                with PILImage.open(tmp_path) as pil_img:
                    width, height = pil_img.size
            except:
                width = height = None
            
            # Formatear resultados
            formatted_results = []
            for result in results:
                # Determinar nivel de confianza
                confidence_level = "muy_alta" if result['confidence_percentage'] >= 90 else \
                                 "alta" if result['confidence_percentage'] >= 75 else \
                                 "media" if result['confidence_percentage'] >= 60 else "baja"
                
                formatted_result = {
                    "rank": result['rank'],
                    "similarity_score": result['similarity_score'],
                    "confidence_percentage": result['confidence_percentage'],
                    "confidence_level": confidence_level,
                    "model_name": result['model_name'],
                    "brand": result['brand'],
                    "color": result['color'],
                    "size": result.get('size'),
                    "price": result.get('price'),
                    "description": result.get('description'),
                    "image_path": result.get('image_path')
                }
                formatted_results.append(formatted_result)
            
            # Información del modelo
            model_info = {
                "model_name": settings.MODEL_NAME,
                "device": classifier.device if classifier else "unknown",
                "embedding_dimension": 768 if settings.MODEL_NAME == "ViT-L/14" else 512,
                "total_products": len(classifier.sneaker_metadata) if hasattr(classifier, 'sneaker_metadata') and classifier.sneaker_metadata else 0
            }
            
            # Información de la imagen
            image_info = {
                "filename": image.filename or "unknown",
                "size_bytes": len(content),
                "content_type": image.content_type,
                "width": width,
                "height": height
            }
            
            return ClassificationResponse(
                success=True,
                processing_time_ms=round(processing_time, 2),
                timestamp=datetime.now(),
                results=formatted_results,
                query_image_info=image_info,
                model_info=model_info,
                total_matches_found=len(formatted_results)
            )
            
        finally:
            # Limpiar archivo temporal
            try:
                os.unlink(tmp_path)
            except:
                pass
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en clasificación: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Error procesando imagen: {str(e)}"
        )

@router.get("/stats", response_model=StatsResponse)
async def get_statistics():
    """Obtener estadísticas de la base de datos"""
    
    classifier = get_classifier()
    if not classifier:
        raise HTTPException(status_code=503, detail="Classifier not available")
    
    try:
        stats = classifier.get_model_statistics()
        return StatsResponse(**stats)
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas: {e}")
        raise HTTPException(status_code=500, detail="Error obteniendo estadísticas")

@router.post("/rebuild-database")
async def rebuild_database():
    """
    Reconstruir base de datos de embeddings
    ⚠️ NOTA: Esto requiere que el dataset esté disponible
    """
    
    classifier = get_classifier()
    if not classifier:
        raise HTTPException(status_code=503, detail="Classifier not available")
    
    try:
        # Verificar que existe el dataset
        if not os.path.exists(settings.DATASET_PATH):
            raise HTTPException(
                status_code=400, 
                detail=f"Dataset no encontrado en {settings.DATASET_PATH}"
            )
        
        # Contar imágenes disponibles
        import glob
        image_count = len(glob.glob(os.path.join(settings.DATASET_PATH, "*", "*.jpg")))
        image_count += len(glob.glob(os.path.join(settings.DATASET_PATH, "*", "*.png")))
        
        if image_count == 0:
            raise HTTPException(
                status_code=400,
                detail="No se encontraron imágenes en el dataset"
            )
        
        logger.info(f"Iniciando rebuild de base de datos con {image_count} imágenes...")
        
        # Rebuild (esto puede tomar tiempo)
        classifier.build_database(settings.DATASET_PATH)
        
        # Obtener nuevas estadísticas
        stats = classifier.get_model_statistics()
        
        return {
            "success": True,
            "message": "Base de datos reconstruida exitosamente",
            "images_processed": stats.get("total_images", 0),
            "models_found": stats.get("total_models", 0),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en rebuild: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error rebuilding database: {str(e)}")