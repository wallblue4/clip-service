# test_classification.py
import os
import sys
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# Agregar el directorio de la app al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.models.classification import SneakerClassificationSystem

def test_basic_initialization():
    """Test básico de inicialización"""
    print("🧪 Test 1: Inicialización básica")
    
    try:
        classifier = SneakerClassificationSystem(
            database_path="./data/test_database"
        )
        print("✅ Inicialización exitosa")
        print(f"   - Dispositivo: {classifier.device}")
        print(f"   - Ruta DB: {classifier.database_path}")
        return True
    except Exception as e:
        print(f"❌ Error en inicialización: {e}")
        return False

def test_clip_model_loading():
    """Test de carga del modelo CLIP"""
    print("\n🧪 Test 2: Carga del modelo CLIP")
    
    try:
        classifier = SneakerClassificationSystem(
            database_path="./data/test_database"
        )
        classifier.load_clip_model()
        print("✅ Modelo CLIP cargado exitosamente")
        print(f"   - Modelo: {classifier.model is not None}")
        print(f"   - Preprocessor: {classifier.preprocess is not None}")
        return True
    except Exception as e:
        print(f"❌ Error cargando modelo CLIP: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Ejecutando tests del sistema de clasificación")
    print("=" * 50)
    
    # Ejecutar tests
    test1 = test_basic_initialization()
    test2 = test_clip_model_loading()
    
    print("\n" + "=" * 50)
    print("📊 RESUMEN DE TESTS:")
    print(f"✅ Tests exitosos: {sum([test1, test2])}/2")
    
    if test1 and test2:
        print("🎉 Todos los tests básicos pasaron!")
        print("\n🚀 El sistema está listo para procesar imágenes")
    else:
        print("⚠️ Algunos tests fallaron. Revisa los errores arriba.")