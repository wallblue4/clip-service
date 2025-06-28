import os
import sys
from pathlib import Path

# Agregar app al path
sys.path.append('./app')

def test_complete_system():
    """Test completo del sistema con datos sintéticos"""
    
    print("🚀 TEST COMPLETO DEL SISTEMA")
    print("=" * 60)
    
    # 1. Crear BD sintética si no existe
    database_path = Path("./data/sneaker_database")
    if not database_path.exists() or not (database_path / "embeddings.npy").exists():
        print("📦 Creando base de datos sintética...")
        from create_test_database import create_synthetic_database
        create_synthetic_database()
    else:
        print("✅ Base de datos ya existe")
    
    # 2. Test del sistema de clasificación
    print(f"\n🧪 Testing sistema de clasificación...")
    from app.models.classification import SneakerClassificationSystem
    
    try:
        # Inicializar
        classifier = SneakerClassificationSystem(
            database_path="./data/sneaker_database"
        )
        
        # Cargar BD
        classifier.load_database()
        
        # Estadísticas
        stats = classifier.get_model_statistics()
        print(f"✅ Sistema cargado:")
        print(f"   📊 Imágenes: {stats['total_images']}")
        print(f"   👟 Modelos: {stats['total_models']}")
        print(f"   🏷️ Marcas: {stats['total_brands']}")
        
    except Exception as e:
        print(f"❌ Error en clasificación: {e}")
        return False
    
    # 3. Test de clasificación real
    print(f"\n🎯 Testing clasificación...")
    try:
        # Usar una de las imágenes sintéticas existentes
        test_images = list(Path("./data/test_dataset").glob("*/*.jpg"))
        if test_images:
            test_image = str(test_images[0])
            print(f"🖼️ Usando imagen: {Path(test_image).name}")
            
            results = classifier.classify_sneaker(test_image, top_k=3)
            
            if results:
                print(f"✅ Clasificación exitosa:")
                for result in results[:3]:
                    print(f"   {result['rank']}. {result['model_name']} ({result['confidence_percentage']:.1f}%)")
            else:
                print("⚠️ Sin resultados")
        else:
            print("⚠️ No se encontraron imágenes de prueba")
            
    except Exception as e:
        print(f"❌ Error clasificando: {e}")
        return False
    
    # 4. Test de API endpoints (simular)
    print(f"\n🌐 Testing configuración API...")
    try:
        from app.core.config import settings
        print(f"✅ Configuración cargada:")
        print(f"   🔧 Device: {settings.device}")
        print(f"   📁 DB Path: {settings.DATABASE_PATH}")
        print(f"   🚪 Port: {settings.PORT}")
        
    except Exception as e:
        print(f"❌ Error en configuración: {e}")
        return False
    
    print(f"\n🎉 ¡SISTEMA COMPLETO FUNCIONANDO!")
    print(f"🚀 Listo para:")
    print(f"   - Recibir archivos reales de Colab")
    print(f"   - Servir requests de clasificación") 
    print(f"   - Integrarse con tu app principal")
    
    return True

if __name__ == "__main__":
    success = test_complete_system()
    
    if success:
        print(f"\n💡 PRÓXIMOS PASOS:")
        print(f"   1. python -m app.main  # Iniciar microservicio")
        print(f"   2. curl http://localhost:8001/health  # Verificar")
        print(f"   3. Reemplazar con BD real de Colab cuando esté lista")