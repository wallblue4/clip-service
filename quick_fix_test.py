# quick_fix_test.py
import sys
from pathlib import Path

def quick_test():
    """Test rápido de la configuración corregida"""
    
    print("🔧 TEST RÁPIDO DE CONFIGURACIÓN")
    print("=" * 40)
    
    try:
        # Test import
        sys.path.append('./app')
        from app.core.config import settings
        
        print("✅ Config importada correctamente")
        print(f"📁 DATABASE_PATH: {settings.DATABASE_PATH}")
        print(f"📁 database_path_absolute: {settings.database_path_absolute}")
        print(f"✅ database_exists: {settings.database_exists}")
        
        # Verificar archivos
        db_path = Path(settings.DATABASE_PATH)
        if db_path.exists():
            print(f"\n📂 Contenido de {db_path}:")
            for file in db_path.iterdir():
                print(f"   - {file.name}")
        else:
            print(f"\n❌ Directorio no existe: {db_path}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    
    if success:
        print(f"\n🚀 Configuración OK - Puedes ejecutar:")
        print(f"   python -m app.main")
    else:
        print(f"\n⚠️ Revisa los errores arriba")