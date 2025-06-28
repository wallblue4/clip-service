import requests
import json
from pathlib import Path
import time

def test_live_service():
    """Test completo del servicio en vivo"""
    
    print("🚀 TESTING MICROSERVICIO EN VIVO")
    print("=" * 50)
    
    base_url = "http://localhost:8001"
    
    # 1. Test endpoint raíz
    print("1️⃣ Test endpoint raíz...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            data = response.json()
            print("✅ Endpoint raíz OK")
            print(f"   Service: {data.get('service', 'N/A')}")
            print(f"   Version: {data.get('version', 'N/A')}")
            print(f"   Status: {data.get('status', 'N/A')}")
        else:
            print(f"❌ Error: {response.status_code}")
    except Exception as e:
        print(f"❌ Error conectando: {e}")
        return False
    
    # 2. Test health check
    print("\n2️⃣ Test health check...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check OK")
            print(f"   Status: {data.get('status', 'N/A')}")
            print(f"   Classifier loaded: {data.get('classifier_loaded', False)}")
            print(f"   Device: {data.get('device', 'N/A')}")
            print(f"   Total products: {data.get('total_products', 0)}")
        else:
            print(f"❌ Error: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # 3. Test estadísticas
    print("\n3️⃣ Test estadísticas...")
    try:
        response = requests.get(f"{base_url}/stats")
        if response.status_code == 200:
            data = response.json()
            print("✅ Estadísticas OK")
            print(f"   Total imágenes: {data.get('total_images', 0)}")
            print(f"   Modelos únicos: {data.get('total_models', 0)}")
            print(f"   Marcas únicas: {data.get('total_brands', 0)}")
        else:
            print(f"❌ Error: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # 4. Test clasificación con imagen
    print("\n4️⃣ Test clasificación con imagen...")
    
    # Buscar una imagen de prueba
    test_images = list(Path("./data/test_dataset").glob("*/*.jpg"))
    
    if test_images:
        test_image = test_images[0]
        print(f"🖼️ Usando imagen: {test_image}")
        
        try:
            start_time = time.time()
            
            with open(test_image, 'rb') as f:
                files = {"image": (test_image.name, f, "image/jpeg")}
                params = {"top_k": 3}
                
                response = requests.post(
                    f"{base_url}/classify",
                    files=files,
                    params=params,
                    timeout=30
                )
            
            processing_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Clasificación exitosa!")
                print(f"   Tiempo total: {processing_time:.2f}ms")
                print(f"   Tiempo CLIP: {data.get('processing_time_ms', 0):.2f}ms")
                print(f"   Resultados encontrados: {len(data.get('results', []))}")
                
                # Mostrar top 3 resultados
                results = data.get('results', [])
                print(f"\n🏆 Top 3 resultados:")
                for result in results[:3]:
                    print(f"   {result['rank']}. {result['model_name']}")
                    print(f"      Marca: {result['brand']}")
                    print(f"      Confianza: {result['confidence_percentage']:.2f}%")
                    print(f"      Nivel: {result['confidence_level']}")
                
            else:
                print(f"❌ Error clasificación: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print("⚠️ No se encontraron imágenes de prueba")
    
    # 5. Test con imagen inexistente (manejo de errores)
    print("\n5️⃣ Test manejo de errores...")
    try:
        # Crear archivo de texto como "imagen"
        fake_image = Path("fake_image.txt")
        fake_image.write_text("Esta no es una imagen")
        
        with open(fake_image, 'rb') as f:
            files = {"image": ("fake.jpg", f, "image/jpeg")}
            response = requests.post(f"{base_url}/classify", files=files)
        
        if response.status_code >= 400:
            print("✅ Manejo de errores OK")
            print(f"   Status: {response.status_code}")
        else:
            print("⚠️ Debería haber fallado con imagen inválida")
        
        # Limpiar
        fake_image.unlink()
        
    except Exception as e:
        print(f"❌ Error testing errores: {e}")
    
    print(f"\n🎉 ¡TESTING COMPLETADO!")
    return True

def test_performance():
    """Test de performance básico"""
    
    print(f"\n⚡ TEST DE PERFORMANCE")
    print("=" * 30)
    
    base_url = "http://localhost:8001"
    test_images = list(Path("./data/test_dataset").glob("*/*.jpg"))
    
    if not test_images:
        print("⚠️ No hay imágenes para test de performance")
        return
    
    # Test con 5 imágenes
    times = []
    
    for i, test_image in enumerate(test_images[:5]):
        print(f"🔄 Test {i+1}/5: {test_image.name}")
        
        try:
            start_time = time.time()
            
            with open(test_image, 'rb') as f:
                files = {"image": (test_image.name, f, "image/jpeg")}
                response = requests.post(f"{base_url}/classify", files=files)
            
            total_time = (time.time() - start_time) * 1000
            times.append(total_time)
            
            if response.status_code == 200:
                data = response.json()
                clip_time = data.get('processing_time_ms', 0)
                print(f"   ✅ Total: {total_time:.2f}ms, CLIP: {clip_time:.2f}ms")
            else:
                print(f"   ❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\n📊 Estadísticas de performance:")
        print(f"   ⚡ Promedio: {avg_time:.2f}ms")
        print(f"   🚀 Mínimo: {min_time:.2f}ms")
        print(f"   🐌 Máximo: {max_time:.2f}ms")

if __name__ == "__main__":
    # Verificar que el servicio esté corriendo
    try:
        response = requests.get("http://localhost:8001/", timeout=5)
        if response.status_code == 200:
            print("✅ Servicio detectado en http://localhost:8001")
            
            # Ejecutar tests
            test_live_service()
            test_performance()
            
        else:
            print("❌ Servicio no responde correctamente")
    except:
        print("❌ Servicio no está corriendo")
        print("💡 Ejecuta primero: python -m app.main")