# download_sample_images.py
import requests
import os
from pathlib import Path

def download_image(url, filepath):
    """Descargar imagen desde URL"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✅ Descargado: {filepath}")
        return True
    except Exception as e:
        print(f"❌ Error descargando {url}: {e}")
        return False

# URLs de ejemplo (puedes reemplazar con tus propias imágenes)
sample_images = {
    "data/dataset/nike_air_max_90/sample1.jpg": "https://via.placeholder.com/400x300/FF0000/FFFFFF?text=Nike+Air+Max+90",
    "data/dataset/adidas_ultraboost/sample1.jpg": "https://via.placeholder.com/400x300/0000FF/FFFFFF?text=Adidas+Ultraboost",
    "data/dataset/jordan_1_retro/sample1.jpg": "https://via.placeholder.com/400x300/00FF00/FFFFFF?text=Jordan+1+Retro"
}

def download_samples():
    """Descargar imágenes de ejemplo"""
    print("📥 Descargando imágenes de ejemplo...")
    
    for filepath, url in sample_images.items():
        # Crear directorio si no existe
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        download_image(url, filepath)
    
    print("✅ Imágenes de ejemplo listas")

if __name__ == "__main__":
    download_samples()