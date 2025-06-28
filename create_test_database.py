import os
import sys
import numpy as np
import sqlite3
import json
import faiss
from pathlib import Path
from PIL import Image
import logging
from datetime import datetime

# Agregar app al path
sys.path.append('./app')

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_synthetic_database():
    """Crear base de datos sintética para testing"""
    
    print("🏭 CREANDO BASE DE DATOS SINTÉTICA")
    print("=" * 50)
    
    # Parámetros
    num_models = 20  # 20 modelos diferentes
    images_per_model = 10  # 10 imágenes por modelo
    total_images = num_models * images_per_model
    embedding_dim = 768  # ViT-L/14
    
    print(f"📊 Configuración:")
    print(f"   - Modelos: {num_models}")
    print(f"   - Imágenes por modelo: {images_per_model}")
    print(f"   - Total imágenes: {total_images}")
    print(f"   - Dimensiones: {embedding_dim}")
    
    # Crear directorios
    database_path = Path("./data/sneaker_database")
    dataset_path = Path("./data/test_dataset")
    
    database_path.mkdir(parents=True, exist_ok=True)
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    # 1. DEFINIR MODELOS SINTÉTICOS
    synthetic_models = create_model_definitions()
    print(f"✅ Definidos {len(synthetic_models)} modelos sintéticos")
    
    # 2. CREAR IMÁGENES SINTÉTICAS
    print(f"\n🖼️ Creando imágenes sintéticas...")
    all_image_paths, all_metadata = create_synthetic_images(
        dataset_path, synthetic_models, images_per_model
    )
    print(f"✅ Creadas {len(all_image_paths)} imágenes")
    
    # 3. GENERAR EMBEDDINGS SINTÉTICOS
    print(f"\n🔢 Generando embeddings sintéticos...")
    embeddings = create_synthetic_embeddings(all_metadata, embedding_dim)
    print(f"✅ Generados {len(embeddings)} embeddings")
    
    # 4. CREAR ÍNDICE FAISS
    print(f"\n🔍 Creando índice FAISS...")
    create_faiss_index(embeddings, database_path)
    print(f"✅ Índice FAISS creado")
    
    # 5. CREAR BASE DE DATOS SQLITE
    print(f"\n🗄️ Creando base de datos SQLite...")
    create_sqlite_database(all_metadata, database_path)
    print(f"✅ SQLite creado")
    
    # 6. GUARDAR ARCHIVOS ADICIONALES
    print(f"\n💾 Guardando archivos adicionales...")
    save_additional_files(embeddings, all_metadata, database_path)
    print(f"✅ Archivos guardados")
    
    # 7. VERIFICAR ESTRUCTURA
    print(f"\n🔍 Verificando estructura creada...")
    verify_created_structure(database_path)
    
    print(f"\n🎉 ¡BASE DE DATOS SINTÉTICA CREADA!")
    print(f"📁 Ubicación: {database_path}")
    
    return database_path

def create_model_definitions():
    """Definir modelos sintéticos realistas"""
    
    models = [
        # Nike
        {"name": "nike_air_max_90", "brand": "Nike", "color": "Blanco/Negro", "price": 120.0, "pattern": [255, 0, 0]},
        {"name": "nike_air_force_1", "brand": "Nike", "color": "Blanco", "price": 110.0, "pattern": [255, 255, 255]},
        {"name": "nike_jordan_1_retro", "brand": "Nike", "color": "Chicago", "price": 170.0, "pattern": [255, 0, 0]},
        {"name": "nike_dunk_low", "brand": "Nike", "color": "Panda", "price": 100.0, "pattern": [0, 0, 0]},
        {"name": "nike_blazer_mid", "brand": "Nike", "color": "Vintage", "price": 100.0, "pattern": [200, 150, 100]},
        
        # Adidas  
        {"name": "adidas_ultraboost_22", "brand": "Adidas", "color": "Negro", "price": 180.0, "pattern": [0, 0, 255]},
        {"name": "adidas_stan_smith", "brand": "Adidas", "color": "Blanco/Verde", "price": 80.0, "pattern": [0, 255, 0]},
        {"name": "adidas_gazelle", "brand": "Adidas", "color": "Azul", "price": 90.0, "pattern": [0, 100, 200]},
        {"name": "adidas_campus_00s", "brand": "Adidas", "color": "Gris", "price": 95.0, "pattern": [128, 128, 128]},
        {"name": "adidas_samba_og", "brand": "Adidas", "color": "Negro/Blanco", "price": 85.0, "pattern": [50, 50, 50]},
        
        # New Balance
        {"name": "new_balance_574", "brand": "New Balance", "color": "Gris", "price": 90.0, "pattern": [150, 150, 150]},
        {"name": "new_balance_990v5", "brand": "New Balance", "color": "Made in USA", "price": 185.0, "pattern": [100, 100, 150]},
        {"name": "new_balance_530", "brand": "New Balance", "color": "Blanco/Gris", "price": 85.0, "pattern": [200, 200, 200]},
        
        # Vans
        {"name": "vans_old_skool", "brand": "Vans", "color": "Negro/Blanco", "price": 65.0, "pattern": [0, 0, 0]},
        {"name": "vans_authentic", "brand": "Vans", "color": "Blanco", "price": 60.0, "pattern": [255, 255, 255]},
        
        # Converse
        {"name": "converse_chuck_taylor", "brand": "Converse", "color": "Blanco", "price": 60.0, "pattern": [255, 255, 255]},
        {"name": "converse_chuck_70", "brand": "Converse", "color": "Negro", "price": 75.0, "pattern": [0, 0, 0]},
        
        # Puma
        {"name": "puma_suede_classic", "brand": "Puma", "color": "Azul", "price": 70.0, "pattern": [0, 50, 150]},
        {"name": "puma_rs_x", "brand": "Puma", "color": "Multicolor", "price": 110.0, "pattern": [100, 200, 50]},
        
        # Reebok
        {"name": "reebok_classic_leather", "brand": "Reebok", "color": "Blanco", "price": 75.0, "pattern": [240, 240, 240]},
    ]
    
    # Agregar descripciones
    for model in models:
        model["description"] = f"Zapatilla {model['brand']} {model['name'].replace('_', ' ').title()}"
    
    return models

def create_synthetic_images(dataset_path, models, images_per_model):
    """Crear imágenes sintéticas distintivas por modelo"""
    
    all_image_paths = []
    all_metadata = []
    
    for model_idx, model in enumerate(models):
        model_name = model["name"]
        model_dir = dataset_path / model_name
        model_dir.mkdir(exist_ok=True)
        
        # Crear imágenes para este modelo
        for img_idx in range(images_per_model):
            # Crear imagen sintética con patrón distintivo
            img_array = create_distinctive_image(model, img_idx)
            
            # Guardar imagen
            img_filename = f"img_{img_idx+1:03d}.jpg"
            img_path = model_dir / img_filename
            
            img = Image.fromarray(img_array)
            img.save(img_path)
            
            # Agregar a listas
            all_image_paths.append(str(img_path))
            
            metadata = {
                "model_name": model_name,
                "image_path": str(img_path),
                "brand": model["brand"],
                "color": model["color"],
                "price": model["price"],
                "description": model["description"],
                "model_index": model_idx,
                "image_index": img_idx
            }
            all_metadata.append(metadata)
        
        print(f"   📷 {model_name}: {images_per_model} imágenes")
    
    return all_image_paths, all_metadata

def create_distinctive_image(model, img_idx):
    """Crear imagen sintética distintiva para un modelo"""
    
    # Imagen base aleatoria
    img_array = np.random.randint(50, 200, (300, 400, 3), dtype=np.uint8)
    
    # Patrón distintivo de la marca
    pattern_color = model["pattern"]
    
    # Diferentes patrones para diferentes marcas
    if model["brand"] == "Nike":
        # Swoosh simulado (franja diagonal)
        for i in range(100, 200):
            for j in range(50, min(350, 50 + i)):
                if i + j < 300:
                    img_array[i, j] = pattern_color
    
    elif model["brand"] == "Adidas":
        # Tres franjas verticales
        img_array[:, 50:80] = pattern_color
        img_array[:, 160:190] = pattern_color  
        img_array[:, 270:300] = pattern_color
    
    elif model["brand"] == "New Balance":
        # Logo "N" simulado
        img_array[100:200, 180:200] = pattern_color
        img_array[120:180, 200:220] = pattern_color
    
    elif model["brand"] == "Vans":
        # Franja lateral
        img_array[:, 350:400] = pattern_color
    
    elif model["brand"] == "Converse":
        # Círculo (logo estrella simulado)
        center_y, center_x = 150, 200
        radius = 30
        y, x = np.ogrid[:300, :400]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        img_array[mask] = pattern_color
    
    else:
        # Patrón genérico - rectángulo
        img_array[50:100, 50:350] = pattern_color
    
    # Variación por imagen individual
    noise = np.random.randint(-20, 20, img_array.shape, dtype=np.int16)
    img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Marca de imagen específica
    img_array[10:30, 10:50] = [255, 255, 255]  # Cuadro blanco
    
    return img_array

def create_synthetic_embeddings(metadata, embedding_dim):
    """Crear embeddings sintéticos que respeten similitudes por modelo"""
    
    embeddings = []
    
    # Crear embeddings base por modelo único
    unique_models = {}
    for meta in metadata:
        model_name = meta["model_name"]
        if model_name not in unique_models:
            # Embedding base para este modelo
            base_embedding = np.random.randn(embedding_dim).astype(np.float32)
            
            # Hacer que modelos de la misma marca sean más similares
            brand = meta["brand"]
            if brand == "Nike":
                base_embedding[:100] += 2.0  # Sesgo hacia dimensiones específicas
            elif brand == "Adidas":
                base_embedding[100:200] += 2.0
            elif brand == "New Balance":
                base_embedding[200:300] += 2.0
            # etc...
            
            unique_models[model_name] = base_embedding
    
    # Crear variaciones para cada imagen
    for meta in metadata:
        model_name = meta["model_name"]
        base_embedding = unique_models[model_name]
        
        # Agregar variación pequeña para esta imagen específica
        variation = np.random.randn(embedding_dim) * 0.1
        image_embedding = base_embedding + variation
        
        # Normalizar
        image_embedding = image_embedding / np.linalg.norm(image_embedding)
        
        embeddings.append(image_embedding)
    
    return np.array(embeddings, dtype=np.float32)

def create_faiss_index(embeddings, database_path):
    """Crear índice FAISS"""
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner Product para similitud coseno
    
    # Normalizar embeddings
    faiss.normalize_L2(embeddings)
    
    # Agregar al índice
    index.add(embeddings)
    
    # Guardar
    index_path = database_path / "faiss_index.idx"
    faiss.write_index(index, str(index_path))
    
    print(f"   🔍 Índice FAISS: {embeddings.shape[0]} vectores, {dimension} dims")

def create_sqlite_database(metadata, database_path):
    """Crear base de datos SQLite"""
    
    db_path = database_path / "sneakers.db"
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Crear tabla
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sneakers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            brand TEXT,
            color TEXT,
            size TEXT,
            price REAL,
            description TEXT,
            image_path TEXT NOT NULL,
            embedding_index INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Insertar datos
    for i, meta in enumerate(metadata):
        cursor.execute('''
            INSERT INTO sneakers (model_name, brand, color, size, price, description, image_path, embedding_index)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            meta['model_name'],
            meta['brand'], 
            meta['color'],
            'Universal',  # Talla genérica para testing
            meta['price'],
            meta['description'],
            meta['image_path'],
            i
        ))
    
    conn.commit()
    conn.close()
    
    print(f"   🗄️ SQLite: {len(metadata)} registros")

def save_additional_files(embeddings, metadata, database_path):
    """Guardar archivos adicionales"""
    
    # Guardar embeddings como NumPy
    embeddings_path = database_path / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    
    # Guardar metadata como JSON
    metadata_path = database_path / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    # Crear archivo de info
    info = {
        "created_at": datetime.now().isoformat(),
        "total_images": len(embeddings),
        "total_models": len(set(m['model_name'] for m in metadata)),
        "embedding_dimension": embeddings.shape[1],
        "type": "synthetic_test_data",
        "brands": list(set(m['brand'] for m in metadata))
    }
    
    info_path = database_path / "database_info.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)

def verify_created_structure(database_path):
    """Verificar estructura creada"""
    
    required_files = [
        "embeddings.npy",
        "faiss_index.idx", 
        "metadata.json",
        "sneakers.db",
        "database_info.json"
    ]
    
    print(f"📁 Archivos creados:")
    for file in required_files:
        file_path = database_path / file
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024*1024)
            print(f"   ✅ {file}: {size_mb:.2f} MB")
        else:
            print(f"   ❌ {file}: FALTANTE")

if __name__ == "__main__":
    create_synthetic_database()