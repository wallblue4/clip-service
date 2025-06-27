import faiss
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import clip
import os
import numpy as np
import pickle
from typing import List, Union, Tuple, Dict
from PIL import Image
import sqlite3
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import logging
import gc

logger = logging.getLogger(__name__)

class SneakerClassificationSystem:
    def __init__(self, database_path: str = "./data/sneaker_database", device: str = None):
        """
        Sistema de clasificación de tenis usando embeddings de CLIP
        
        Args:
            database_path: Ruta donde se almacena la base de datos
            device: 'cuda', 'cpu' o None (auto-detectar)
        """
        # Auto-detectar dispositivo
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"🚀 Inicializando SneakerClassificationSystem en {self.device}")

        self.database_path = database_path
        self.model = None
        self.preprocess = None
        self.index = None
        self.sneaker_metadata = []

        # Crear directorios necesarios
        os.makedirs(database_path, exist_ok=True)

        # Rutas de archivos
        self.embeddings_path = os.path.join(database_path, "embeddings.npy")
        self.index_path = os.path.join(database_path, "faiss_index.idx")
        self.metadata_path = os.path.join(database_path, "metadata.json")
        self.db_path = os.path.join(database_path, "sneakers.db")

        # Inicializar base de datos SQLite
        self.init_database()

    def init_database(self):
        """Inicializar base de datos SQLite para metadata"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

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

            # Crear índices para mejorar performance
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_model_name ON sneakers(model_name)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_brand ON sneakers(brand)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_embedding_index ON sneakers(embedding_index)
            ''')

            conn.commit()
            conn.close()
            logger.info("✅ Base de datos SQLite inicializada")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando base de datos: {e}")
            raise

    def load_clip_model(self):
        """Cargar modelo CLIP"""
        if self.model is not None:
            logger.info("📋 Modelo CLIP ya está cargado")
            return

        try:
            logger.info("📥 Cargando modelo CLIP...")
            self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)
            logger.info(f"✅ Modelo CLIP ViT-L/14 cargado en {self.device}")

            # Mostrar información del dispositivo
            if self.device == "cuda":
                try:
                    gpu_name = torch.cuda.get_device_name()
                    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                    logger.info(f"🔥 GPU: {gpu_name}")
                    logger.info(f"💾 Memoria GPU total: {total_memory:.1f} GB")
                except Exception as e:
                    logger.warning(f"No se pudo obtener info GPU: {e}")
            else:
                logger.info("💻 Usando CPU para inferencia")

        except Exception as e:
            logger.error(f"❌ Error cargando modelo CLIP: {e}")
            raise

    def get_image_paths_by_model(self, base_directory: str) -> Dict[str, List[str]]:
        """Obtener rutas de imágenes organizadas por modelo"""
        model_images = {}
        base_path = Path(base_directory)

        if not base_path.exists():
            raise FileNotFoundError(f"Directorio no encontrado: {base_directory}")

        logger.info(f"🔍 Escaneando directorio: {base_directory}")

        # Buscar directorios de modelos
        model_dirs = [d for d in base_path.iterdir() if d.is_dir()]
        
        if not model_dirs:
            logger.warning(f"⚠️ No se encontraron subdirectorios en {base_directory}")
            return model_images

        for model_dir in tqdm(model_dirs, desc="📁 Escaneando modelos"):
            model_name = model_dir.name
            image_paths = []

            # Extensiones de imagen soportadas
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff']
            
            for ext in extensions:
                # Buscar archivos con cada extensión (case insensitive)
                pattern = f"*{ext}"
                image_paths.extend([str(p) for p in model_dir.glob(pattern)])
                pattern = f"*{ext.upper()}"
                image_paths.extend([str(p) for p in model_dir.glob(pattern)])

            if image_paths:
                model_images[model_name] = list(set(image_paths))  # Eliminar duplicados
                logger.debug(f"📷 {model_name}: {len(image_paths)} imágenes")

        logger.info(f"✅ Encontrados {len(model_images)} modelos con imágenes")
        total_images = sum(len(paths) for paths in model_images.values())
        logger.info(f"📊 Total de imágenes: {total_images}")
        
        return model_images

    def get_image_embedding(self, image_path: str) -> np.ndarray:
        """Generar embedding para una imagen"""
        try:
            # Cargar y procesar imagen
            image = Image.open(image_path).convert("RGB")
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(image_input).float()

            return image_features.cpu().numpy()[0]
            
        except Exception as e:
            logger.error(f"❌ Error procesando imagen {image_path}: {e}")
            return None

    def get_batch_embeddings(self, image_paths: List[str], batch_size: int = 32) -> Tuple[np.ndarray, List[str]]:
        """
        Generar embeddings para múltiples imágenes en lotes
        
        Returns:
            Tuple of (embeddings_array, successful_image_paths)
        """
        all_embeddings = []
        successful_paths = []

        # Ajustar batch size según dispositivo
        if self.device == "cuda":
            batch_size = min(batch_size, 64)  # Más conservador para evitar OOM
        else:
            batch_size = min(batch_size, 16)   # CPU más lento

        logger.info(f"🔄 Procesando {len(image_paths)} imágenes en lotes de {batch_size}")

        for i in tqdm(range(0, len(image_paths), batch_size), desc="🖼️ Procesando lotes"):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            batch_valid_paths = []

            # Cargar lote de imágenes
            for path in batch_paths:
                try:
                    image = Image.open(path).convert("RGB")
                    batch_images.append(self.preprocess(image))
                    batch_valid_paths.append(path)
                except Exception as e:
                    logger.warning(f"⚠️ Error cargando imagen {path}: {e}")
                    continue

            # Procesar lote si hay imágenes válidas
            if batch_images:
                try:
                    batch_tensor = torch.stack(batch_images).to(self.device)

                    with torch.no_grad():
                        batch_features = self.model.encode_image(batch_tensor).float()

                    # Agregar a resultados
                    embeddings_np = batch_features.cpu().numpy()
                    all_embeddings.extend(embeddings_np)
                    successful_paths.extend(batch_valid_paths)

                    # Limpiar memoria GPU si es necesario
                    if self.device == "cuda":
                        del batch_tensor, batch_features
                        torch.cuda.empty_cache()

                except Exception as e:
                    logger.error(f"❌ Error procesando lote {i//batch_size + 1}: {e}")
                    continue

            # Limpiar memoria cada ciertos lotes
            if i % (batch_size * 10) == 0:
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()

        if not all_embeddings:
            raise ValueError("❌ No se pudieron generar embeddings para ninguna imagen")

        embeddings_array = np.array(all_embeddings)
        logger.info(f"✅ Embeddings generados: {len(embeddings_array)} de {len(image_paths)} imágenes")
        
        return embeddings_array, successful_paths

    def build_database(self, images_directory: str, sneaker_info_file: str = None):
        """Construir base de datos de embeddings"""
        if not self.model:
            self.load_clip_model()

        logger.info("🏗️ Iniciando construcción de base de datos...")

        # Obtener rutas de imágenes
        model_images = self.get_image_paths_by_model(images_directory)
        
        if not model_images:
            raise ValueError(f"No se encontraron imágenes en {images_directory}")

        # Cargar información adicional si existe
        sneaker_info = {}
        if sneaker_info_file and os.path.exists(sneaker_info_file):
            try:
                with open(sneaker_info_file, 'r', encoding='utf-8') as f:
                    sneaker_info = json.load(f)
                logger.info(f"📋 Cargada información de {len(sneaker_info)} modelos")
            except Exception as e:
                logger.warning(f"⚠️ Error cargando archivo de info: {e}")

        # Preparar listas para procesamiento
        all_image_paths = []
        all_metadata = []

        for model_name, image_paths in model_images.items():
            for img_path in image_paths:
                all_image_paths.append(img_path)

                # Crear metadata
                model_info = sneaker_info.get(model_name, {})
                metadata = {
                    'model_name': model_name,
                    'image_path': img_path,
                    'brand': model_info.get('brand', 'Unknown'),
                    'color': model_info.get('color', 'Unknown'),
                    'size': model_info.get('size', 'Unknown'),
                    'price': float(model_info.get('price', 0.0)),
                    'description': model_info.get('description', '')
                }
                all_metadata.append(metadata)

        logger.info(f"🖼️ Procesando {len(all_image_paths)} imágenes...")

        # Generar embeddings
        embeddings, successful_paths = self.get_batch_embeddings(all_image_paths)
        
        # Filtrar metadata para imágenes exitosas
        path_to_metadata = {meta['image_path']: meta for meta in all_metadata}
        filtered_metadata = [path_to_metadata[path] for path in successful_paths if path in path_to_metadata]

        logger.info(f"✅ Embeddings exitosos: {len(embeddings)}")
        logger.info(f"📊 Metadata filtrada: {len(filtered_metadata)}")

        # Crear índice FAISS
        logger.info("🔗 Creando índice FAISS...")
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product para similitud coseno

        # Normalizar embeddings para similitud coseno
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))

        # Guardar en base de datos SQLite
        logger.info("💾 Guardando metadata en base de datos...")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Limpiar tabla existente
        cursor.execute('DELETE FROM sneakers')

        # Insertar nuevos datos
        for i, metadata in enumerate(tqdm(filtered_metadata, desc="💾 Guardando metadata")):
            cursor.execute('''
                INSERT INTO sneakers (model_name, brand, color, size, price, description, image_path, embedding_index)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metadata['model_name'],
                metadata['brand'],
                metadata['color'],
                metadata['size'],
                metadata['price'],
                metadata['description'],
                metadata['image_path'],
                i
            ))

        conn.commit()
        conn.close()

        # Guardar archivos
        logger.info("💾 Guardando archivos de embeddings...")
        np.save(self.embeddings_path, embeddings)
        faiss.write_index(self.index, self.index_path)

        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_metadata, f, ensure_ascii=False, indent=2)

        # Estadísticas finales
        unique_models = len(set(meta['model_name'] for meta in filtered_metadata))
        unique_brands = len(set(meta['brand'] for meta in filtered_metadata))

        logger.info("🎉 Base de datos creada exitosamente!")
        logger.info(f"📊 Estadísticas finales:")
        logger.info(f"   - Imágenes procesadas: {len(embeddings)}")
        logger.info(f"   - Modelos únicos: {unique_models}")
        logger.info(f"   - Marcas únicas: {unique_brands}")
        logger.info(f"   - Dimensión embeddings: {dimension}")

        # Mostrar uso de memoria
        if self.device == "cuda":
            try:
                memory_used = torch.cuda.memory_allocated() / 1e9
                logger.info(f"   - Memoria GPU usada: {memory_used:.2f} GB")
            except:
                pass

        # Limpiar memoria
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def load_database(self):
        """Cargar base de datos existente"""
        if not self.model:
            self.load_clip_model()

        # Verificar archivos necesarios
        required_files = [self.index_path, self.metadata_path, self.db_path]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            raise FileNotFoundError(f"❌ Archivos de base de datos faltantes: {missing_files}")

        logger.info("📥 Cargando base de datos...")
        
        try:
            # Cargar índice FAISS
            self.index = faiss.read_index(self.index_path)
            
            # Cargar metadata
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.sneaker_metadata = json.load(f)

            logger.info(f"✅ Base de datos cargada: {len(self.sneaker_metadata)} imágenes indexadas")
            
            # Verificar consistencia
            if self.index.ntotal != len(self.sneaker_metadata):
                logger.warning(f"⚠️ Inconsistencia: FAISS({self.index.ntotal}) vs Metadata({len(self.sneaker_metadata)})")

        except Exception as e:
            logger.error(f"❌ Error cargando base de datos: {e}")
            raise

    def classify_sneaker(self, image_path: str, top_k: int = 5) -> List[Dict]:
        """Clasificar un tenis dado una imagen"""
        if not self.index:
            raise RuntimeError("Base de datos no cargada. Ejecuta load_database() primero.")

        logger.info(f"🔍 Clasificando imagen: {os.path.basename(image_path)}")

        # Generar embedding de la imagen query
        query_embedding = self.get_image_embedding(image_path)
        if query_embedding is None:
            return []

        # Normalizar para similitud coseno
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)

        # Buscar similares
        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))

        # Preparar resultados
        results = []
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            if idx == -1:  # FAISS retorna -1 si no hay suficientes resultados
                break
                
            cursor.execute('''
                SELECT * FROM sneakers WHERE embedding_index = ?
            ''', (int(idx),))

            row = cursor.fetchone()
            if row:
                # Convertir similitud coseno a porcentaje
                confidence = max(0.0, min(100.0, float(distance) * 100))
                
                result = {
                    'rank': i + 1,
                    'similarity_score': float(distance),
                    'confidence_percentage': confidence,
                    'model_name': row[1],
                    'brand': row[2],
                    'color': row[3],
                    'size': row[4],
                    'price': row[5],
                    'description': row[6],
                    'image_path': row[7]
                }
                results.append(result)

        conn.close()
        
        logger.info(f"✅ Clasificación completada: {len(results)} resultados")
        return results

    def get_model_statistics(self) -> Dict:
        """Obtener estadísticas de la base de datos"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Estadísticas básicas
            cursor.execute('SELECT COUNT(*) FROM sneakers')
            total_images = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(DISTINCT model_name) FROM sneakers')
            total_models = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(DISTINCT brand) FROM sneakers')
            total_brands = cursor.fetchone()[0]

            # Top modelos
            cursor.execute('''
                SELECT model_name, COUNT(*) as count
                FROM sneakers
                GROUP BY model_name
                ORDER BY count DESC
                LIMIT 10
            ''')
            top_models = cursor.fetchall()

            # Estadísticas por marca
            cursor.execute('''
                SELECT brand, COUNT(*) as count
                FROM sneakers
                GROUP BY brand
                ORDER BY count DESC
            ''')
            brands_stats = cursor.fetchall()

            conn.close()

            return {
                'total_images': total_images,
                'total_models': total_models,
                'total_brands': total_brands,
                'top_models': top_models,
                'brands_stats': brands_stats
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            return {
                'total_images': 0,
                'total_models': 0,
                'total_brands': 0,
                'top_models': [],
                'brands_stats': []
            }