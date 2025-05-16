import json
import logging
from typing import List, Dict, Any
import time

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
# Continue using the deprecated version as requested
from langchain_postgres import PGVector
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_file="config.json"):
    """Charge la configuration depuis un fichier JSON."""
    try:
        with open(config_file, "r") as file:
            config = json.load(file)
            return config
    except FileNotFoundError:
        logger.error(f"Le fichier {config_file} n'a pas été trouvé.")
        return {}
    except json.JSONDecodeError:
        logger.error(f"Erreur lors de la lecture du fichier {config_file}.")
        return {}

class VectorDatabaseSetup:
    """Classe pour configurer la base de données et l'extension pgvector."""
    
    def __init__(self, config_file="config.json"):
        self.config = load_config(config_file)
        self.db_config = self.config.get("postgres", {})
        
    def get_connection_params(self):
        """Récupère les paramètres de connexion à la base de données."""
        return {
            "host": self.db_config.get("host", "localhost"),
            "port": self.db_config.get("port", 5432),
            "user": self.db_config.get("user", "postgres"),
            "password": self.db_config.get("password", "")
        }
    
    def get_connection_string(self, include_dbname=True):
        """Génère la chaîne de connexion PostgreSQL."""
        params = self.get_connection_params()
        dbname = self.db_config.get("database", "vectordb")
        auth = f"{params['user']}"
        if params['password']:  # Only include password if it exists
            auth = f"{params['user']}:{params['password']}"
        
        if include_dbname:
            return f"postgresql://{auth}@{params['host']}:{params['port']}/{dbname}"
        else:
            return f"postgresql://{auth}@{params['host']}:{params['port']}"
    
    def setup_database(self):
        """Vérifie la connectivité à la base de données et prépare le schéma nécessaire pour PGVector."""
        try:
            # Se connecter à la base de données existante
            conn = psycopg2.connect(self.get_connection_string())
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Vérifier que la connexion fonctionne
            cursor.execute("SELECT 1")
            
            # Vérifier que l'extension pgvector est installée
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Créer une table de collections avec une colonne uuid explicite
            # C'est la clé de notre correction - s'assurer que la table a bien une colonne uuid
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS langchain_pg_collection (
                uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                collection_name TEXT UNIQUE,
                cmetadata JSONB
            )   
            """)
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info("Connexion à la base de données réussie et schéma préparé.")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la configuration de la base de données: {e}")
            return False

class EmbeddingManager:
    """Classe pour gérer les embeddings et leur stockage."""
    
    def __init__(self, config_file="config.json"):
        self.config = load_config(config_file)
        self.model_name = self.config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.db_setup = VectorDatabaseSetup(config_file)
        self.connection_string = self.db_setup.get_connection_string()
        self._embedding_model = None
    
    @property
    def embedding_model(self):
        """Charge le modèle d'embedding de manière paresseuse."""
        if self._embedding_model is None:
            try:
                self._embedding_model = HuggingFaceEmbeddings(model_name=self.model_name)
                logger.info(f"Modèle d'embedding {self.model_name} chargé avec succès.")
            except Exception as e:
                logger.error(f"Erreur lors du chargement du modèle d'embedding: {e}")
                raise
        return self._embedding_model
    
    def get_collection_name(self, ingredient):
        """Génère un nom de collection standardisé pour un ingrédient."""
        # Nettoyer et normaliser le nom pour éviter des problèmes de compatibilité
        collection_name = ingredient.lower().replace(" ", "_").replace("-", "_")
        return f"collection_{collection_name}"
    
    def store_chunks(self, chunks: List[Document], ingredient: str, overwrite: bool = False) -> Dict[str, Any]:
        """
        Transforme les chunks en embeddings et les stocke dans la base de données vectorielle.
        
        Args:
            chunks: Liste des chunks à stocker
            ingredient: Nom de l'ingrédient pour nommer la collection
            overwrite: Si True, supprime la collection existante avant d'ajouter les nouveaux documents
            
        Returns:
            Dictionnaire contenant les résultats de l'opération
        """
        if not chunks:
            return {
                "status": "error",
                "message": "Aucun chunk à stocker.",
                "ingredient": ingredient,
                "stored_chunks": 0
            }
        
        # Vérifier que la base de données est correctement configurée
        if not self.db_setup.setup_database():
            return {
                "status": "error",
                "message": "Échec de la configuration de la base de données.",
                "ingredient": ingredient,
                "stored_chunks": 0
            }
        
        collection_name = self.get_collection_name(ingredient)
        start_time = time.time()
        
        try:
            conn_string = self.db_setup.get_connection_string()
            
            # Initialiser le vectorstore avec la nouvelle implémentation
            vectorstore = PGVector.from_documents(
                documents=chunks,  
                embedding=self.embedding_model,
                collection_name=collection_name,
                connection=conn_string,
                use_jsonb=True,
                pre_delete_collection=overwrite
            )
            
            # Ajouter les documents à la base vectorielle
            vectorstore.add_documents(chunks)
            
            end_time = time.time()
            processing_time = round(end_time - start_time, 2)
            
            logger.info(f"{len(chunks)} chunks stockés dans la collection {collection_name} en {processing_time} secondes.")
            
            return {
                "status": "success",
                "message": f"{len(chunks)} chunks transformés en embeddings et stockés avec succès.",
                "ingredient": ingredient,
                "collection_name": collection_name,
                "stored_chunks": len(chunks),
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Erreur lors du stockage des embeddings: {e}")
            return {
                "status": "error",
                "message": f"Erreur lors du stockage des embeddings: {str(e)}",
                "ingredient": ingredient,
                "stored_chunks": 0
            }
    
    def list_collections(self) -> List[str]:
        """Liste toutes les collections disponibles dans la base de données."""
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()
            
            # Vérifier d'abord si la table existe
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'langchain_pg_collection'
                )
            """)
            
            table_exists = cursor.fetchone()[0]
            
            if not table_exists:
                logger.info("La table langchain_pg_collection n'existe pas encore.")
                return []
            
            cursor.execute("SELECT collection_name FROM langchain_pg_collection")
            collections = [row[0] for row in cursor.fetchall()]
            cursor.close()
            conn.close()
            return collections
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des collections: {e}")
            return []
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Récupère des informations sur une collection spécifique."""
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()
            
            # Vérifier si la table collection existe
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'langchain_pg_collection'
                )
            """)
            
            table_exists = cursor.fetchone()[0]
            
            if not table_exists:
                return {
                    "status": "error",
                    "message": "La table des collections n'existe pas encore.",
                    "collection_name": collection_name
                }
            
            # Vérifier si la collection existe
            cursor.execute("SELECT 1 FROM langchain_pg_collection WHERE collection_name = %s", (collection_name,))
            exists = cursor.fetchone()
            
            if not exists:
                return {
                    "status": "error",
                    "message": f"La collection {collection_name} n'existe pas.",
                    "collection_name": collection_name
                }
            
            # Récupérer les métadonnées de la collection
            cursor.execute("SELECT cmetadata FROM langchain_pg_collection WHERE collection_name = %s", (collection_name,))
            metadata_row = cursor.fetchone()
            metadata = metadata_row[0] if metadata_row else {}
            
            # Le nom de la table d'embedding peut être différent selon la version de langchain_postgres
            # Essayer de déterminer dynamiquement le nom de la table d'embedding
            embedding_table_name = f"langchain_pg_embedding_{collection_name}"
            alt_embedding_table_name = f"langchain_embedding_{collection_name}"
            
            # Vérifier quelle table existe
            cursor.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_name IN (%s, %s)
            """, (embedding_table_name, alt_embedding_table_name))
            
            table_result = cursor.fetchone()
            
            if not table_result:
                return {
                    "status": "success",
                    "collection_name": collection_name,
                    "document_count": 0,
                    "metadata": metadata,
                    "note": "La collection existe mais la table d'embedding n'a pas été trouvée."
                }
            
            actual_table_name = table_result[0]
            
            # Récupérer le nombre de documents dans la collection
            cursor.execute(f"SELECT COUNT(*) FROM {actual_table_name}")
            count = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            return {
                "status": "success",
                "collection_name": collection_name,
                "document_count": count,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des informations sur la collection {collection_name}: {e}")
            return {
                "status": "error",
                "message": f"Erreur: {str(e)}",
                "collection_name": collection_name
            }


# Fonction d'utilité pour stocker directement les chunks d'un ingrédient
def store_article_chunks(chunks: List[Document], ingredient: str, overwrite: bool = False, config_file: str = "config.json") -> Dict[str, Any]:
    """
    Fonction utilitaire pour stocker directement les chunks d'articles dans la base vectorielle.
    
    Args:
        chunks: Liste des chunks à stocker
        ingredient: Nom de l'ingrédient
        overwrite: Si True, écrase la collection existante
        config_file: Chemin vers le fichier de configuration
        
    Returns:
        Résultats de l'opération de stockage
    """
    manager = EmbeddingManager(config_file)
    return manager.store_chunks(chunks, ingredient, overwrite)


# Exemple d'intégration avec article_chunker.py
def process_and_store_ingredient(ingredient: str, overwrite: bool = False) -> Dict[str, Any]:
    """
    Traite les articles d'un ingrédient et stocke les chunks résultants.
    
    Args:
        ingredient: Nom de l'ingrédient
        overwrite: Si True, écrase la collection existante
        
    Returns:
        Résultats combinés des opérations de traitement et de stockage
    """
    # Importer la fonction de traitement des articles
    from article_chunker import process_downloaded_articles
    
    # Traiter les articles
    processing_results = process_downloaded_articles(ingredient)
    
    # Vérifier si le traitement a réussi
    if processing_results.get("status") != "success":
        return {
            "status": "error",
            "message": f"Erreur lors du traitement des articles: {processing_results.get('message', 'Erreur inconnue')}",
            "ingredient": ingredient,
            "processing_results": processing_results,
            "embedding_results": None
        }
    
    # Récupérer les chunks
    chunks = processing_results.get("chunks", [])
    
    # Stocker les chunks
    embedding_results = store_article_chunks(chunks, ingredient, overwrite)
    
    # Combiner les résultats
    return {
        "status": embedding_results.get("status"),
        "message": f"Traitement et stockage terminés. {embedding_results.get('stored_chunks')} chunks stockés.",
        "ingredient": ingredient,
        "processing_results": processing_results,
        "embedding_results": embedding_results
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Obtenir l'ingrédient depuis les arguments de ligne de commande
        ingredient = sys.argv[1]
        
        # Vérifier s'il faut écraser la collection existante
        overwrite = "--overwrite" in sys.argv
        
        print(f"Traitement et stockage des articles pour l'ingrédient: {ingredient}")
        print(f"Mode écrasement: {'activé' if overwrite else 'désactivé'}")
        
        results = process_and_store_ingredient(ingredient, overwrite)
        
        # Afficher les résultats
        print(f"\nStatut: {results['status']}")
        print(f"Message: {results['message']}")
        
        if results.get("processing_results"):
            print("\nInformations de traitement:")
            print(f"- Fichiers traités: {results['processing_results'].get('processed_files', 0)}")
            print(f"- Total des chunks: {results['processing_results'].get('total_chunks', 0)}")
        
        if results.get("embedding_results") and results["embedding_results"].get("status") == "success":
            print("\nInformations de stockage:")
            print(f"- Collection: {results['embedding_results'].get('collection_name', '')}")
            print(f"- Chunks stockés: {results['embedding_results'].get('stored_chunks', 0)}")
            print(f"- Temps de traitement: {results['embedding_results'].get('processing_time', 0)} secondes")
    else:
        print("Usage: python embedding_store.py <ingredient> [--overwrite]")