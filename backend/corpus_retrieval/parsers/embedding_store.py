import logging
from typing import List, Dict, Any
import time

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from article_chunker import process_downloaded_articles
from utils import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorDatabaseSetup:
    """Class to configure the database and the pgvector extension."""
    
    def __init__(self, config_file="config.json"):
        self.config = load_config(config_file)
        self.db_config = self.config.get("postgres", {})
        
    def get_connection_params(self):
        """Retrieve the database connection parameters."""
        return {
            "host": self.db_config.get("host", "localhost"),
            "port": self.db_config.get("port", 5432),
            "user": self.db_config.get("user", "postgres"),
            "password": self.db_config.get("password", "")
        }
    
    def get_connection_string(self, include_dbname=True):
        """Generate the PostgreSQL connection string."""
        params = self.get_connection_params()
        dbname = self.db_config.get("database", "vectordb")
        auth = f"{params['user']}"
        if params['password']:
            auth = f"{params['user']}:{params['password']}"
        
        if include_dbname:
            return f"postgresql://{auth}@{params['host']}:{params['port']}/{dbname}"
        else:
            return f"postgresql://{auth}@{params['host']}:{params['port']}"
    
    def setup_database(self):
        """Checks database connectivity and prepares the necessary schema for PGVector."""
        try:
            # Connects to the existing PostgreSQL database
            conn = psycopg2.connect(self.get_connection_string())
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Checks that the connection is successful
            cursor.execute("SELECT 1")
            
            # Checks if the pgvector extension is installed
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Create a table for collections with an explicit uuid column
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
            
            logger.info("Database connection successful and schema prepared.")
            return True
            
        except Exception as e:
            logger.error(f"Error during database setup: {e}")
            return False

class EmbeddingManager:
    """Class to manage embeddings and their storage."""
    
    def __init__(self, config_file="config.json"):
        self.config = load_config(config_file)
        self.model_name = self.config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2") # Default model
        self.db_setup = VectorDatabaseSetup(config_file)
        self.connection_string = self.db_setup.get_connection_string()
        self._embedding_model = None
    
    @property
    def embedding_model(self):
        """Loads the embedding model."""
        if self._embedding_model is None:
            try:
                self._embedding_model = HuggingFaceEmbeddings(model_name=self.model_name)
                logger.info(f"Embedding model {self.model_name} loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading embedding model: {e}")
                raise
        return self._embedding_model
    
    def get_collection_name(self, ingredient):
        """Generate a standardized collection name for an ingredient."""
        # Clean and normalize the name to avoid compatibility issues
        collection_name = ingredient.lower().replace(" ", "_").replace("-", "_")
        return f"collection_{collection_name}"
    
    def store_chunks(self, chunks: List[Document], ingredient: str, overwrite: bool = False) -> Dict[str, Any]:
        """
        Transforms the chunks into embeddings and stores them in the vector database.

        Args:
            chunks: List of chunks to store
            ingredient: Name of the ingredient to name the collection
            overwrite: If True, deletes the existing collection before adding new documents

        Returns:
            Dictionary containing the results of the operation
        """
        if not chunks:
            return {
                "status": "error",
                "message": "Aucun chunk à stocker.",
                "ingredient": ingredient,
                "stored_chunks": 0
            }
        
        # Ensure the database is properly configured
        if not self.db_setup.setup_database():
            return {
                "status": "error",
                "message": "Database setup failed.",
                "ingredient": ingredient,
                "stored_chunks": 0
            }
        
        collection_name = self.get_collection_name(ingredient)
        start_time = time.time()
        
        try:
            conn_string = self.db_setup.get_connection_string()
            
            vectorstore = PGVector.from_documents(
                documents=chunks,  
                embedding=self.embedding_model,
                collection_name=collection_name,
                connection=conn_string,
                use_jsonb=True,
                pre_delete_collection=overwrite
            )
            
            # Add documents to the vectorstore
            vectorstore.add_documents(chunks)
            
            end_time = time.time()
            processing_time = round(end_time - start_time, 2)
            
            logger.info(f"{len(chunks)} chunks stored in collection {collection_name} in {processing_time} seconds.")
            
            return {
                "status": "success",
                "message": f"{len(chunks)} chunks transformed into embeddings and successfully stored.",
                "ingredient": ingredient,
                "collection_name": collection_name,
                "stored_chunks": len(chunks),
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Error while storing embeddings: {e}")
            return {
                "status": "error",
                "message": f"Error while storing embeddings: {str(e)}",
                "ingredient": ingredient,
                "stored_chunks": 0
            }
    
    def list_collections(self) -> List[str]:
        """Lists all available collections in the database."""
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()
            
            # Check if the table langchain_pg_collection exists
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
        """Retrieve information about a specific collection."""
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()
            
            # Check if the table langchain_pg_collection exists
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
            
            # Check if the collection exists
            cursor.execute("SELECT 1 FROM langchain_pg_collection WHERE name = %s", (collection_name,))
            exists = cursor.fetchone()
            
            if not exists:
                return {
                    "status": "error",
                    "message": f"The collection {collection_name} does not exist.",
                    "collection_name": collection_name
                }
            
            # Gather metadata for the collection
            cursor.execute("SELECT cmetadata FROM langchain_pg_collection WHERE name = %s", (collection_name,))
            metadata_row = cursor.fetchone()
            metadata = metadata_row[0] if metadata_row else {}
            
            # The embedding table name may differ depending on the version of langchain_postgres
            # Try to dynamically determine the embedding table name
            embedding_table_name = f"langchain_pg_embedding_{collection_name}"
            alt_embedding_table_name = f"langchain_embedding_{collection_name}"
            
            # Check which embedding table exists
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
                    "note": "The collection exists but the embedding table was not found."
                }
            
            actual_table_name = table_result[0]
            
            # Count the number of documents in the collection
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


# Utility function to directly store article chunks for an ingredient
def store_article_chunks(chunks: List[Document], ingredient: str, overwrite: bool = False, config_file: str = "config.json") -> Dict[str, Any]:
    """
    Utility function to directly store article chunks in the vector database.
    
    Args:
        chunks: List of chunks to store
        ingredient: Name of the ingredient
        overwrite: If True, overwrite the existing collection
        config_file: Path to the configuration file
        
    Returns:
        Results of the storage operation
    """
    manager = EmbeddingManager(config_file)
    return manager.store_chunks(chunks, ingredient, overwrite)


def process_and_store_ingredient(ingredient: str, overwrite: bool = False) -> Dict[str, Any]:
    """
    Processes the articles for a given ingredient and stores the resulting chunks.

    Args:
        ingredient: Name of the ingredient
        overwrite: If True, overwrites the existing collection

    Returns:
        Combined results of the processing and storage operations
    """

    processing_results = process_downloaded_articles(ingredient)
    
    # Check if the processing was successful
    if processing_results.get("status") != "success":
        return {
            "status": "error",
            "message": f"Error while processing articles: {processing_results.get('message', 'Unknown error')}",
            "ingredient": ingredient,
            "processing_results": processing_results,
            "embedding_results": None
        }
    
    # Get the chunks from the processing results
    chunks = processing_results.get("chunks", [])
    
    # Stock the chunks in the vector database
    embedding_results = store_article_chunks(chunks, ingredient, overwrite)
    
    # Combine the results
    return {
        "status": embedding_results.get("status"),
        "message": f"Processing and storage completed. {embedding_results.get('stored_chunks')} chunks stored.",
        "ingredient": ingredient,
        "processing_results": processing_results,
        "embedding_results": embedding_results
    }