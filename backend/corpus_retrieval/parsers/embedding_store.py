import logging
from typing import List, Dict, Any
import time

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from .article_chunker import process_downloaded_articles
from ...utils import load_config

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
    
    def create_database_if_not_exists(self):
        """Create the database if it doesn't exist."""
        try:
            dbname = self.db_config.get("database", "vectordb")
            
            # Connect to the default 'postgres' database to create our target database
            conn = psycopg2.connect(self.get_connection_string(include_dbname=False) + "/postgres")
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Check if the database exists
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
            exists = cursor.fetchone()
            
            if not exists:
                # Create the database
                cursor.execute(f'CREATE DATABASE "{dbname}"')
                logger.info(f"Database '{dbname}' created successfully.")
            else:
                logger.info(f"Database '{dbname}' already exists.")
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error creating database: {e}")
            return False

    def setup_database(self):
        """Set of SQL queries that check the existence of the database, verify database connectivity, 
        and prepare the necessary schema for PGVector."""
        try:
            # Ensure the database exists
            if not self.create_database_if_not_exists():
                return False
            
            # Connect to target database
            conn = psycopg2.connect(self.get_connection_string())
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Check that the connection is successful
            cursor.execute("SELECT 1")
            
            # Check if the pgvector extension is installed
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Create the collections table with the correct structure (according to PGVector)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS langchain_pg_collection (
                uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name TEXT UNIQUE NOT NULL,
                cmetadata JSONB
            )   
            """)
            
            # Create the embeddings table with the correct structure (according to PGVector)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                collection_id UUID REFERENCES langchain_pg_collection(uuid) ON DELETE CASCADE,
                embedding vector,
                document TEXT,
                cmetadata JSONB
            )
            """)
            
            # Create an index on the embedding column for better performance
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS langchain_pg_embedding_embedding_idx 
            ON langchain_pg_embedding USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
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
    
    def store_chunks_and_embeddings(self, chunks: List[Document], ingredient: str, overwrite: bool = False) -> Dict[str, Any]:
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
                "message": "No chunks to store.",
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
                logger.info("The langchain_pg_collection table does not exist yet.")
                cursor.close()
                conn.close()
                return []
            
            cursor.execute("SELECT name FROM langchain_pg_collection")
            collections = [row[0] for row in cursor.fetchall()]
            cursor.close()
            conn.close()
            return collections
        except Exception as e:
            logger.error(f"Error retrieving collections: {e}")
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
                cursor.close()
                conn.close()
                return {
                    "status": "error",
                    "message": "The collections table does not exist yet.",
                    "collection_name": collection_name
                }
            
            # Check if the collection exists (using column name 'name')
            cursor.execute("SELECT uuid, cmetadata FROM langchain_pg_collection WHERE name = %s", (collection_name,))
            collection_row = cursor.fetchone()
            
            if not collection_row:
                cursor.close()
                conn.close()
                return {
                    "status": "error",
                    "message": f"The collection {collection_name} does not exist.",
                    "collection_name": collection_name
                }
            
            collection_uuid, metadata = collection_row
            
            # Check if the embeddings table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'langchain_pg_embedding'
                )
            """)
            
            embedding_table_exists = cursor.fetchone()[0]
            
            if not embedding_table_exists:
                cursor.close()
                conn.close()
                return {
                    "status": "success",
                    "collection_name": collection_name,
                    "document_count": 0,
                    "metadata": metadata,
                    "note": "The collection exists but the embedding table was not found."
                }
            
            # Count the number of documents in the collection
            cursor.execute("SELECT COUNT(*) FROM langchain_pg_embedding WHERE collection_id = %s", (collection_uuid,))
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
            logger.error(f"Error retrieving information for collection {collection_name}: {e}")
            return {
                "status": "error",
                "message": f"Error: {str(e)}",
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
    return manager.store_chunks_and_embeddings(chunks, ingredient, overwrite)


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
    
    # Store the chunks in the vector database
    embedding_results = store_article_chunks(chunks, ingredient, overwrite)
    
    # Combine the results
    return {
        "status": embedding_results.get("status"),
        "message": f"Processing and storage completed. {embedding_results.get('stored_chunks')} chunks stored.",
        "ingredient": ingredient,
        "processing_results": processing_results,
        "embedding_results": embedding_results
    }