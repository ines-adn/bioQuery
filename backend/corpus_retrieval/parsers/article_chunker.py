import os
import json
import logging
from typing import List, Dict, Any
import uuid

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_file="config.json"):
    """Loads config from a JSON file."""

    # File loading
    try:
        with open(config_file, "r") as file:
            config = json.load(file)
            return config
    
    # Error handling
    except FileNotFoundError:
        logger.error(f"The file {config_file} was not found.")
        return {}
    except json.JSONDecodeError:
        logger.error(f"Error reading the file {config_file}.")
        return {}

class ArticleProcessor:
    """Class to process downloaded PDF articles for a specific ingredient."""
    
    def __init__(
        self,
        ingredient=None,
        chunk_size=1000,
        chunk_overlap=200,
        config_file="config.json"
    ):
        self.ingredient = ingredient
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.config = load_config(config_file)
        self.base_dir = self.config.get("base_dir", "")
        
        # Initialize the folder path based on the ingredient
        if self.ingredient:
            self.folder_path = os.path.join(self.base_dir, "backend", "data", "articles", self.ingredient)
    
    def process_article(self, file_path) -> List[Document]:
        """
        Process a single PDF article file and return a list of chunked Document objects.

        Args:
            file_path (str): Path to the PDF file.

        Returns:
            List[Document]: List of chunked Document objects with metadata.

        """
        if not os.path.exists(file_path) or not file_path.endswith(".pdf"):
            logger.error(f"Fichier invalide: {file_path}")
            return []
        
        try:
            # Load the PDF file
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Add metadata to each document
            file_name = os.path.basename(file_path)
            
            for doc in documents:
                doc.metadata["file_name"] = file_name
                doc.metadata["source"] = file_path
                doc.metadata["ingredient"] = self.ingredient
                doc.metadata["id"] = str(uuid.uuid4())
            
            # Chunk the documents with the specified size and overlap
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
            )
            
            chunks = text_splitter.split_documents(documents)
            logger.info(f"Article {file_name} processed successfully: {len(chunks)} chunks created.")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing article {file_path}: {e}")
            return []
    
    def process_ingredient_articles(self) -> Dict[str, Any]:
        """Processes all PDF articles for the specified ingredient."""
        if not os.path.exists(self.folder_path):
            logger.error(f"The folder {self.folder_path} does not exist.")
            return {
                "status": "error",
                "message": f"The articles folder for ingredient {self.ingredient} does not exist.",
                "ingredient": self.ingredient
            }
        
        results = {
            "status": "in_progress",
            "ingredient": self.ingredient,
            "processed_files": 0,
            "total_chunks": 0,
            "errors": [],
            "chunks": []
        }
        
        # Search for PDF files in the folder starting with the ingredient prefix
        prefix = f"{self.ingredient}_article_"
        pdf_files = [f for f in os.listdir(self.folder_path) 
                    if f.endswith(".pdf") and f.startswith(prefix)]
        
        if not pdf_files:
            logger.warning(f"No PDF files found in the folder {self.folder_path}")
            results["status"] = "warning"
            results["message"] = f"No PDF files found for ingredient {self.ingredient}."
            return results
        
        # Process each PDF file
        for filename in pdf_files:
            file_path = os.path.join(self.folder_path, filename)
            
            chunks = self.process_article(file_path)
            
            if chunks:
                results["chunks"].extend(chunks)
                results["processed_files"] += 1
                results["total_chunks"] += len(chunks)
            else:
                results["errors"].append(f"Failed to process {filename}")
        
        # Update the status based on the processing results
        if results["processed_files"] > 0:
            results["status"] = "success"
            results["message"] = f"Processing completed for {self.ingredient}. {results['processed_files']} files processed, {results['total_chunks']} chunks created."
        else:
            results["status"] = "error"
            results["message"] = f"No files could be processed for ingredient {self.ingredient}."
        
        return results

    
def process_downloaded_articles(ingredient, chunk_size=1000, chunk_overlap=200):
    """
    Utility function to process downloaded articles for a given ingredient.
    This function can be called directly from other modules.
    """
    processor = ArticleProcessor(
        ingredient=ingredient,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    return processor.process_ingredient_articles()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Obtain the ingredient from command line arguments
        ingredient = sys.argv[1]
        
        print(f"Processing articles for ingredient: {ingredient}")
        results = process_downloaded_articles(ingredient)
        
        # Afficher les résultats
        print(f"Status: {results['status']}")
        print(f"Message: {results['message']}")
        print(f"Processed files: {results['processed_files']}")
        print(f"Total chunks: {results['total_chunks']}")
        
        if results['errors']:
            print("\nErreurs:")
            for error in results['errors']:
                print(f"- {error}")
                
        if 'chunks' in results and results['chunks']:
            print("\nFirst chunk (excerpt):")
            print(results['chunks'][0].page_content[:200] + "...")
    else:
        print("Usage: python process_articles.py <ingredient>")