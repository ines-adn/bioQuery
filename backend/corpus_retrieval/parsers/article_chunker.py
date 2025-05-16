import os
import json
import logging
from typing import List, Dict, Any
import uuid

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuration du logging
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

class ArticleProcessor:
    """Classe pour traiter les articles PDF téléchargés pour un ingrédient spécifique."""
    
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
        
        # Initialiser le chemin du dossier si un ingrédient est fourni
        if self.ingredient:
            self.folder_path = os.path.join(self.base_dir, "backend", "data", "articles", self.ingredient)
    
    def process_article(self, file_path) -> List[Document]:
        """Traite un seul article PDF."""
        if not os.path.exists(file_path) or not file_path.endswith(".pdf"):
            logger.error(f"Fichier invalide: {file_path}")
            return []
        
        try:
            # Charger le PDF
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Ajouter des métadonnées
            file_name = os.path.basename(file_path)
            
            for doc in documents:
                doc.metadata["file_name"] = file_name
                doc.metadata["source"] = file_path
                doc.metadata["ingredient"] = self.ingredient
                doc.metadata["id"] = str(uuid.uuid4())
            
            # Chunking
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
            )
            
            chunks = text_splitter.split_documents(documents)
            logger.info(f"Article {file_name} traité avec succès: {len(chunks)} chunks créés.")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement de l'article {file_path}: {e}")
            return []
    
    def process_ingredient_articles(self) -> Dict[str, Any]:
        """Traite tous les articles PDF pour l'ingrédient spécifié."""
        if not os.path.exists(self.folder_path):
            logger.error(f"Le dossier {self.folder_path} n'existe pas.")
            return {
                "status": "error",
                "message": f"Le dossier des articles pour l'ingrédient {self.ingredient} n'existe pas.",
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
        
        # Rechercher les fichiers PDF commençant par le nom de l'ingrédient
        prefix = f"{self.ingredient}_article_"
        pdf_files = [f for f in os.listdir(self.folder_path) 
                    if f.endswith(".pdf") and f.startswith(prefix)]
        
        if not pdf_files:
            logger.warning(f"Aucun fichier PDF trouvé dans le dossier {self.folder_path}")
            results["status"] = "warning"
            results["message"] = f"Aucun fichier PDF trouvé pour l'ingrédient {self.ingredient}."
            return results
        
        # Traiter chaque PDF
        for filename in pdf_files:
            file_path = os.path.join(self.folder_path, filename)
            
            chunks = self.process_article(file_path)
            
            if chunks:
                results["chunks"].extend(chunks)
                results["processed_files"] += 1
                results["total_chunks"] += len(chunks)
            else:
                results["errors"].append(f"Échec du traitement de {filename}")
        
        # Mise à jour du statut final
        if results["processed_files"] > 0:
            results["status"] = "success"
            results["message"] = f"Traitement terminé pour {self.ingredient}. {results['processed_files']} fichiers traités, {results['total_chunks']} chunks créés."
        else:
            results["status"] = "error"
            results["message"] = f"Aucun fichier n'a pu être traité pour l'ingrédient {self.ingredient}."
        
        return results

    
def process_downloaded_articles(ingredient, chunk_size=1000, chunk_overlap=200):
    """
    Fonction d'utilité pour traiter les articles téléchargés pour un ingrédient.
    Cette fonction peut être appelée directement depuis d'autres modules.
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
        # Obtenir l'ingrédient depuis les arguments de ligne de commande
        ingredient = sys.argv[1]
        
        print(f"Traitement des articles pour l'ingrédient: {ingredient}")
        results = process_downloaded_articles(ingredient)
        
        # Afficher les résultats
        print(f"Statut: {results['status']}")
        print(f"Message: {results['message']}")
        print(f"Fichiers traités: {results['processed_files']}")
        print(f"Total des chunks: {results['total_chunks']}")
        
        if results['errors']:
            print("\nErreurs:")
            for error in results['errors']:
                print(f"- {error}")
                
        if 'chunks' in results and results['chunks']:
            print("\nPremier chunk (extrait):")
            print(results['chunks'][0].page_content[:200] + "...")
    else:
        print("Usage: python process_articles.py <ingredient>")