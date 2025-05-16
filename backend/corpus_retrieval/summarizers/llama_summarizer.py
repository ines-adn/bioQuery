import logging
import time
from typing import List, Dict, Any
import uuid
import os
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_ollama import ChatOllama

# Importer les fonctions nécessaires des autres modules
from backend.corpus_retrieval.parsers.embedding_store import EmbeddingManager, load_config

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LlamaLLM(ChatOllama):
    def __init__(self, model_name: str, temperature: float = 0.5):
        super().__init__(
            model=model_name,
            temperature=temperature,
            base_url="http://localhost:11434",
        )

class IngredientSummarizer:
    """Classe pour résumer les informations sur un ingrédient à partir des chunks stockés."""
    
    def __init__(self, config_file="backend/config.json"):
        self.config = load_config(config_file)
        self.embedding_manager = EmbeddingManager(config_file)
        self.model_name = self.config.get("llm_model", "llama3.1")
        self.temperature = self.config.get("llm_temperature", 0.5)
        self._llm = None
    
    @property
    def llm(self):
        """Charge le modèle LLM de manière paresseuse."""
        if self._llm is None:
            try:
                self._llm = LlamaLLM(model_name=self.model_name, temperature=self.temperature)
                logger.info(f"Modèle LLM {self.model_name} chargé avec succès.")
            except Exception as e:
                logger.error(f"Erreur lors du chargement du modèle LLM: {e}")
                raise
        return self._llm
    
    def retrieve_chunks(self, ingredient: str, limit: int = 100) -> List[Document]:
        """Récupère les chunks stockés pour un ingrédient donné."""
        collection_name = self.embedding_manager.get_collection_name(ingredient)
        
        try:
            # Vérifier si la collection existe
            collection_info = self.embedding_manager.get_collection_info(collection_name)
            if collection_info.get("status") != "success":
                logger.error(f"Collection {collection_name} non trouvée: {collection_info.get('message')}")
                return []
            
            # Établir une connexion à la base de données
            conn_string = self.embedding_manager.db_setup.get_connection_string()
            
            # Accéder directement à la base de données
            import psycopg2
            from langchain_core.documents import Document
            
            conn = psycopg2.connect(conn_string)
            cursor = conn.cursor()
            
            # Récupérer l'ID de la collection
            cursor.execute("SELECT uuid FROM langchain_pg_collection WHERE name = %s", (collection_name,))
            collection_id = cursor.fetchone()
            
            if not collection_id:
                logger.error(f"Collection {collection_name} non trouvée dans la base de données")
                return []
            
            collection_id = collection_id[0]
            
            # Récupérer les documents de cette collection
            cursor.execute("""
                SELECT document, cmetadata FROM langchain_pg_embedding 
                WHERE collection_id = %s
                LIMIT %s
            """, (collection_id, limit))
            
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            
            # Convertir les résultats en objets Document
            documents = []
            for row in rows:
                content = row[0]
                metadata = row[1] if row[1] else {}
                documents.append(Document(page_content=content, metadata=metadata))
            
            logger.info(f"Récupération de {len(documents)} documents pour {ingredient}")
            return documents
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des chunks pour {ingredient}: {e}")
            return []
    
    def generate_summary_prompt(self, ingredient: str, chunks: List[Document]) -> str:
        """Génère le prompt pour le résumé à partir des chunks."""
        # Extraire le contenu des chunks
        contents = [doc.page_content for doc in chunks]
        
        # Joindre les contenus avec un séparateur
        context = "\n---\n".join(contents)
        
        # Construire le prompt complet
        prompt = f"""
            Tu es un expert dans la recherche scientifique sur les ingrédients naturels et leurs applications en cosmétique, aromathérapie, et santé naturelle.

            Je vais te fournir des extraits d'articles scientifiques sur l'ingrédient "{ingredient}". Sur la base de ces informations, prépare un résumé complet et organisé qui couvre:

            1. Description générale de l'ingrédient et ses origines
            2. Composition chimique et principes actifs principaux
            3. Propriétés et bénéfices prouvés scientifiquement
            4. Applications thérapeutiques et cosmétiques 
            5. Précautions d'emploi et contre-indications éventuelles
            6. État actuel de la recherche et perspectives futures

            Assure-toi que ton résumé est:
            - Basé uniquement sur les informations fournies dans les extraits
            - Organisé clairement avec des sous-sections
            - Factuel et évite les exagérations
            - Précis dans la terminologie scientifique

            Voici les extraits d'articles scientifiques sur {ingredient}:

            {context}
            """
        return prompt
    
    def summarize_ingredient(self, ingredient: str, max_chunks: int = 50) -> Dict[str, Any]:
        """Génère un résumé complet sur un ingrédient à partir des chunks stockés."""
        start_time = time.time()
        
        # Récupérer les chunks
        chunks = self.retrieve_chunks(ingredient, limit=max_chunks)
        if not chunks:
            return {
                "status": "error",
                "message": f"Aucun chunk trouvé pour l'ingrédient {ingredient}.",
                "ingredient": ingredient
            }
        
        try:
            # Générer le prompt
            prompt = self.generate_summary_prompt(ingredient, chunks)
            
            # Créer le template pour LangChain
            prompt_template = PromptTemplate.from_template(prompt)
            
            # Créer et exécuter la chaîne LLM
            chain = LLMChain(
                llm=self.llm,
                prompt=prompt_template,
                output_parser=StrOutputParser()
            )
            
            # Exécuter la chaîne (sans entrées supplémentaires car tout est dans le template)
            summary = chain.invoke({})
            
            end_time = time.time()
            processing_time = round(end_time - start_time, 2)
            
            return {
                "status": "success",
                "message": f"Résumé généré avec succès pour {ingredient}.",
                "ingredient": ingredient,
                "summary": summary,
                "chunks_processed": len(chunks),
                "processing_time": processing_time,
                "summary_id": str(uuid.uuid4())
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du résumé pour {ingredient}: {e}")
            return {
                "status": "error",
                "message": f"Erreur lors de la génération du résumé: {str(e)}",
                "ingredient": ingredient
            }
    
    def save_summary(self, ingredient: str, summary: str, summary_id: str = None) -> Dict[str, Any]:
        """Sauvegarde le résumé généré dans un fichier."""
        if summary_id is None:
            summary_id = str(uuid.uuid4())
        
        try:
            # Créer le dossier des résumés s'il n'existe pas
            import os
            summary_dir = os.path.join(self.config.get("base_dir", ""), "backend", "data", "summaries")
            os.makedirs(summary_dir, exist_ok=True)
            
            # Créer le nom de fichier
            filename = f"{ingredient}_summary_{summary_id}.md"
            filepath = os.path.join(summary_dir, filename)
            
            # Écrire le résumé dans le fichier
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(summary)
            
            logger.info(f"Résumé sauvegardé dans {filepath}")
            
            return {
                "status": "success",
                "message": "Résumé sauvegardé avec succès.",
                "ingredient": ingredient,
                "summary_id": summary_id,
                "filepath": filepath
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du résumé pour {ingredient}: {e}")
            return {
                "status": "error",
                "message": f"Erreur lors de la sauvegarde du résumé: {str(e)}",
                "ingredient": ingredient
            }


def generate_ingredient_summary(ingredient: str, save_to_file: bool = True, max_chunks: int = 50) -> Dict[str, Any]:
    """
    Fonction utilitaire pour générer un résumé pour un ingrédient.
    Cette fonction peut être appelée directement depuis d'autres modules.
    """
    summarizer = IngredientSummarizer()
    
    # Générer le résumé
    summary_result = summarizer.summarize_ingredient(ingredient, max_chunks=max_chunks)
    
    # Si le résumé a été généré avec succès et qu'on veut le sauvegarder
    if summary_result.get("status") == "success" and save_to_file:
        save_result = summarizer.save_summary(
            ingredient=ingredient,
            summary=summary_result.get("summary", ""),
            summary_id=summary_result.get("summary_id")
        )
        
        # Ajouter les informations de sauvegarde au résultat
        summary_result["save_result"] = save_result
    
    return summary_result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Obtenir l'ingrédient depuis les arguments de ligne de commande
        ingredient = sys.argv[1]
        
        # Option pour sauvegarder ou non le résumé
        save_to_file = "--no-save" not in sys.argv
        
        print(f"Génération du résumé pour l'ingrédient: {ingredient}")
        print(f"Sauvegarde dans un fichier: {'Oui' if save_to_file else 'Non'}")
        
        results = generate_ingredient_summary(ingredient, save_to_file=save_to_file)
        
        # Afficher les résultats
        print(f"\nStatut: {results['status']}")
        print(f"Message: {results['message']}")
        
        if results.get("status") == "success":
            print(f"\nRésumé pour {ingredient}:")
            print("=" * 80)
            print(results.get("summary", ""))
            print("=" * 80)
            
            print("\nInformations de traitement:")
            print(f"- Chunks traités: {results.get('chunks_processed', 0)}")
            print(f"- Temps de traitement: {results.get('processing_time', 0)} secondes")
            
            if save_to_file and results.get("save_result", {}).get("status") == "success":
                print(f"\nRésumé sauvegardé dans: {results.get('save_result', {}).get('filepath', '')}")
    else:
        print("Usage: python llama_summarizer.py <ingredient> [--no-save]")