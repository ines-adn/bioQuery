import logging
import time
from typing import List, Dict, Any
import uuid
import os
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

# Importer les fonctions nécessaires des autres modules
from ..parsers.embedding_store import EmbeddingManager, load_config

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LlamaLLM(ChatOllama):
    def __init__(self, model_name: str, temperature: float = 0.5):
        super().__init__(
            model=model_name,
            temperature=temperature,
            base_url="http://localhost:11434",
            request_timeout=1800.0  # 30 minutes de timeout
        )

# Constantes pour les types de LLM
LLM_TYPE_OLLAMA = "ollama"
LLM_TYPE_OPENAI = "openai"

class IngredientSummarizer:
    """Classe pour résumer les informations sur un ingrédient à partir des chunks stockés."""
    
    def __init__(self, config_file="config.json", llm_type=None, model_name=None, temperature=None):
        self.config = load_config(config_file)
        self.embedding_manager = EmbeddingManager(config_file)
        
        # Utilisation plus explicite du type de LLM et du modèle
        self.llm_type = llm_type or self.config.get("llm_type", LLM_TYPE_OPENAI)
        
        # Paramètres pour OpenAI
        self.openai_model = model_name or self.config.get("openai_model", "gpt-3.5-turbo")
        
        # Paramètres pour Ollama
        self.ollama_model = model_name or self.config.get("ollama_model", "llama3.1")
        
        # Température commune
        self.temperature = temperature or self.config.get("llm_temperature", 0.5)
        
        self._llm = None
    
    @property
    def llm(self):
        """Charge le modèle LLM de manière paresseuse."""
        if self._llm is None:
            try:
                if self.llm_type == LLM_TYPE_OPENAI:
                    if not os.environ.get("OPENAI_API_KEY"):
                        logger.warning("Variable d'environnement OPENAI_API_KEY non définie. Vérifiez votre configuration.")
                    
                    self._llm = ChatOpenAI(
                        model=self.openai_model,
                        temperature=self.temperature
                    )
                    logger.info(f"Modèle OpenAI {self.openai_model} chargé avec succès.")
                else:  # Par défaut, utiliser Ollama
                    self._llm = LlamaLLM(model_name=self.ollama_model, temperature=self.temperature)
                    logger.info(f"Modèle Ollama {self.ollama_model} chargé avec succès.")
            except Exception as e:
                logger.error(f"Erreur lors du chargement du modèle: {e}")
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
            
            # Accéder directement à la base de données
            import psycopg2
            
            conn_string = self.embedding_manager.db_setup.get_connection_string()
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
        
        # Construire le prompt complet - version plus concise
        prompt = f"""Tu es un expert en vulgarisation scientifique spécialisé dans les ingrédients naturels.

            Je vais te fournir des extraits d'articles scientifiques sur l'ingrédient "{ingredient}". Rédige un paragraphe détaillé, en Français, dans un style encyclopédique similaire à Wikipedia qui synthétise les connaissances scientifiques actuelles sur cet ingrédient.

            OBJECTIF PRINCIPAL : Présenter une vue d'ensemble complète, équilibrée et factuelle qui valorise les découvertes scientifiques significatives tout en restant accessible aux non-spécialistes.

            Ton texte doit :

            1. Commencer par une introduction claire définissant l'ingrédient, son origine, sa composition principale et son contexte d'utilisation historique et contemporain

            2. Développer les propriétés scientifiquement établies en précisant systématiquement :
            - La nature des études (in vitro, animales, essais cliniques humains)
            - La qualité méthodologique des recherches (taille d'échantillon, durée, etc.)
            - Les mécanismes d'action identifiés lorsqu'ils sont mentionnés

            3. Présenter de façon équilibrée :
            - Les bénéfices démontrés avec leur niveau de preuve scientifique
            - Les limitations, effets indésirables ou précautions d'emploi
            - Les populations pouvant particulièrement bénéficier ou devant éviter cet ingrédient

            4. Contextualiser les résultats dans le paysage scientifique global (consensus, controverses ou recherches en cours)

            5. Conclure avec une synthèse objective de l'état actuel des connaissances 
            6. IMPORTANT : La façon dont l'humain peut utiliser l'ingrédient (voie cutanée, orale, etc.)

            Le style doit être :
            - Formel et encyclopédique, sans formulations commerciales ou subjectives
            - Précis, avec des tournures comme "des études suggèrent que..." ou "les recherches indiquent..." suivies de données spécifiques
            - Structuré avec des transitions logiques entre les différents aspects abordés
            - Accessible, en expliquant systématiquement les termes techniques

            IMPORTANT : Base-toi UNIQUEMENT sur les informations présentes dans les extraits fournis. N'invente pas de données ou de conclusions non mentionnées dans les sources. Si les informations sont limitées ou préliminaires, reflète fidèlement cette réalité.
            IMPORTANT : Ne donne des informations que sur l'ingrédient {ingredient} et ne fais pas de comparaisons avec d'autres ingrédients ou produits.
            
            IMPORTANT : Le paragraphe doit être en Français et compter 300 mots.
            Extraits sur {ingredient} :
            {context}

            Ton texte doit :

            1. Commencer par une introduction claire définissant l'ingrédient, son origine, sa composition principale et son contexte d'utilisation historique et contemporain

            2. Développer les propriétés scientifiquement établies en précisant systématiquement :
            - La nature des études (in vitro, animales, essais cliniques humains)
            - La qualité méthodologique des recherches (taille d'échantillon, durée, etc.)
            - Les mécanismes d'action identifiés lorsqu'ils sont mentionnés

            3. Présenter de façon équilibrée :
            - Les bénéfices démontrés avec leur niveau de preuve scientifique
            - Les limitations, effets indésirables ou précautions d'emploi
            - Les populations pouvant particulièrement bénéficier ou devant éviter cet ingrédient

            4. Contextualiser les résultats dans le paysage scientifique global (consensus, controverses ou recherches en cours)

            5. Conclure avec une synthèse objective de l'état actuel des connaissances 
            6. IMPORTANT : La façon dont l'humain peut utiliser l'ingrédient (voie cutanée, orale, etc.)

            Le style doit être :
            - Formel et encyclopédique, sans formulations commerciales ou subjectives
            - Précis, avec des tournures comme "des études suggèrent que..." ou "les recherches indiquent..." suivies de données spécifiques
            - Structuré avec des transitions logiques entre les différents aspects abordés
            - Accessible, en expliquant systématiquement les termes techniques

            IMPORTANT : Base-toi UNIQUEMENT sur les informations présentes dans les extraits fournis. N'invente pas de données ou de conclusions non mentionnées dans les sources. Si les informations sont limitées ou préliminaires, reflète fidèlement cette réalité.
            IMPORTANT : Ne donne des informations que sur l'ingrédient {ingredient} et ne fais pas de comparaisons avec d'autres ingrédients ou produits.
            
            IMPORTANT : Le paragraphe doit être en Français et compter 300 mots.
                """
        prompt_size = len(prompt)
        logger.info(f"Taille du prompt: {prompt_size} caractères, {len(prompt.split())} mots")
        
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
            # Pour les grands volumes, utiliser l'approche par lots si nécessaire
            if len(chunks) > 10 and self.llm_type == LLM_TYPE_OPENAI:
                summary = self.summarize_in_batches(ingredient, chunks)
            else:
                # Générer le prompt
                prompt = self.generate_summary_prompt(ingredient, chunks)
                
                # Créer le template pour LangChain
                prompt_template = PromptTemplate.from_template(prompt)
                
                # Méthode moderne recommandée pour remplacer LLMChain
                from langchain_core.runnables import RunnablePassthrough
                chain = prompt_template | self.llm | StrOutputParser()
                
                # Exécuter la chaîne
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
    
    def summarize_in_batches(self, ingredient: str, chunks: List[Document], batch_size: int = 50) -> str:
        """Génère un résumé progressivement par lots de documents."""
        # Résumer par lots
        batch_summaries = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            logger.info(f"Traitement du lot {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
            
            prompt = prompt = f"""Tu es un expert en vulgarisation scientifique spécialisé dans les ingrédients naturels.

            Je vais te fournir des extraits d'articles scientifiques sur l'ingrédient "{ingredient}". Rédige un paragraphe détaillé, en Français, dans un style encyclopédique similaire à Wikipedia qui synthétise les connaissances scientifiques actuelles sur cet ingrédient.

            OBJECTIF PRINCIPAL : Présenter une vue d'ensemble complète, équilibrée et factuelle qui valorise les découvertes scientifiques significatives tout en restant accessible aux non-spécialistes.

            Ton texte doit :

            1. Commencer par une introduction claire définissant l'ingrédient, son origine, sa composition principale et son contexte d'utilisation historique et contemporain

            2. Développer les propriétés scientifiquement établies en précisant systématiquement :
            - La nature des études (in vitro, animales, essais cliniques humains)
            - La qualité méthodologique des recherches (taille d'échantillon, durée, etc.)
            - Les mécanismes d'action identifiés lorsqu'ils sont mentionnés

            3. Présenter de façon équilibrée :
            - Les bénéfices démontrés avec leur niveau de preuve scientifique
            - Les limitations, effets indésirables ou précautions d'emploi
            - Les populations pouvant particulièrement bénéficier ou devant éviter cet ingrédient

            4. Contextualiser les résultats dans le paysage scientifique global (consensus, controverses ou recherches en cours)

            5. Conclure avec une synthèse objective de l'état actuel des connaissances 
            6. IMPORTANT : La façon dont l'humain peut utiliser l'ingrédient (voie cutanée, orale, etc.)

            Le style doit être :
            - Formel et encyclopédique, sans formulations commerciales ou subjectives
            - Précis, avec des tournures comme "des études suggèrent que..." ou "les recherches indiquent..." suivies de données spécifiques
            - Structuré avec des transitions logiques entre les différents aspects abordés
            - Accessible, en expliquant systématiquement les termes techniques

            IMPORTANT : Base-toi UNIQUEMENT sur les informations présentes dans les extraits fournis. N'invente pas de données ou de conclusions non mentionnées dans les sources. Si les informations sont limitées ou préliminaires, reflète fidèlement cette réalité.
            IMPORTANT : Ne donne des informations que sur l'ingrédient {ingredient} et ne fais pas de comparaisons avec d'autres ingrédients ou produits.
            
            IMPORTANT : Le paragraphe doit être en Français et compter 300 mots.
            Extraits sur {ingredient} :

            Ton texte doit :

            1. Commencer par une introduction claire définissant l'ingrédient, son origine, sa composition principale et son contexte d'utilisation historique et contemporain

            2. Développer les propriétés scientifiquement établies en précisant systématiquement :
            - La nature des études (in vitro, animales, essais cliniques humains)
            - La qualité méthodologique des recherches (taille d'échantillon, durée, etc.)
            - Les mécanismes d'action identifiés lorsqu'ils sont mentionnés

            3. Présenter de façon équilibrée :
            - Les bénéfices démontrés avec leur niveau de preuve scientifique
            - Les limitations, effets indésirables ou précautions d'emploi
            - Les populations pouvant particulièrement bénéficier ou devant éviter cet ingrédient

            4. Contextualiser les résultats dans le paysage scientifique global (consensus, controverses ou recherches en cours)

            5. Conclure avec une synthèse objective de l'état actuel des connaissances 
            6. IMPORTANT : La façon dont l'humain peut utiliser l'ingrédient (voie cutanée, orale, etc.)

            Le style doit être :
            - Formel et encyclopédique, sans formulations commerciales ou subjectives
            - Précis, avec des tournures comme "des études suggèrent que..." ou "les recherches indiquent..." suivies de données spécifiques
            - Structuré avec des transitions logiques entre les différents aspects abordés
            - Accessible, en expliquant systématiquement les termes techniques

            IMPORTANT : Base-toi UNIQUEMENT sur les informations présentes dans les extraits fournis. N'invente pas de données ou de conclusions non mentionnées dans les sources. Si les informations sont limitées ou préliminaires, reflète fidèlement cette réalité.
            IMPORTANT : Ne donne des informations que sur l'ingrédient {ingredient} et ne fais pas de comparaisons avec d'autres ingrédients ou produits.
            
            IMPORTANT : Le paragraphe doit être en Français et compter 300 mots.
                """
     
            prompt += "\n---\n".join([doc.page_content for doc in batch])
            
            batch_summary = self.llm.invoke(prompt).content
            batch_summaries.append(batch_summary)
        
        # Résumer les résumés
        summaries_joined = "\n\n".join([f"Résumé {i+1}:\n{summary}" for i, summary in enumerate(batch_summaries)])
        final_prompt = f"""Tu es un expert en vulgarisation scientifique spécialisé dans les ingrédients naturels.

        Voici plusieurs résumés partiels sur l'ingrédient "{ingredient}". Synthétise-les en un paragraphe détaillé (300 mots), en Français, dans un style encyclopédique similaire à Wikipedia.

        OBJECTIF PRINCIPAL : Présenter une vue d'ensemble complète, équilibrée et factuelle qui valorise les découvertes scientifiques significatives tout en restant accessible aux non-spécialistes.

        Ton texte doit :

        1. Commencer par une introduction claire définissant l'ingrédient, son origine, sa composition principale et son contexte d'utilisation historique et contemporain

        2. Développer les propriétés scientifiquement établies en précisant systématiquement :
        - La nature des études (in vitro, animales, essais cliniques humains)
        - La qualité méthodologique des recherches (taille d'échantillon, durée, etc.)
        - Les mécanismes d'action identifiés lorsqu'ils sont mentionnés

        3. Présenter de façon équilibrée :
        - Les bénéfices démontrés avec leur niveau de preuve scientifique
        - Les limitations, effets indésirables ou précautions d'emploi
        - Les populations pouvant particulièrement bénéficier ou devant éviter cet ingrédient

        4. Contextualiser les résultats dans le paysage scientifique global (consensus, controverses ou recherches en cours)

        5. Conclure avec une synthèse objective de l'état actuel des connaissances 
        6. IMPORTANT : La façon dont l'humain peut utiliser l'ingrédient (voie cutanée, orale, etc.)

        Le style doit être :
        - Formel et encyclopédique, sans formulations commerciales ou subjectives
        - Précis, avec des tournures comme "des études suggèrent que..." ou "les recherches indiquent..." suivies de données spécifiques
        - Structuré avec des transitions logiques entre les différents aspects abordés
        - Accessible, en expliquant systématiquement les termes techniques

        IMPORTANT : Base-toi UNIQUEMENT sur les informations présentes dans les résumés fournis. N'invente pas de données ou de conclusions non mentionnées dans les sources.
        IMPORTANT : Ne donne des informations que sur l'ingrédient {ingredient} et ne fais pas de comparaisons avec d'autres ingrédients ou produits.

        IMPORTANT : Le paragraphe doit être en Français et compter 300 mots.

        Résumés à synthétiser :
        {summaries_joined}
        """
        return self.llm.invoke(final_prompt).content
    
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


def generate_ingredient_summary(ingredient: str, save_to_file: bool = True, max_chunks: int = 50, 
                               llm_type: str = None, model_name: str = None) -> Dict[str, Any]:
    """
    Fonction utilitaire pour générer un résumé pour un ingrédient.
    Cette fonction peut être appelée directement depuis d'autres modules.
    
    Args:
        ingredient (str): Nom de l'ingrédient à résumer
        save_to_file (bool): Sauvegarder le résumé dans un fichier
        max_chunks (int): Nombre maximum de chunks à traiter
        llm_type (str): Type de LLM à utiliser ('openai' ou 'ollama')
        model_name (str): Nom du modèle à utiliser (dépend du type de LLM)
    
    Returns:
        Dict[str, Any]: Résultat du processus de génération de résumé
    """
    summarizer = IngredientSummarizer(
        llm_type=llm_type,
        model_name=model_name
    )
    
    # Générer le résumé
    summary_result = summarizer.summarize_ingredient(ingredient, max_chunks=max_chunks)
    
    # Si le résumé a été généré avec succès et qu'on veut le sauvegarder
    if summary_result.get("status") == "success" and save_to_file:
        save_result = summarizer.save_summary(
            ingredient=ingredient,
            summary=summary_result.get("summary", ""),
            summary_id=summary_result.get("summary_id")
        )
        
        summary_result["save_result"] = save_result
    
    return summary_result


if __name__ == "__main__":
    import sys
    import argparse
    
    # Créer un analyseur d'arguments en ligne de commande
    parser = argparse.ArgumentParser(description="Génère un résumé pour un ingrédient spécifique")
    parser.add_argument("ingredient", help="Nom de l'ingrédient à résumer")
    parser.add_argument("--no-save", action="store_true", help="Ne pas sauvegarder le résumé dans un fichier")
    parser.add_argument("--max-chunks", type=int, default=50, help="Nombre maximum de chunks à traiter")
    parser.add_argument("--llm-type", choices=[LLM_TYPE_OPENAI, LLM_TYPE_OLLAMA], default=None, 
                       help=f"Type de LLM à utiliser ({LLM_TYPE_OPENAI} ou {LLM_TYPE_OLLAMA})")
    parser.add_argument("--model", default=None, help="Nom du modèle à utiliser (ex: gpt-3.5-turbo pour OpenAI, llama3.1 pour Ollama)")
    
    args = parser.parse_args()
    
    # Déterminer le type de LLM à utiliser
    llm_type = args.llm_type
    
    # Si aucun type n'est spécifié mais qu'une clé OpenAI existe, utiliser OpenAI par défaut
    if llm_type is None:
        if os.environ.get("OPENAI_API_KEY"):
            llm_type = LLM_TYPE_OPENAI
            print("Variable d'environnement OPENAI_API_KEY détectée. Utilisation d'OpenAI par défaut.")
        else:
            llm_type = LLM_TYPE_OLLAMA
            print("Aucune variable d'environnement OPENAI_API_KEY détectée. Utilisation d'Ollama par défaut.")
    
    # Vérifier si une clé API OpenAI est nécessaire mais manquante
    if llm_type == LLM_TYPE_OPENAI and not os.environ.get("OPENAI_API_KEY"):
        print("ATTENTION: Le type LLM OpenAI est spécifié mais la variable d'environnement OPENAI_API_KEY n'est pas définie.")
        print("Vous pouvez la définir avec: export OPENAI_API_KEY=votre_clé_api")
        sys.exit(1)
    
    print(f"Génération du résumé pour l'ingrédient: {args.ingredient}")
    print(f"Sauvegarde dans un fichier: {'Non' if args.no_save else 'Oui'}")
    print(f"Nombre maximum de chunks: {args.max_chunks}")
    print(f"Type de LLM: {llm_type}")
    print(f"Modèle: {args.model or 'Par défaut selon le type de LLM'}")
    
    results = generate_ingredient_summary(
        ingredient=args.ingredient,
        save_to_file=not args.no_save,
        max_chunks=args.max_chunks,
        llm_type=llm_type,
        model_name=args.model
    )
    
    # Afficher les résultats
    print(f"\nStatut: {results['status']}")
    print(f"Message: {results['message']}")
    
    if results.get("status") == "success":
        print(f"\nRésumé pour {args.ingredient}:")
        print("=" * 80)
        print(results.get("summary", ""))
        print("=" * 80)
        
        print("\nInformations de traitement:")
        print(f"- Chunks traités: {results.get('chunks_processed', 0)}")
        print(f"- Temps de traitement: {results.get('processing_time', 0)} secondes")
        
        if not args.no_save and results.get("save_result", {}).get("status") == "success":
            print(f"\nRésumé sauvegardé dans: {results.get('save_result', {}).get('filepath', '')}")