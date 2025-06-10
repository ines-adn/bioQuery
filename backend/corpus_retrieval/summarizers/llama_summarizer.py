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

# Import required functions from other modules
from ..parsers.embedding_store import EmbeddingManager
from utils import load_config

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LlamaLLM(ChatOllama):
    def __init__(self, model_name: str, temperature: float = 0.5):
        super().__init__(
            model=model_name,
            temperature=temperature,
            base_url="http://localhost:11434",
            request_timeout=1800.0,  # 30 minutes timeout
            num_predict=300,
            top_k=10,
            top_p=0.9,
        )

# Constants for LLM types
LLM_TYPE_OLLAMA = "ollama"
LLM_TYPE_OPENAI = "openai"

class IngredientSummarizer:
    """Class to summarize information about an ingredient from stored chunks."""
    
    def __init__(self, config_file="config.json", llm_type=None, model_name=None, temperature=None, 
             max_tokens_ollama=2000, max_chunks_ollama=10):
        
        self.config = load_config(config_file)
        self.embedding_manager = EmbeddingManager(config_file)
        
        # Explicit use of LLM type and model
        self.llm_type = llm_type or self.config.get("llm_type", LLM_TYPE_OPENAI)
        
        # OpenAI parameters
        self.openai_model = model_name or self.config.get("openai_model", "gpt-3.5-turbo")
        
        # Ollama parameters
        self.ollama_model = model_name or self.config.get("ollama_model", "llama3.1")
        
        # Common temperature
        self.temperature = temperature or self.config.get("llm_temperature", 0.5)
        
        # Ollama-specific parameters
        self.max_tokens_ollama = max_tokens_ollama
        self.max_chunks_ollama = max_chunks_ollama
        
        self._llm = None

    @property
    def llm(self):
        """Lazily load the LLM model with the correct parameters."""
        if self._llm is None:
            try:
                if self.llm_type == LLM_TYPE_OPENAI:
                    if not os.environ.get("OPENAI_API_KEY"):
                        logger.warning("Environment variable OPENAI_API_KEY not set. Check your configuration.")
                    
                    self._llm = ChatOpenAI(
                        model=self.openai_model,
                        temperature=self.temperature
                    )
                    logger.info(f"OpenAI model {self.openai_model} loaded successfully.")
                else:  # Default to Ollama
                    self._llm = LlamaLLM(
                        model_name=self.ollama_model, 
                        temperature=self.temperature
                    )
                    # Note: ChatOllama does not have a direct max_tokens parameter, so we control
                    # this by reducing the size of the data sent
                    logger.info(f"Ollama model {self.ollama_model} loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                raise
        return self._llm
    
    def retrieve_chunks(self, ingredient: str, limit: int = 100) -> List[Document]:
        """Retrieve stored chunks for a given ingredient with adaptive limit."""

        # Use an adaptive limit depending on LLM type
        if limit is None:
            if self.llm_type == LLM_TYPE_OLLAMA:
                limit = self.max_chunks_ollama
            else:
                limit = 100  # Default limit for OpenAI
        
        logger.info(f"Retrieving chunks for {ingredient} with limit: {limit}")
        collection_name = self.embedding_manager.get_collection_name(ingredient)
        
        try:
            # Check if the collection exists
            collection_info = self.embedding_manager.get_collection_info(collection_name)
            if collection_info.get("status") != "success":
                logger.error(f"Collection {collection_name} not found: {collection_info.get('message')}")
                return []
            
            # Directly access the database
            import psycopg2
            
            conn_string = self.embedding_manager.db_setup.get_connection_string()
            conn = psycopg2.connect(conn_string)
            cursor = conn.cursor()
            
            # Retrieve the collection ID
            cursor.execute("SELECT uuid FROM langchain_pg_collection WHERE name = %s", (collection_name,))
            collection_id = cursor.fetchone()
            
            if not collection_id:
                logger.error(f"Collection {collection_name} not found in database")
                return []
            
            collection_id = collection_id[0]
            
            # Retrieve documents from this collection
            cursor.execute("""
                SELECT document, cmetadata FROM langchain_pg_embedding 
                WHERE collection_id = %s
                LIMIT %s
            """, (collection_id, limit))
            
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            
            # Convert results to Document objects
            documents = []
            for row in rows:
                content = row[0]
                metadata = row[1] if row[1] else {}
                documents.append(Document(page_content=content, metadata=metadata))
            
            logger.info(f"Retrieved {len(documents)} documents for {ingredient}")
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving chunks for {ingredient}: {e}")
            return []
    
    def generate_summary_prompt(self, ingredient: str, chunks: List[Document], language: str = "fr") -> str:
        """Generate the summary prompt from chunks with optimized size."""
        # Extract chunk contents
        contents = [doc.page_content for doc in chunks]
        
        # Reduce content size for Ollama
        if self.llm_type == LLM_TYPE_OLLAMA:
            # Limit each chunk to about 300 characters
            contents = [content[:200] + "..." if len(content) > 200 else content 
                    for content in contents]
        
        # Join contents with a separator
        context = "\n---\n".join(contents)
        
        # Determine output language
        output_language = "Français" if language == "fr" else "English"
        is_english = language == "en"
        
        # Build the prompt - simplified version for Ollama BUT in the correct language
        if self.llm_type == LLM_TYPE_OLLAMA:
            if is_english:
                prompt = f"""You are a scientific expert. Summarize the scientific information about the ingredient "{ingredient}" in English, maximum 200 words.

    Points to include:
    1. Definition, origin and composition
    2. Scientifically proven properties and benefits
    3. Precautions and limitations
    4. Usage (topical, oral, etc.)

    Do not include introductory sentences or conclusions. Just provide the summary directly.

    IMPORTANT: Write your response entirely in English.

    Context:
    {context}
    """
            else:
                prompt = f"""Tu es un expert scientifique. Résume les informations scientifiques sur l'ingrédient "{ingredient}" en français, en 200 mots maximum.

    Points à inclure:
    1. Définition, origine et composition
    2. Propriétés et bénéfices scientifiquement prouvés
    3. Précautions et limitations
    4. Utilisation (voie cutanée, orale, etc.)

    Ne place pas de phrases introductives ou de conclusions. Donne simplement le résumé directement.

    IMPORTANT: Écris ta réponse entièrement en français.

    Contexte:
    {context}
    """
        else:
            # Use the full prompt for OpenAI - also adapted according to the language
            if is_english:
                prompt = f"""You are an expert in scientific popularization specialized in natural ingredients.

    I will provide you with scientific article extracts about the ingredient "{ingredient}". Write a detailed paragraph in English, in an encyclopedic style similar to Wikipedia that synthesizes current scientific knowledge about this ingredient.

    MAIN OBJECTIVE: Present a comprehensive, balanced and factual overview that highlights significant scientific discoveries while remaining accessible to non-specialists.

    Your text should:

    1. Begin with a clear introduction defining the ingredient, its origin, main composition and historical and contemporary usage context

    2. Develop scientifically established properties by systematically specifying:
    - The nature of studies (in vitro, animal studies, human clinical trials)
    - Methodological quality of research (sample size, duration, etc.)
    - Identified mechanisms of action when mentioned

    3. Present in a balanced way:
    - Demonstrated benefits with their level of scientific evidence
    - Limitations, adverse effects or precautions for use
    - Populations that may particularly benefit or should avoid this ingredient

    4. Contextualize results in the global scientific landscape (consensus, controversies or ongoing research)

    5. Conclude with an objective synthesis of the current state of knowledge
    6. IMPORTANT: How humans can use the ingredient (topical, oral, etc.)

    The style should be:
    - Formal and encyclopedic, without commercial or subjective formulations
    - Precise, with phrases like "studies suggest that..." or "research indicates..." followed by specific data
    - Structured with logical transitions between different aspects covered
    - Accessible, systematically explaining technical terms

    IMPORTANT: Base yourself ONLY on information present in the provided extracts. Do not invent data or conclusions not mentioned in the sources. If information is limited or preliminary, faithfully reflect this reality.
    IMPORTANT: Only provide information about the ingredient {ingredient} and do not make comparisons with other ingredients or products.

    IMPORTANT: The paragraph must be in English and be 300 words long.

    Extracts about {ingredient}:
    {context}
    """
            else:
                # Original French prompt for OpenAI
                prompt = f"""Tu es un expert en vulgarisation scientifique spécialisé dans les ingrédients naturels.

    Je vais te fournir des extraits d'articles scientifiques sur l'ingrédient "{ingredient}". Rédige un paragraphe détaillé, en français, dans un style encyclopédique similaire à Wikipedia qui synthétise les connaissances scientifiques actuelles sur cet ingrédient.

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

    IMPORTANT : Le paragraphe doit être en français et compter 300 mots.

    Extraits sur {ingredient} :
    {context}
    """
        
        prompt_size = len(prompt)
        logger.info(f"Taille du prompt: {prompt_size} caractères, {len(prompt.split())} mots")
        logger.info(f"Langue demandée: {output_language}, LLM type: {self.llm_type}")
        
        return prompt
    
    def summarize_ingredient(self, ingredient: str, max_chunks: int = 50, language: str = "fr") -> Dict[str, Any]:
        """Génère un résumé complet sur un ingrédient à partir des chunks stockés."""
        start_time = time.time()
        
        chunks = self.retrieve_chunks(ingredient, limit=max_chunks)
        if not chunks:
            return {
                "status": "error",
                "message": f"Aucun chunk trouvé pour l'ingrédient {ingredient}.",
                "ingredient": ingredient,
                "language": language
            }
        
        try:
    
            if len(chunks) > 10 and self.llm_type == LLM_TYPE_OPENAI:
                summary = self.summarize_in_batches(ingredient, chunks, language=language)
            else:
                # Generate the prompt for the LLM
                prompt = self.generate_summary_prompt(ingredient, chunks, language=language)
                
                # Create the prompt template
                prompt_template = PromptTemplate.from_template(prompt)
                
                chain = prompt_template | self.llm | StrOutputParser()
                
                summary = chain.invoke({})
            
            end_time = time.time()
            processing_time = round(end_time - start_time, 2)
            
            return {
                "status": "success",
                "message": f"Résumé généré avec succès pour {ingredient} en {language}.",
                "ingredient": ingredient,
                "summary": summary,
                "chunks_processed": len(chunks),
                "processing_time": processing_time,
                "summary_id": str(uuid.uuid4()),
                "language": language
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du résumé pour {ingredient}: {e}")
            return {
                "status": "error",
                "message": f"Erreur lors de la génération du résumé: {str(e)}",
                "ingredient": ingredient,
                "language": language
            }
    
    def summarize_in_batches(self, ingredient: str, chunks: List[Document], batch_size: int = 50, language: str = "fr") -> str:
        """Generate a summary progressively in batches of documents."""

        is_english = language == "en"
        
        batch_summaries = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            logger.info(f"Traitement du lot {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
            
            if is_english:
                prompt = f"""You are an expert in scientific popularization specialized in natural ingredients.

    I will provide you with scientific article extracts about the ingredient "{ingredient}". Write a detailed paragraph in English, in an encyclopedic style similar to Wikipedia that synthesizes current scientific knowledge about this ingredient.

    MAIN OBJECTIVE: Present a comprehensive, balanced and factual overview that highlights significant scientific discoveries while remaining accessible to non-specialists.

    Your text should:

    1. Begin with a clear introduction defining the ingredient, its origin, main composition and historical and contemporary usage context

    2. Develop scientifically established properties by systematically specifying:
    - The nature of studies (in vitro, animal studies, human clinical trials)
    - Methodological quality of research (sample size, duration, etc.)
    - Identified mechanisms of action when mentioned

    3. Present in a balanced way:
    - Demonstrated benefits with their level of scientific evidence
    - Limitations, adverse effects or precautions for use
    - Populations that may particularly benefit or should avoid this ingredient

    4. Contextualize results in the global scientific landscape (consensus, controversies or ongoing research)

    5. Conclude with an objective synthesis of the current state of knowledge
    6. IMPORTANT: How humans can use the ingredient (topical, oral, etc.)

    The style should be:
    - Formal and encyclopedic, without commercial or subjective formulations
    - Precise, with phrases like "studies suggest that..." or "research indicates..." followed by specific data
    - Structured with logical transitions between different aspects covered
    - Accessible, systematically explaining technical terms

    IMPORTANT: Base yourself ONLY on information present in the provided extracts. Do not invent data or conclusions not mentioned in the sources. If information is limited or preliminary, faithfully reflect this reality.
    IMPORTANT: Only provide information about the ingredient {ingredient} and do not make comparisons with other ingredients or products.

    IMPORTANT: The paragraph must be in English and be 300 words long.

    Extracts about {ingredient}:
    """
            else:
                prompt = f"""Tu es un expert en vulgarisation scientifique spécialisé dans les ingrédients naturels.

    Je vais te fournir des extraits d'articles scientifiques sur l'ingrédient "{ingredient}". Rédige un paragraphe détaillé, en français, dans un style encyclopédique similaire à Wikipedia qui synthétise les connaissances scientifiques actuelles sur cet ingrédient.

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

    IMPORTANT : Le paragraphe doit être en français et compter 300 mots.

    Extraits sur {ingredient} :
    """

            prompt += "\n---\n".join([doc.page_content for doc in batch])
            
            batch_summary = self.llm.invoke(prompt).content
            batch_summaries.append(batch_summary)
        
        #Final synthesis prompt also in correct language
        summaries_joined = "\n\n".join([f"Résumé {i+1}:\n{summary}" for i, summary in enumerate(batch_summaries)])
        
        if is_english:
            final_prompt = f"""You are an expert in scientific popularization specialized in natural ingredients.

    Here are several partial summaries about the ingredient "{ingredient}". Synthesize them into a detailed paragraph (300 words) in English, in an encyclopedic style similar to Wikipedia.

    MAIN OBJECTIVE: Present a comprehensive, balanced and factual overview that highlights significant scientific discoveries while remaining accessible to non-specialists.

    Your text should:

    1. Begin with a clear introduction defining the ingredient, its origin, main composition and historical and contemporary usage context

    2. Develop scientifically established properties by systematically specifying:
    - The nature of studies (in vitro, animal studies, human clinical trials)
    - Methodological quality of research (sample size, duration, etc.)
    - Identified mechanisms of action when mentioned

    3. Present in a balanced way:
    - Demonstrated benefits with their level of scientific evidence
    - Limitations, adverse effects or precautions for use
    - Populations that may particularly benefit or should avoid this ingredient

    4. Contextualize results in the global scientific landscape (consensus, controversies or ongoing research)

    5. Conclude with an objective synthesis of the current state of knowledge
    6. IMPORTANT: How humans can use the ingredient (topical, oral, etc.)

    The style should be:
    - Formal and encyclopedic, without commercial or subjective formulations
    - Precise, with phrases like "studies suggest that..." or "research indicates..." followed by specific data
    - Structured with logical transitions between different aspects covered
    - Accessible, systematically explaining technical terms

    IMPORTANT: Base yourself ONLY on information present in the provided summaries. Do not invent data or conclusions not mentioned in the sources.
    IMPORTANT: Only provide information about the ingredient {ingredient} and do not make comparisons with other ingredients or products.

    IMPORTANT: The paragraph must be in English and be 300 words long.

    Summaries to synthesize:
    {summaries_joined}
    """
        else:
            final_prompt = f"""Tu es un expert en vulgarisation scientifique spécialisé dans les ingrédients naturels.

    Voici plusieurs résumés partiels sur l'ingrédient "{ingredient}". Synthétise-les en un paragraphe détaillé (300 mots), en français, dans un style encyclopédique similaire à Wikipedia.

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

    IMPORTANT : Le paragraphe doit être en français et compter 300 mots.

    Résumés à synthétiser :
    {summaries_joined}
    """
        
        return self.llm.invoke(final_prompt).content

    
    def save_summary(self, ingredient: str, summary: str, summary_id: str = None) -> Dict[str, Any]:
        """Save the generated summary to a file."""
        if summary_id is None:
            summary_id = str(uuid.uuid4())
        
        try:
            # Create the summaries directory if it does not exist
            import os
            summary_dir = os.path.join(self.config.get("base_dir", ""), "backend", "data", "summaries")
            os.makedirs(summary_dir, exist_ok=True)
            
            # Create the filename
            filename = f"{ingredient}_summary_{summary_id}.md"
            filepath = os.path.join(summary_dir, filename)
            
            # Write the summary to the file
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(summary)
            
            logger.info(f"Summary saved at {filepath}")
            
            return {
                "status": "success",
                "message": "Summary saved successfully.",
                "ingredient": ingredient,
                "summary_id": summary_id,
                "filepath": filepath
            }
            
        except Exception as e:
            logger.error(f"Error while saving summary for {ingredient}: {e}")
            return {
                "status": "error",
                "message": f"Error while saving summary: {str(e)}",
                "ingredient": ingredient
            }


def generate_ingredient_summary(ingredient: str, save_to_file: bool = True, max_chunks: int = 50, 
                              llm_type: str = None, model_name: str = None,
                              max_tokens_ollama: int = 2000, max_chunks_ollama: int = 10,
                              language: str = "fr") -> Dict[str, Any]:
    """
    Utility function to generate a summary for an ingredient.
    This function can be called directly from other modules.

    Args:
        ingredient (str): Name of the ingredient to summarize
        save_to_file (bool): Whether to save the summary to a file
        max_chunks (int): Maximum number of chunks to process for OpenAI
        llm_type (str): Type of LLM to use ('openai' or 'ollama')
        model_name (str): Name of the model to use (depends on LLM type)
        max_tokens_ollama (int): Token limit for Ollama
        max_chunks_ollama (int): Maximum number of chunks to process for Ollama
        language (str): Language of the summary ("fr" for French, "en" for English)

    Returns:
        Dict[str, Any]: Result of the summary generation process
    """
    summarizer = IngredientSummarizer(
        llm_type=llm_type,
        model_name=model_name,
        max_tokens_ollama=max_tokens_ollama,
        max_chunks_ollama=max_chunks_ollama
    )
    
    # If using Ollama, use the specific Ollama chunk limit
    actual_max_chunks = max_chunks
    if llm_type == LLM_TYPE_OLLAMA or (llm_type is None and summarizer.llm_type == LLM_TYPE_OLLAMA):
        actual_max_chunks = max_chunks_ollama
    
    # Generate the summary
    summary_result = summarizer.summarize_ingredient(
        ingredient=ingredient, 
        max_chunks=actual_max_chunks,
        language=language
    )
    
    # Save the summary to a file if success
    if summary_result.get("status") == "success" and save_to_file:
        lang_suffix = f"_{language}" if language != "fr" else ""
        
        save_result = summarizer.save_summary(
            ingredient=ingredient,
            summary=summary_result.get("summary", ""),
            summary_id=summary_result.get("summary_id") + lang_suffix
        )
        
        summary_result["save_result"] = save_result
    
    return summary_result