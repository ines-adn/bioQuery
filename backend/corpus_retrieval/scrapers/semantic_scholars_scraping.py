import requests
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
import time
import json
import os
import glob
from datetime import datetime

last_request_time = None

class LlamaLLM(ChatOllama):
    def __init__(self, model_name: str, temperature: float = 0.5):
        super().__init__(
            model=model_name,
            temperature=temperature,
            base_url="http://localhost:11434",
            num_predict=300,
            top_k=10,
            top_p=0.9,
        )

class SemanticScolarSearch:
    def __init__(self, ingredient, use_openai=None, config=None):
        self.ingredient = ingredient
        
        # Load config if not provided
        if config is None:
            config = load_config()
        
        # Ensure config is a dictionary
        if not isinstance(config, dict):
            config = {}
        
        # Déterminer automatiquement si on utilise OpenAI
        if use_openai is None:
            # D'abord vérifier la config, puis la variable d'environnement
            config_llm_type = config.get("llm_type", "openai")
            self.use_openai = (config_llm_type == "openai" and 
                            os.environ.get("OPENAI_API_KEY") is not None)
        else:
            self.use_openai = use_openai
            
        # Initialiser le modèle approprié
        if self.use_openai:
            print(f"Utilisation d'OpenAI pour générer la requête de recherche pour {ingredient}")
            try:
                self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
            except Exception as e:
                print(f"Erreur lors de l'initialisation d'OpenAI: {e}")
                # Fallback to Llama
                self.use_openai = False
                llama_model_name = config.get("ollama_model", "llama3.1")
                self.llm = LlamaLLM(llama_model_name)
        else:
            print(f"Utilisation de Llama local pour générer la requête de recherche pour {ingredient}")
            # Charger la config depuis le fichier config.json si possible
            llama_model_name = config.get("ollama_model", "llama3.1")
            try:
                self.llm = LlamaLLM(llama_model_name)
            except Exception as e:
                print(f"Erreur lors de l'initialisation de Llama: {e}")
                raise Exception(f"Impossible d'initialiser le modèle LLM: {e}")

    def generate_query(self) -> str:
        """ Utilise le LLM pour générer une requête optimisée pour Google Scholar. """
        prompt = f"""Tu es un expert en sciences. Formule une requête Google Scholar 
        pour trouver des articles académiques sur les effets de {self.ingredient} sur le corps humain (peau, santé, ...).
        
        Exemple de sortie attendue : 
        "Aloe Vera skin hydration"
        
        Génère uniquement la requête, sans explications.
        Ta requête doit contenir au maximum quatre mots-clés.
        La requête doit être en anglais.
        La requête ne doit pas contenir les mots "and" ou "or".
        La requête doit être une simple déclaration, sans condition.
        """
        
        try:
            output = self.llm.invoke(prompt)
            # Extraire le contenu selon le type de LLM
            query = output.content.strip() if hasattr(output, 'content') else str(output).strip()
            
            # Nettoyage de la requête
            query = query.replace('"', '').replace("'", "").strip()
            query = ' '.join(query.split())
            
            if not query:
                # Fallback query if LLM fails
                query = f"{self.ingredient} health effects"
                
            return query
        except Exception as e:
            print(f"Erreur lors de la génération de la requête: {e}")
            # Fallback query
            return f"{self.ingredient} health effects"

# Charger la configuration depuis le fichier config.json
def load_config(config_file="config.json"):
    """Load configuration from JSON file with error handling."""
    try:
        if not os.path.exists(config_file):
            print(f"Le fichier {config_file} n'existe pas. Utilisation de la configuration par défaut.")
            return {
                "base_dir": ".",
                "llm_type": "openai",
                "ollama_model": "llama3.1"
            }
            
        with open(config_file, "r", encoding='utf-8') as file:
            config = json.load(file)
            # Validate config structure
            if not isinstance(config, dict):
                print("Configuration invalide. Utilisation de la configuration par défaut.")
                return {
                    "base_dir": ".",
                    "llm_type": "openai", 
                    "ollama_model": "llama3.1"
                }
            return config
    except json.JSONDecodeError as e:
        print(f"Erreur lors de la lecture du fichier config.json: {e}")
        return {
            "base_dir": ".",
            "llm_type": "openai",
            "ollama_model": "llama3.1"
        }
    except Exception as e:
        print(f"Erreur inattendue lors du chargement de la configuration: {e}")
        return {
            "base_dir": ".",
            "llm_type": "openai",
            "ollama_model": "llama3.1"
        }

# Télécharger le PDF
def download_pdf(pdf_url, save_path="article.pdf"):
    """ Télécharge un PDF à partir d'une URL. """
    if not pdf_url:
        print("Aucun PDF disponible pour cet article.")
        return False
    
    try:
        response = requests.get(pdf_url, stream=True, timeout=30)
        
        if response.status_code == 200:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
            print(f"PDF téléchargé : {save_path}")
            return True
        else:
            print(f"Impossible de télécharger le PDF. Status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Erreur lors du téléchargement du PDF: {e}")
        return False

# Recherche des articles sur Semantic Scholar
def search_semantic_scholar(query, num_results=3):
    """ Recherche des articles sur Semantic Scholar et retourne leurs informations. """
    global last_request_time
    
    if not query or not query.strip():
        print("Requête vide ou invalide")
        return []
        
    # URL de l'API
    API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query.strip(),
        "limit": num_results,
        "fields": "title,url,openAccessPdf"
    }
    
    retries = 3
    for attempt in range(retries):
        try:
            print(f"Making request at {datetime.now()}")
            print(f"Last request was at: {last_request_time if last_request_time else 'Never'}")
            
            response = requests.get(API_URL, params=params, timeout=30)

            last_request_time = datetime.now()
            
            if response.status_code == 429:
                print("Rate limit headers:")
                for header, value in response.headers.items():
                    if 'rate' in header.lower() or 'retry' in header.lower():
                        print(f"  {header}: {value}")
            
            if response.status_code == 200:
                data = response.json().get("data", [])
                if data:
                    print(f"Trouvé {len(data)} articles pour la requête: '{query}'")
                    return data
                else:
                    print(f"Aucun article trouvé pour la requête: '{query}'")
                    return []
            elif response.status_code == 429:
                print(f"Erreur 429 : Trop de requêtes. Tentative {attempt + 1}/{retries}. Attente de 60 secondes.")
                time.sleep(60)
            else:
                print(f"Erreur API: {response.status_code} - {response.text}")
                break
        except requests.exceptions.Timeout:
            print(f"Timeout lors de la requête API. Tentative {attempt + 1}/{retries}")
        except requests.exceptions.RequestException as e:
            print(f"Erreur de connexion: {e}")
            break
        except Exception as e:
            print(f"Erreur inattendue lors de la recherche: {e}")
            break
    
    return []

def check_cached_articles(folder_path, ingredient):
    """
    Vérifie si des articles sont déjà téléchargés localement pour cet ingrédient.
    
    Args:
        folder_path: Chemin vers le dossier où les articles sont stockés
        ingredient: Nom de l'ingrédient
        
    Returns:
        list: Liste des métadonnées des articles en cache ou None si aucun cache
    """
    try:
        # Vérifier si le dossier existe
        if not os.path.exists(folder_path):
            return None
        
        # Rechercher les fichiers PDF correspondant à l'ingrédient
        ingredient_underscore = ingredient.replace(" ", "_").lower()
        pattern = os.path.join(folder_path, f"{ingredient_underscore}_article_*.pdf")
        pdf_files = glob.glob(pattern)
        
        if not pdf_files:
            return None
        
        # Vérifier si un fichier de métadonnées existe
        metadata_path = os.path.join(folder_path, f"{ingredient_underscore}_metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    if isinstance(metadata, list) and len(metadata) > 0:
                        print(f"Utilisation des articles en cache pour {ingredient} ({len(metadata)} articles)")
                        return metadata
            except (json.JSONDecodeError, IOError) as e:
                print(f"Erreur lors de la lecture des métadonnées en cache: {e}")
        
        # Si pas de métadonnées valides, créer une liste basique à partir des noms de fichiers
        cached_articles = []
        for i, pdf_path in enumerate(sorted(pdf_files), 1):
            # Extraire le nom de base sans extension
            filename = os.path.basename(pdf_path)
            title = filename.replace(f"{ingredient_underscore}_article_", "").replace(".pdf", "")
            
            cached_articles.append({
                "number": i,
                "title": f"Cached article {i} for {ingredient}",
                "url": None,
                "pdf": pdf_path
            })
        
        print(f"Utilisation des articles en cache pour {ingredient} ({len(cached_articles)} articles)")
        return cached_articles
    except Exception as e:
        print(f"Erreur lors de la vérification du cache: {e}")
        return None

def save_metadata(metadata, folder_path, ingredient):
    """
    Sauvegarde les métadonnées des articles téléchargés.
    
    Args:
        metadata: Liste des métadonnées des articles
        folder_path: Chemin vers le dossier où sauvegarder les métadonnées
        ingredient: Nom de l'ingrédient
    """
    try:
        ingredient_underscore = ingredient.replace(" ", "_").lower()
        metadata_path = os.path.join(folder_path, f"{ingredient_underscore}_metadata.json")
        
        os.makedirs(folder_path, exist_ok=True)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"Métadonnées sauvegardées dans {metadata_path}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde des métadonnées: {e}")

def search_and_download_from_semantic_scholars(ingredient, use_cache=True, use_openai_for_query=None):
    """
    Lance la recherche et télécharge les articles, avec support du cache local.
    
    Args:
        ingredient: Nom de l'ingrédient à rechercher
        use_cache: Utiliser le cache local si disponible
        use_openai_for_query: Utiliser OpenAI pour générer la requête (si None, décide automatiquement)
        
    Returns:
        list: Liste des métadonnées des articles ou dict avec erreur
    """
    try:
        # Validate input
        if not ingredient or not isinstance(ingredient, str) or not ingredient.strip():
            return {"error": "Nom d'ingrédient invalide ou vide"}
        
        ingredient = ingredient.strip()
        
        # Load config with error handling
        config = load_config()
        if not isinstance(config, dict):
            return {"error": "Configuration invalide"}

        base_dir = config.get("base_dir", ".")
        if not base_dir:
            base_dir = "."

        folder_name = ingredient.replace(" ", "_").lower()
        folder_path = os.path.join(base_dir, "backend", "data", "articles", folder_name)
        
        # Vérifier d'abord si des articles sont déjà en cache
        if use_cache:
            cached_articles = check_cached_articles(folder_path, ingredient)
            if cached_articles and len(cached_articles) > 0:
                # Marquer ces résultats comme provenant du cache
                for article in cached_articles:
                    if isinstance(article, dict):
                        article["from_cache"] = True
                return cached_articles
        
        # Si pas de cache ou cache désactivé, procéder au scraping
        try:
            os.makedirs(folder_path, exist_ok=True)
        except Exception as e:
            print(f"Erreur lors de la création du dossier: {e}")
            return {"error": f"Impossible de créer le dossier de destination: {e}"}
        
        # Vérifier la variable d'environnement OpenAI pour décider quel LLM utiliser
        config_llm_type = config.get("llm_type", "openai")
        has_openai_key = os.environ.get("OPENAI_API_KEY") is not None

        # Si use_openai_for_query est explicitement défini, l'utiliser
        # Sinon, utiliser la configuration et vérifier la clé API
        if use_openai_for_query is not None:
            effective_use_openai = use_openai_for_query
        else:
            effective_use_openai = (config_llm_type == "openai" and has_openai_key)
        
        # Log pour debug
        print(f"[DEBUG] Variable d'environnement OPENAI_API_KEY présente: {has_openai_key}")
        print(f"[DEBUG] Utilisation d'OpenAI pour la requête: {effective_use_openai}")
        
        # Initialize search tool with error handling
        try:
            search_tool = SemanticScolarSearch(ingredient, use_openai=effective_use_openai, config=config)
            query = search_tool.generate_query()
        except Exception as e:
            print(f"Erreur lors de l'initialisation de l'outil de recherche: {e}")
            return {"error": f"Impossible d'initialiser l'outil de recherche: {e}"}
        
        if not query:
            return {"error": "Impossible de générer une requête de recherche"}
            
        print(f"Requête générée: '{query}'")
        
        # Search for articles
        articles = search_semantic_scholar(query, num_results=3)
        if not articles:
            return {"error": "Aucun article trouvé pour la requête donnée."}
        
        # Process articles
        results = []
        for i, article in enumerate(articles, 1):
            if not isinstance(article, dict):
                continue
                
            result = {
                "number": i,
                "title": article.get("title", f"Article {i}"), 
                "url": article.get("url", ""),
                "from_cache": False
            }
            
            # Handle PDF download
            pdf_url = article.get("openAccessPdf")
            if pdf_url:
                if isinstance(pdf_url, dict):
                    pdf_url = pdf_url.get("url", "")
                elif not isinstance(pdf_url, str):
                    pdf_url = ""

                if pdf_url:
                    ingredient_underscore = ingredient.replace(" ", "_").lower()
                    save_path = os.path.join(folder_path, f"{ingredient_underscore}_article_{i}.pdf")
                    if download_pdf(pdf_url, save_path):
                        result["pdf"] = save_path
                    else:
                        result["pdf"] = None
                else:
                    result["pdf"] = None
            else:
                result["pdf"] = None

            results.append(result)
        
        if not results:
            return {"error": "Aucun article valide trouvé"}
        
        # Sauvegarder les métadonnées pour les utiliser comme cache la prochaine fois
        save_metadata(results, folder_path, ingredient)
        
        return results
        
    except Exception as e:
        error_msg = f"Erreur inattendue dans search_and_download_from_semantic_scholars: {str(e)}"
        print(error_msg)
        return {"error": error_msg}

if __name__ == "__main__":
    ingredient = input("Ingrédient : ")
    use_cache = input("Utiliser le cache si disponible ? (o/n) : ").lower() == 'o'
    
    # Option pour choisir explicitement OpenAI pour la requête
    use_openai = input("Forcer l'utilisation d'OpenAI pour la requête ? (o/n/auto) : ").lower()
    if use_openai == "o":
        use_openai_for_query = True
    elif use_openai == "n":
        use_openai_for_query = False
    else:
        use_openai_for_query = None  # Automatique selon la variable d'environnement
    
    results = search_and_download_from_semantic_scholars(
        ingredient, 
        use_cache=use_cache,
        use_openai_for_query=use_openai_for_query
    )
    print(results)