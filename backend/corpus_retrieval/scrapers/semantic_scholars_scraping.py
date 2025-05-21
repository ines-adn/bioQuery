import requests
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
import time
import json
import os
import glob

class LlamaLLM(ChatOllama):
    def __init__(self, model_name: str, temperature: float = 0.5):
        super().__init__(
            model=model_name,
            temperature=temperature,
            base_url="http://localhost:11434",
        )

class SemanticScolarSearch:
    def __init__(self, ingredient, use_openai=None):
        self.ingredient = ingredient
        
        # Déterminer automatiquement si on utilise OpenAI
        if use_openai is None:
            # Vérifier si la variable d'environnement OPENAI_API_KEY est définie
            self.use_openai = os.environ.get("OPENAI_API_KEY") is not None
        else:
            self.use_openai = use_openai
            
        # Initialiser le modèle approprié
        if self.use_openai:
            print(f"Utilisation d'OpenAI pour générer la requête de recherche pour {ingredient}")
            self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
        else:
            print(f"Utilisation de Llama local pour générer la requête de recherche pour {ingredient}")
            self.llm = LlamaLLM("llama3.1")

    def generate_query(self) -> str:
        """ Utilise le LLM pour générer une requête optimisée pour Google Scholar. """
        prompt = f"""Tu es un expert en sciences. Formule une requête Google Scholar 
        pour trouver des articles académiques sur les effets de {self.ingredient} sur le corps humain (peau, santé, ...).
        
        Exemple de sortie attendue : 
        "Effects of Aloe Vera on skin hydration"
        
        Génère uniquement la requête, sans explications.
        Ta requête doit contenir au maximum quatre mots-clés.
        La requête doit être en anglais.
        La requête ne doit pas contenir les mots "and" ou "or".
        La requête doit être une simple déclaration, sans condition.
        """
        
        output = self.llm.invoke(prompt)
        # Extraire le contenu selon le type de LLM
        if self.use_openai:
            query = output.content.strip()
        else:
            query = output.content.strip()
        
        # Nettoyage de la requête
        query = query.replace('"', '').replace("'", "").strip()
        query = ' '.join(query.split())
        return query

# Charger la configuration depuis le fichier config.json
def load_config(config_file="config.json"):
    try:
        with open(config_file, "r") as file:
            config = json.load(file)
            return config
    except FileNotFoundError:
        print(f"Le fichier {config_file} n'a pas été trouvé.")
        return None
    except json.JSONDecodeError:
        print("Erreur lors de la lecture du fichier config.json.")
        return None

# Télécharger le PDF
def download_pdf(pdf_url, save_path="article.pdf"):
    """ Télécharge un PDF à partir d'une URL. """
    if not pdf_url:
        print("Aucun PDF disponible pour cet article.")
        return
    
    response = requests.get(pdf_url, stream=True)
    
    if response.status_code == 200:
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print(f"PDF téléchargé : {save_path}")
    else:
        print("Impossible de télécharger le PDF.")

# Recherche des articles sur Semantic Scholar
def search_semantic_scholar(query, num_results=3):
    """ Recherche des articles sur Semantic Scholar et retourne leurs informations. """
    # URL de l'API
    API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": num_results,  # Limiter à 3 résultats
        "fields": "title,url,openAccessPdf"
    }
    
    retries = 3  # Nombre d'essais en cas d'erreur 429
    for _ in range(retries):
        response = requests.get(API_URL, params=params)
        
        if response.status_code == 200:
            data = response.json().get("data", [])
            if data:
                return data
        elif response.status_code == 429:
            print("Erreur 429 : Trop de requêtes. Attente de 60 secondes.")
            time.sleep(60)  # Attente de 60 secondes avant de réessayer
        else:
            print(f"Erreur API: {response.status_code}")
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
                print(f"Utilisation des articles en cache pour {ingredient} ({len(metadata)} articles)")
                return metadata
        except (json.JSONDecodeError, IOError):
            # Si le fichier de métadonnées est corrompu, on continue
            pass
    
    # Si pas de métadonnées, créer une liste basique à partir des noms de fichiers
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

def save_metadata(metadata, folder_path, ingredient):
    """
    Sauvegarde les métadonnées des articles téléchargés.
    
    Args:
        metadata: Liste des métadonnées des articles
        folder_path: Chemin vers le dossier où sauvegarder les métadonnées
        ingredient: Nom de l'ingrédient
    """
    ingredient_underscore = ingredient.replace(" ", "_").lower()
    metadata_path = os.path.join(folder_path, f"{ingredient_underscore}_metadata.json")
    
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"Métadonnées sauvegardées dans {metadata_path}")
    except IOError as e:
        print(f"Erreur lors de la sauvegarde des métadonnées: {e}")

def search_and_download_from_semantic_scholars(ingredient, use_cache=True, use_openai_for_query=None):
    """
    Lance la recherche et télécharge les articles, avec support du cache local.
    
    Args:
        ingredient: Nom de l'ingrédient à rechercher
        use_cache: Utiliser le cache local si disponible
        use_openai_for_query: Utiliser OpenAI pour générer la requête (si None, décide automatiquement)
        
    Returns:
        list: Liste des métadonnées des articles
    """
    config = load_config()
    if not config:
        return {"error": "Configuration invalide"}

    base_dir = config.get("base_dir", "")
    if not base_dir:
        return {"error": "Chemin base_dir non défini"}

    folder_name = ingredient.replace(" ", "_").lower()
    folder_path = os.path.join(base_dir, "backend", "data", "articles", folder_name)
    
    # Vérifier d'abord si des articles sont déjà en cache
    if use_cache:
        cached_articles = check_cached_articles(folder_path, ingredient)
        if cached_articles and len(cached_articles) > 0:
            # Marquer ces résultats comme provenant du cache
            for article in cached_articles:
                article["from_cache"] = True
            return cached_articles
    
    # Si pas de cache ou cache désactivé, procéder au scraping
    os.makedirs(folder_path, exist_ok=True)
    
    # Vérifier la variable d'environnement OpenAI pour décider quel LLM utiliser
    has_openai_key = os.environ.get("OPENAI_API_KEY") is not None
    
    # Si use_openai_for_query est explicitement défini, l'utiliser
    # Sinon, utiliser OpenAI si la clé API est disponible
    effective_use_openai = use_openai_for_query if use_openai_for_query is not None else has_openai_key
    
    # Log pour debug
    print(f"[DEBUG] Variable d'environnement OPENAI_API_KEY présente: {has_openai_key}")
    print(f"[DEBUG] Utilisation d'OpenAI pour la requête: {effective_use_openai}")
    
    search_tool = SemanticScolarSearch(ingredient, use_openai=effective_use_openai)
    query = search_tool.generate_query()
    print(f"Requête générée: '{query}'")
    
    articles = search_semantic_scholar(query, num_results=3)
    if not articles:
        return {"error": "Aucun article trouvé pour la requête donnée."}
    
    results = []
    for i, article in enumerate(articles, 1):
        result = {
            "number": i,
            "title": article["title"], 
            "url": article["url"],
            "from_cache": False
        }
        
        pdf_url = article.get("openAccessPdf")
        if pdf_url and isinstance(pdf_url, dict):
            pdf_url = pdf_url.get("url", "")
        else:
            pdf_url = ""

        if pdf_url:
            ingredient_underscore = ingredient.replace(" ", "_").lower()
            save_path = os.path.join(folder_path, f"{ingredient_underscore}_article_{i}.pdf")
            download_pdf(pdf_url, save_path)
            result["pdf"] = save_path
        else:
            result["pdf"] = None

        results.append(result)
    
    # Sauvegarder les métadonnées pour les utiliser comme cache la prochaine fois
    save_metadata(results, folder_path, ingredient)
    
    return results

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