import requests
from langchain_ollama import ChatOllama
import time
import json
import os

class LlamaLLM(ChatOllama):
    def __init__(self, model_name: str, temperature: float = 0.5):
        super().__init__(
            model=model_name,
            temperature=temperature,
            base_url="http://localhost:11434",
        )

class SemanticScolarSearch:
    def __init__(self, ingredient):
        self.ingredient = ingredient
        self.llm = LlamaLLM("llama3.1")

    def generate_query(self) -> str:
        """ Utilise Llama pour générer une requête optimisée pour Google Scholar. """
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



def search_and_download_from_semantic_scholars(ingredient):
    
    """ Lance la recherche et télécharge les articles. """
    config = load_config()
    if not config:
        return {"error": "Configuration invalide"}

    search_tool = SemanticScolarSearch(ingredient)
    query = search_tool.generate_query()
    base_dir = config.get("base_dir", "")

    if not base_dir:
        return {"error": "Chemin base_dir non défini"}

    folder_name = ingredient.replace(" ", "_").lower()
    folder_path = os.path.join(base_dir, "backend", "data", "articles", folder_name)
    os.makedirs(folder_path, exist_ok=True)

    articles = search_semantic_scholar(query, num_results=3)
    results = []

    for i, article in enumerate(articles, 1):
        result = {"title": article["title"], "url": article["url"]}
        
        pdf_url = article.get("openAccessPdf")
        if pdf_url and isinstance(pdf_url, dict):
            pdf_url = pdf_url.get("url", "")
        else:
            pdf_url = ""

        if pdf_url:
            ingredient_underscore = ingredient.replace(" ", "_")
            save_path = os.path.join(folder_path, f"{ingredient_underscore}_article_{i}.pdf")
            download_pdf(pdf_url, save_path)
            result["pdf"] = save_path
        else:
            result["pdf"] = None

        results.append(result)

    if not articles:
        return {"error": "Aucun article trouvé pour la requête donnée."}
    
    return [{"number": i, "title": article["title"], "url": article["url"]} for i, article in enumerate(articles, 1)]

if __name__ == "__main__":
    ingredient = input("Ingrédient : ")
    results = search_and_download_from_semantic_scholars(ingredient)
    print(results)