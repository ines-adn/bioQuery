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
    def __init__(self, ingredient, allegation):
        self.ingredient = ingredient
        self.allegation = allegation
        self.llm = LlamaLLM("llama3.1")

    def generate_query(self) -> str:
        """ Utilise Llama pour générer une requête optimisée pour Google Scholar. """
        prompt = f"""Tu es un expert en sciences. Formule une requête Google Scholar 
        pour trouver des articles académiques prouvant que {self.ingredient} est {self.allegation}.
        
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

# URL de l'API
API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

# Charger la configuration
config = load_config()

# Utilisation de la classe pour effectuer une recherche
ingredient = "Aloe Vera"
allegation = "hydratant"
search_tool = SemanticScolarSearch(ingredient, allegation)
query = search_tool.generate_query()

if config:
    base_dir = config.get("base_dir", "")  # Récupérer le chemin de base
    if base_dir:
        # Création du dossier avec le nom "nettoyé" pour la query
        folder_name = query.replace(" ", "_").lower()  # Remplacer les espaces par des underscores et convertir en minuscule
        folder_path = os.path.join(base_dir, "backend", "corpus-retrieval", "data", "articles", folder_name)

        # Créer le dossier si nécessaire
        os.makedirs(folder_path, exist_ok=True)
        print(f"Dossier créé ou déjà existant : {folder_path}")

        # Recherche d'articles
        articles = search_semantic_scholar(query, num_results=3)

        if articles:
            for i, article in enumerate(articles, 1):
                print(f"\n📄 Article {i}")
                print(f"📌 Titre : {article['title']}")
                print(f"🔗 URL : {article['url']}")
                
                # 📥 Téléchargement du PDF si disponible
                open_access_pdf = article.get("openAccessPdf", None)
                pdf_url = open_access_pdf.get("url", "") if open_access_pdf else ""
                
                if pdf_url:
                    save_path = os.path.join(folder_path, f"article_{i}.pdf")
                    download_pdf(pdf_url, save_path=save_path)
                else:
                    print("⚠️ Aucun PDF disponible.")
        else:
            print("Aucun article trouvé.")
    else:
        print("Le chemin de base n'est pas défini dans config.json.")
else:
    print("La configuration a échoué. Veuillez vérifier le fichier config.json.")