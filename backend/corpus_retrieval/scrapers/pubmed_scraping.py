import requests
import json
import os
from langchain_ollama import ChatOllama

class LlamaLLM(ChatOllama):
    def __init__(self, model_name: str, temperature: float = 0.5):
        super().__init__(
            model=model_name,
            temperature=temperature,
            base_url="http://localhost:11434",
        )

class PubMedSearch:
    def __init__(self, ingredient, allegation):
        self.ingredient = ingredient
        self.allegation = allegation
        self.llm = LlamaLLM("llama3.1")

    def generate_query(self) -> str:
        """ Utilise Llama pour générer une requête optimisée pour PubMed. """
        prompt = f"""Tu es un expert en sciences. Formule une requête PubMed
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
        query = output.content.strip().replace('"', '').replace("'", "").strip()
        return ' '.join(query.split())

# Charger la configuration depuis config.json
def load_config(config_file="config.json"):
    try:
        with open(config_file, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Le fichier {config_file} n'a pas été trouvé.")
    except json.JSONDecodeError:
        print("Erreur de lecture du fichier config.json.")
    return None

# Recherche sur PubMed
def search_pubmed(query, num_results=3):
    """ Recherche des articles sur PubMed via l'API Entrez """
    API_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": num_results,
        "retmode": "json"
    }
    response = requests.get(API_URL, params=params)
    if response.status_code == 200:
        ids = response.json().get("esearchresult", {}).get("idlist", [])
        return ids
    return []

# Récupérer les détails des articles
def fetch_pubmed_details(pubmed_ids):
    """ Récupère les titres et résumés des articles via PubMed """
    if not pubmed_ids:
        return []
    API_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pubmed_ids),
        "retmode": "json"
    }
    response = requests.get(API_URL, params=params)
    if response.status_code == 200:
        return response.json().get("result", {})
    return {}

# Télécharger le PDF depuis PMC
def download_pmc_pdf(pubmed_id, save_path):
    pmc_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pubmed_id}/pdf/"
    pdf_response = requests.get(pmc_url, stream=True)
    
    if pdf_response.status_code == 200:
        pdf_path = os.path.join(save_path, f"{pubmed_id}.pdf")
        with open(pdf_path, "wb") as pdf_file:
            for chunk in pdf_response.iter_content(chunk_size=1024):
                pdf_file.write(chunk)
        print(f"✅ PDF téléchargé : {pdf_path}")
    else:
        print(f"❌ PDF non disponible pour PMC{pubmed_id}")



def search_and_download_from_pubmed(ingredient,allegation):

    # Charger la configuration
    config = load_config()
    if not config:
        return {"error": "Configuration invalide"}
    
    search_tool = PubMedSearch(ingredient, allegation)
    query = search_tool.generate_query()
    base_dir = config.get("base_dir", "")
    if not base_dir:
        return {"error": "Chemin base_dir non défini"}
    
    folder_name = query.replace(" ", "_").lower()
    folder_path = os.path.join(base_dir, "backend", "corpus-retrieval", "data", "articles", folder_name)
    os.makedirs(folder_path, exist_ok=True)

    article_ids = search_pubmed(query, num_results=3)
    articles = fetch_pubmed_details(article_ids)

    if articles:
        for i, pubmed_id in enumerate(article_ids, 1):
            article = articles.get(pubmed_id, {})
            title = article.get("title", "Titre inconnu")
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/"
            print(f"\n📄 Article {i}")
            print(f"📌 Titre : {title}")
            print(f"🔗 URL : {url}")
            
            # Télécharger le PDF si disponible sur PMC
            download_pmc_pdf(pubmed_id, folder_path)
    else:
        print("Aucun article trouvé.")
    results_list = []
    if articles:
        for i, pubmed_id in enumerate(article_ids, 1):
            article = articles.get(pubmed_id, {})
            title = article.get("title", "Titre inconnu")
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/"
            results_list.append({"number": i, "title": title, "url": url})
    return {"results": results_list}

if __name__ == "__main__":
    ingredient = input("Entrez l'ingrédient : ")
    allegation = input("Entrez l'allégation : ")
    results = search_and_download_from_pubmed("Aloe Vera", "hydrating")
    print(results)