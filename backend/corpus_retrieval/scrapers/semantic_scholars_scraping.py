import requests
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
import time
import json
import os
import glob
from utils import load_config


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
        
        if not isinstance(config, dict):
            config = {}

        # ------------------ INITIALIZE LLM ------------------ #
        
        # Detects whether to use OpenAI or Llama based on the config or environment variable

        if use_openai is None:

            # First check if the config has a llm_type
            # If not, default to using OpenAI if the environment variable is set

            config_llm_type = config.get("llm_type", "openai")
            self.use_openai = (config_llm_type == "openai" and 
                            os.environ.get("OPENAI_API_KEY") is not None)
        else:
            self.use_openai = use_openai
            
        # Initialize the LLM based on the use_openai flag
        if self.use_openai:
            print(f"Using OpenAI to generate the search query for {ingredient}")
            try:
                self.llm = ChatOpenAI(model="gpt-4.1", temperature=0.5)
            except Exception as e:
                print(f"Error initializing OpenAI: {e}")
                # Fallback to Llama
                self.use_openai = False
                llama_model_name = config.get("ollama_model", "llama3.2:3b")
                self.llm = LlamaLLM(llama_model_name)
        else:
            print(f"Using local Llama to generate the search query for {ingredient}")
            # Load config from config.json if possible
            llama_model_name = config.get("ollama_model", "llama3.2:3b")
            try:
                self.llm = LlamaLLM(llama_model_name)
            except Exception as e:
                print(f"Error initializing Llama: {e}")
                raise Exception(f"Unable to initialize the LLM model: {e}")

    def generate_query(self) -> str:
        """Uses the LLM to generate an optimized query for Semantic Scholar."""

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
            query = output.content.strip() if hasattr(output, 'content') else str(output).strip()
            
            # Clean the query
            query = query.replace('"', '').replace("'", "").strip()
            query = ' '.join(query.split())
            
            if not query:
                # Fallback query if LLM fails
                query = f"{self.ingredient} health effects"
                
            return query
        except Exception as e:
            print(f"Error generating the query: {e}")
            # Fallback query
            return f"{self.ingredient} health effects"
        

def download_pdf(pdf_url, save_path="article.pdf"):
    """Download a PDF from a given URL."""
    if not pdf_url:
        print("No PDF available for this article.")
        return False
    
    try:
        response = requests.get(pdf_url, stream=True, timeout=30)
        
        if response.status_code == 200:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
            print(f"Downloaded PDF: {save_path}")
            return True
        else:
            print(f"Unable to download the PDF. Status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading the PDF: {e}")
        return False


def search_semantic_scholar(query, num_results=3):
    """Search for articles on Semantic Scholar and return their information."""
    
    if not query or not query.strip():
        print("Empty or invalid query")
        return []
        
    API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query.strip(),
        "limit": num_results,
        "fields": "title,url,openAccessPdf"
    }
    
    retries = 3
    for attempt in range(retries):
        try:
            
            response = requests.get(API_URL, params=params, timeout=30)
            
            if response.status_code == 429:
                print("Rate limit headers:")
                for header, value in response.headers.items():
                    if 'rate' in header.lower() or 'retry' in header.lower():
                        print(f"  {header}: {value}")
            
            if response.status_code == 200:
                data = response.json().get("data", [])
                if data:
                    print(f"Found {len(data)} articles for query: '{query}'")
                    return data
                else:
                    print(f"No articles found for query: '{query}'")
                    return []
            elif response.status_code == 429:
                print(f"Error 429: Too many requests. Attempt {attempt + 1}/{retries}. Waiting 60 seconds.")
                time.sleep(60)
            else:
                print(f"API error: {response.status_code} - {response.text}")
                break
        except requests.exceptions.Timeout:
            print(f"Timeout during API request. Attempt {attempt + 1}/{retries}")
        except requests.exceptions.RequestException as e:
            print(f"Connection error: {e}")
            break
        except Exception as e:
            print(f"Unexpected error during search: {e}")
            break
    
    return []

def check_cached_articles(folder_path, ingredient):
    """
    Checks if articles have already been downloaded locally for this ingredient.

    Args:
        folder_path: Path to the folder where articles are stored
        ingredient: Name of the ingredient

    Returns:
        list: List of metadata for cached articles, or None if no cache is found
    """
    try:

        if not os.path.exists(folder_path):
            return None
        
        ingredient_underscore = ingredient.replace(" ", "_").lower()
        pattern = os.path.join(folder_path, f"{ingredient_underscore}_article_*.pdf")
        pdf_files = glob.glob(pattern)
        
        if not pdf_files:
            return None
        
        metadata_path = os.path.join(folder_path, f"{ingredient_underscore}_metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    if isinstance(metadata, list) and len(metadata) > 0:
                        print(f"Using cached articles for {ingredient} ({len(metadata)} articles)")
                        return metadata
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error reading cached metadata: {e}")
        
        cached_articles = []
        for i, pdf_path in enumerate(sorted(pdf_files), 1):
            
            cached_articles.append({
                "number": i,
                "title": f"Cached article {i} for {ingredient}",
                "url": None,
                "pdf": pdf_path
            })
        
        print(f"Using cached articles for {ingredient} ({len(cached_articles)} articles)")
        return cached_articles
    except Exception as e:
        print(f"Error checking cache: {e}")
        return None

def save_metadata(metadata, folder_path, ingredient):
    """
    Saves the metadata of the downloaded articles.

    Args:
        metadata: List of article metadata
        folder_path: Path to the folder where metadata should be saved
        ingredient: Name of the ingredient
    """
    try:
        ingredient_underscore = ingredient.replace(" ", "_").lower()
        metadata_path = os.path.join(folder_path, f"{ingredient_underscore}_metadata.json")
        
        os.makedirs(folder_path, exist_ok=True)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"Metadata saved in {metadata_path}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde des métadonnées: {e}")

def search_and_download_from_semantic_scholars(ingredient, use_cache=False, use_openai_for_query=None):
    """
    Launches the search and downloads articles, with support for local cache.

    Args:
        ingredient: Name of the ingredient to search for
        use_cache: Use local cache if available
        use_openai_for_query: Use OpenAI to generate the query (if None, decides automatically)

    Returns:
        list: List of article metadata or dict with error
    """
    try:
        if not ingredient or not isinstance(ingredient, str) or not ingredient.strip():
            return {"error": "Invalid or empty ingredient name"}
        
        ingredient = ingredient.strip()

        config = load_config()
        if not isinstance(config, dict):
            return {"error": "Invalid configuration"}

        base_dir = config.get("base_dir", ".")
        if not base_dir:
            base_dir = "."

        folder_name = ingredient.replace(" ", "_").lower()
        folder_path = os.path.join(base_dir, "backend", "data", "articles", folder_name)
        
        if use_cache:
            cached_articles = check_cached_articles(folder_path, ingredient)
            if cached_articles and len(cached_articles) > 0:
                for article in cached_articles:
                    if isinstance(article, dict):
                        article["from_cache"] = True
                return cached_articles
        
        try:
            os.makedirs(folder_path, exist_ok=True)
        except Exception as e:
            print(f"Erreur lors de la création du dossier: {e}")
            return {"error": f"Impossible de créer le dossier de destination: {e}"}
        
        # Check the OpenAI environment variable to decide which LLM to use
        config_llm_type = config.get("llm_type", "openai")
        has_openai_key = os.environ.get("OPENAI_API_KEY") is not None

        if use_openai_for_query is not None:
            effective_use_openai = use_openai_for_query
        else:
            effective_use_openai = (config_llm_type == "openai" and has_openai_key)
        
        print(f"[DEBUG] OPENAI_API_KEY environment variable present: {has_openai_key}")
        print(f"[DEBUG] Using OpenAI for query: {effective_use_openai}")

        # ------------- INITALIZE THE SEMANTIC SCOLAR SEARCH TOOL ------------- #

        try:
            search_tool = SemanticScolarSearch(ingredient, use_openai=effective_use_openai, config=config)
            query = search_tool.generate_query()
        except Exception as e:
            print(f"Error initializing the search tool: {e}")
            return {"error": f"Unable to initialize the search tool: {e}"}
        
        if not query:
            return {"error": "Unable to generate a search query"}
            
        print(f"Generated query: '{query}'")
        
        # Search for articles
        articles = search_semantic_scholar(query, num_results=3)
        if not articles:
            return {"error": "No articles found for the given query."}
        
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
            return {"error": "No valid articles found"}
        
        # Save metadata so it can be used as cache next time
        save_metadata(results, folder_path, ingredient)
        
        return results
        
    except Exception as e:
        error_msg = f"Unexpected error in search_and_download_from_semantic_scholars: {str(e)}"
        print(error_msg)
        return {"error": error_msg}