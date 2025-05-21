from fastapi import FastAPI, HTTPException
from corpus_retrieval.scrapers.semantic_scholars_scraping import search_and_download_from_semantic_scholars
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

from corpus_retrieval.summarizers.llama_summarizer import generate_ingredient_summary
from corpus_retrieval.summarizers.llama_summarizer import LLM_TYPE_OPENAI, LLM_TYPE_OLLAMA

# Vérifier la variable d'environnement OpenAI au démarrage
has_openai_key = os.environ.get("OPENAI_API_KEY") is not None
if has_openai_key:
    print(f"Variable d'environnement OPENAI_API_KEY détectée: {os.environ.get('OPENAI_API_KEY')[:5]}...")
else:
    print("ATTENTION: Variable d'environnement OPENAI_API_KEY non détectée.")



# Point d'entrée du backend
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Front React
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/debug/env/")
async def debug_env():
    """
    Endpoint de débogage pour vérifier les variables d'environnement disponibles.
    """
    import sys
    
    # Collecter les informations de debug
    env_info = {
        "OPENAI_API_KEY_exists": os.environ.get("OPENAI_API_KEY") is not None,
        "OPENAI_API_KEY_length": len(os.environ.get("OPENAI_API_KEY", "")) if os.environ.get("OPENAI_API_KEY") else 0,
        "OPENAI_API_KEY_prefix": os.environ.get("OPENAI_API_KEY", "")[:5] + "..." if os.environ.get("OPENAI_API_KEY") else None,
        "env_keys": list(os.environ.keys()),
        "python_version": sys.version,
        "python_path": sys.executable,
        "current_dir": os.getcwd()
    }
    
    return env_info

@app.get("/")
def read_root():
    return {"message": "Welcome to bioQuery API!"}

@app.get("/search/semantic_scholars/")
async def search_semantic_scholars(ingredient: str, use_cache: bool = True, use_openai_for_query: bool = None):
    """ 
    Recherche des articles sur Semantic Scholars avec support de cache.
    Utilise OpenAI pour générer la requête si la variable d'environnement est définie.
    """
    results = search_and_download_from_semantic_scholars(
        ingredient, 
        use_cache=use_cache,
        use_openai_for_query=use_openai_for_query
    )
    return {"results": results}

@app.get("/search/complete/")
async def search_complete(ingredient: str, use_openai: bool = False, use_cache: bool = True, use_openai_for_query: bool = None):
    """
    Effectue le processus complet avec OpenAI si demandé ou si la variable d'environnement est définie.
    Utilise le cache local pour les articles si disponible.
    """
    try:
        # Vérifier si une API key OpenAI est définie
        has_openai_key = os.environ.get("OPENAI_API_KEY") is not None
        
        # Décider d'utiliser OpenAI ou Llama pour le résumé
        use_openai_model = use_openai or has_openai_key
        
        # Logs pour debug
        print(f"[DEBUG] use_openai (paramètre): {use_openai}")
        print(f"[DEBUG] has_openai_key (variable d'environnement): {has_openai_key}")
        print(f"[DEBUG] use_openai_model (décision finale): {use_openai_model}")
        print(f"[DEBUG] use_cache (paramètre): {use_cache}")
        print(f"[DEBUG] use_openai_for_query (paramètre): {use_openai_for_query if use_openai_for_query is not None else 'auto'}")
        
        # 1. Recherche d'articles (avec support de cache et option OpenAI pour la requête)
        semantic_results = search_and_download_from_semantic_scholars(
            ingredient, 
            use_cache=use_cache,
            use_openai_for_query=use_openai_for_query
        )
        
        if isinstance(semantic_results, dict) and "error" in semantic_results:
            return {
                "status": "warning",
                "message": f"Problème lors de la recherche d'articles pour {ingredient}: {semantic_results['error']}",
                "semantic_results": [],
                "summary": None
            }
        
        if not semantic_results or len(semantic_results) == 0:
            return {
                "status": "warning",
                "message": f"Aucun article trouvé pour {ingredient}. Impossible de générer un résumé.",
                "semantic_results": [],
                "summary": None
            }
        
        # 2. Génération du résumé
        try:
            # Déterminer le type de LLM à utiliser
            llm_type = LLM_TYPE_OPENAI if use_openai_model else LLM_TYPE_OLLAMA
            
            if use_openai_model:
                print(f"Utilisation d'OpenAI pour la génération du résumé de {ingredient}")
                # Modèle à utiliser avec OpenAI
                model_name = "gpt-3.5-turbo"
            else:
                print(f"Utilisation de Llama local pour la génération du résumé de {ingredient}")
                # Laisser le modèle par défaut pour Llama
                model_name = None
            
            # Mise à jour des paramètres pour correspondre à la signature actuelle
            summary_result = generate_ingredient_summary(
                ingredient=ingredient,
                save_to_file=True,
                max_chunks=50,
                llm_type=llm_type,
                model_name=model_name
            )
        except Exception as e:
            print(f"Erreur lors de la génération du résumé: {str(e)}")
            return {
                "status": "partial_success",
                "message": f"Articles récupérés mais échec de la génération du résumé: {str(e)}",
                "semantic_results": semantic_results[:3] if len(semantic_results) > 3 else semantic_results,
                "summary": None
            }
        
        # Filtrer les articles
        top_articles = semantic_results[:3] if len(semantic_results) > 3 else semantic_results
        
        # Déterminer si les articles viennent du cache
        using_cache = any(article.get("from_cache", False) for article in semantic_results) if semantic_results else False
        source_info = "cache local" if using_cache else "Semantic Scholar API"
        
        # Déterminer quel modèle a été utilisé pour la requête
        query_model = "OpenAI" if has_openai_key else "Llama local"
        if use_openai_for_query is not None:  # Si explicitement spécifié
            query_model = "OpenAI" if use_openai_for_query else "Llama local"
        
        return {
            "status": "success",
            "message": f"Recherche complète et génération de résumé réussies pour {ingredient}.",
            "source": source_info,
            "query_model": query_model,
            "semantic_results": top_articles,
            "summary": {
                "text": summary_result.get("summary"),
                "id": summary_result.get("summary_id"),
                "processing_time": summary_result.get("processing_time"),
                "chunks_processed": summary_result.get("chunks_processed"),
                "model_used": "OpenAI" if use_openai_model else "Llama local"
            }
        }
    
    except Exception as e:
        # Gestion des erreurs
        error_message = f"Erreur lors du processus complet pour {ingredient}: {str(e)}"
        raise HTTPException(status_code=500, detail=error_message)

@app.get("/search/both/")
async def search_both(ingredient: str, use_cache: bool = True, use_openai_for_query: bool = None):
    """ Recherche des articles sur Semantic Scholars et d'autres sources. """
    semantic_results = search_and_download_from_semantic_scholars(
        ingredient, 
        use_cache=use_cache,
        use_openai_for_query=use_openai_for_query
    )
    
    # Combiner les résultats des deux recherches
    all_results = {
        "semantic_scholars": semantic_results,
    }
    return {"results": all_results}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info")