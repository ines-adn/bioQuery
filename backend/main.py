from fastapi import FastAPI, HTTPException
from corpus_retrieval.scrapers.semantic_scholars_scraping import search_and_download_from_semantic_scholars
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Importer la fonction de génération de résumé
from corpus_retrieval.summarizers.llama_summarizer import generate_ingredient_summary

# Point d'entrée du backend
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Front React
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to bioQuery API!"}

@app.get("/search/semantic_scholars/")
async def search_semantic_scholars(ingredient: str):
    """ Recherche des articles sur Semantic Scholars. """
    results = search_and_download_from_semantic_scholars(ingredient)
    return {"results": results}

@app.get("/search/complete/")
async def search_complete(ingredient: str, use_openai: bool = False):
    """
    Effectue le processus complet avec OpenAI si demandé ou si la variable d'environnement est définie
    """
    try:
        # Vérifier si une API key OpenAI est définie
        import os
        has_openai_key = os.environ.get("OPENAI_API_KEY") is not None
        
        # Décider d'utiliser OpenAI ou Llama
        use_openai_model = use_openai or has_openai_key
        
        # 1. Recherche d'articles
        semantic_results = search_and_download_from_semantic_scholars(ingredient)
        
        if not semantic_results or len(semantic_results) == 0:
            return {
                "status": "warning",
                "message": f"Aucun article trouvé pour {ingredient}. Impossible de générer un résumé.",
                "semantic_results": semantic_results,
                "summary": None
            }
        
        # 2. Génération du résumé
        try:
            if use_openai_model:
                print(f"Utilisation d'OpenAI pour la génération du résumé de {ingredient}")
            else:
                print(f"Utilisation de Llama local pour la génération du résumé de {ingredient}")
                
            summary_result = generate_ingredient_summary(
                ingredient=ingredient,
                save_to_file=True,
                max_chunks=50,
                use_openai=use_openai_model,
                openai_model="gpt-3.5-turbo"  # Vous pouvez aussi utiliser "gpt-4" si vous y avez accès
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
        
        return {
            "status": "success",
            "message": f"Recherche complète et génération de résumé réussies pour {ingredient}.",
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
async def search_both(ingredient: str):
    """ Recherche des articles sur Semantic Scholars et d'autres sources. """
    semantic_results = search_and_download_from_semantic_scholars(ingredient)
    
    # Combiner les résultats des deux recherches
    all_results = {
        "semantic_scholars": semantic_results,
    }
    return {"results": all_results}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info")