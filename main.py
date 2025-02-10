from fastapi import FastAPI
from backend.corpus_retrieval.scrapers.semantic_scholars_scraping import search_and_download_from_semantic_scholars
from backend.corpus_retrieval.scrapers.pubmed_scraping import search_and_download_from_pubmed

# Point d’entrée du backend

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the AromaZone Retrieval Tool"}

# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: str = None):
#     return {"item_id": item_id, "q": q}



@app.get("/search/semantic_scholars/")
async def search_semantic_scholars(ingredient: str, allegation: str):
    """ Recherche des articles sur Semantic Scholars. """
    results = search_and_download_from_semantic_scholars(ingredient, allegation)
    return {"results": results}

@app.get("/search/pubmed/")
async def search_pubmed(ingredient: str, allegation: str):
    """ Recherche des articles sur PubMed. """
    results = search_and_download_from_pubmed(ingredient, allegation)
    return {"results": results}

@app.get("/search/both/")
async def search_both(ingredient: str, allegation: str):
    """ Recherche des articles sur Semantic Scholars et PubMed. """
    semantic_results = search_and_download_from_semantic_scholars(ingredient, allegation)
    pubmed_results = search_and_download_from_pubmed(ingredient, allegation)
    
    # Combiner les résultats des deux recherches
    all_results = {
        "semantic_scholars": semantic_results,
        "pubmed": pubmed_results
    }

    return {"results": all_results}