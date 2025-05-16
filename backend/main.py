from fastapi import FastAPI
from corpus_retrieval.scrapers.semantic_scholars_scraping import search_and_download_from_semantic_scholars
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Point d’entrée du backend

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