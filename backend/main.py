from fastapi import FastAPI, HTTPException
from corpus_retrieval.scrapers.semantic_scholars_scraping import search_and_download_from_semantic_scholars
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from fastapi import APIRouter
from corpus_retrieval.summarizers.llama_summarizer import generate_ingredient_summary
from corpus_retrieval.summarizers.llama_summarizer import LLM_TYPE_OPENAI, LLM_TYPE_OLLAMA
from corpus_retrieval.parsers.article_chunker import process_downloaded_articles
from corpus_retrieval.parsers.embedding_store import store_article_chunks


# Check if the environment variable OPENAI_API_KEY is set
has_openai_key = os.environ.get("OPENAI_API_KEY") is not None
if has_openai_key:
    print(f"Environment variable OPENAI_API_KEY detected: {os.environ.get('OPENAI_API_KEY')[:5]}...")
else:
    print("WARNING: Environment variable OPENAI_API_KEY not detected.")


# Backend FastAPI application setup
app = FastAPI()
router = APIRouter()
app.include_router(router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # React app running on port 3000
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to bioQuery API!"}

@app.get("/search/complete/")
async def search_complete(
    ingredient: str, 
    use_openai: bool = False, 
    use_cache: bool = False, 
    use_openai_for_query: bool = None,
    max_chunks: int = 10,          
    max_chunks_ollama: int = 3,      
    max_tokens_ollama: int = 500,   
    process_pdfs: bool = True,       
    language: str = "fr"            
):
    """
    Performs the complete process using OpenAI if requested or if the environment variable is set.
    Uses the local cache for articles if available.
    Automatically processes PDFs if process_pdfs=True.
    Allows specifying specific limits for Ollama.
    Generates the summary in the specified language (fr or en).
    """
    try:
        if language not in ["fr", "en"]:
            language = "fr"
            
        has_openai_key = os.environ.get("OPENAI_API_KEY") is not None
        
        use_openai_model = use_openai or has_openai_key
        
        # Debugging information
        print(f"[DEBUG] use_openai (paramètre): {use_openai}")
        print(f"[DEBUG] has_openai_key (variable d'environnement): {has_openai_key}")
        print(f"[DEBUG] use_openai_model (décision finale): {use_openai_model}")
        print(f"[DEBUG] use_cache (paramètre): {use_cache}")
        print(f"[DEBUG] use_openai_for_query (paramètre): {use_openai_for_query if use_openai_for_query is not None else 'auto'}")
        print(f"[DEBUG] max_chunks (paramètre): {max_chunks}")
        print(f"[DEBUG] max_chunks_ollama (paramètre): {max_chunks_ollama}")
        print(f"[DEBUG] max_tokens_ollama (paramètre): {max_tokens_ollama}")
        print(f"[DEBUG] process_pdfs (paramètre): {process_pdfs}")
        print(f"[DEBUG] language (paramètre): {language}")
        
        ingredient_normalized = ingredient.lower().replace(" ", "_")
        
        semantic_results = search_and_download_from_semantic_scholars(
            ingredient, 
            use_cache=use_cache,
            use_openai_for_query=use_openai_for_query
        )
        
        if isinstance(semantic_results, dict) and "error" in semantic_results:
            return {
                "status": "warning",
                "message": f"Problem during article search for {ingredient}: {semantic_results['error']}",
                "semantic_results": [],
                "summary": None
            }
        
        if not semantic_results or len(semantic_results) == 0:
            return {
                "status": "warning",
                "message": f"No articles found for {ingredient}. Unable to generate a summary.",
                "semantic_results": [],
                "summary": None
            }
        
        if process_pdfs:
            print(f"Automatically processing PDFs for {ingredient_normalized}...")
            
            processing_results = process_downloaded_articles(ingredient_normalized)
            
            if processing_results.get("status") != "success":
                print(f"Warning: Problem during PDF processing: {processing_results.get('message')}")
            else:
                chunks = processing_results.get("chunks", [])
                
                if not chunks:
                    print(f"Warning: No chunks extracted from PDFs for {ingredient}")
                else:
                    embedding_results = store_article_chunks(chunks, ingredient_normalized, overwrite=False)
                    if embedding_results.get("status") != "success":
                        print(f"Warning: Problem while storing embeddings: {embedding_results.get('message')}")
                    else:
                        print(f"Embeddings successfully created: {embedding_results.get('stored_chunks')} chunks stored")
        
        try:
            llm_type = LLM_TYPE_OPENAI if use_openai_model else LLM_TYPE_OLLAMA
            
            if use_openai_model:
                print(f"Using OpenAI for summary generation of {ingredient} in {language}")
                model_name = "gpt-4.1"
                
                summary_result = generate_ingredient_summary(
                    ingredient=ingredient_normalized,
                    save_to_file=True,
                    max_chunks=max_chunks,
                    llm_type=llm_type,
                    model_name=model_name,
                    language=language
                )
            else:
                print(f"Using local Llama for summary generation of {ingredient} in {language}")
                model_name = None
                
               
                summary_result = generate_ingredient_summary(
                    ingredient=ingredient_normalized,
                    save_to_file=True,
                    max_chunks=max_chunks,
                    llm_type=llm_type,
                    model_name=model_name,
                    max_chunks_ollama=max_chunks_ollama,
                    max_tokens_ollama=max_tokens_ollama,
                    language=language 
                )
        except Exception as e:
            print(f"Error during summary generation: {str(e)}")
            return {
            "status": "partial_success",
            "message": f"Articles retrieved but failed to generate summary: {str(e)}",
            "semantic_results": semantic_results[:3] if len(semantic_results) > 3 else semantic_results,
            "summary": None
            }
        
        top_articles = semantic_results[:3] if len(semantic_results) > 3 else semantic_results
        
        using_cache = any(article.get("from_cache", False) for article in semantic_results) if semantic_results else False
        source_info = "cache local" if using_cache else "Semantic Scholar API"
        
        query_model = "OpenAI" if has_openai_key else "Llama local"
        if use_openai_for_query is not None: 
            query_model = "OpenAI" if use_openai_for_query else "Llama local"
        
        return {
            "status": "success",
            "message": f"Full search and summary generation successful for {ingredient}.",
            "source": source_info,
            "query_model": query_model,
            "max_chunks_used": max_chunks,
            "semantic_results": top_articles,
            "summary": {
            "text": summary_result.get("summary"),
            "id": summary_result.get("summary_id"),
            "processing_time": summary_result.get("processing_time"),
            "chunks_processed": summary_result.get("chunks_processed"),
            "model_used": "OpenAI" if use_openai_model else "Llama local",
            "language": language  # Include language in the response
            }
        }
    
    except Exception as e:
        error_message = f"Error during the complete process for {ingredient}: {str(e)}"
        raise HTTPException(status_code=500, detail=error_message)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info")