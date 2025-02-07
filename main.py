from fastapi import FastAPI

# Point d’entrée du backend

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the AromaZone Retrieval Tool"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}