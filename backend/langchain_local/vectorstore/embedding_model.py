# Modèle pour convertir les textes en vecteurs

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector


class VectorStorePG(PGVector):
    def __init__(
        self, embeddings_model_name: str, collection_name: str, connection: str
    ):
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
        super().__init__(
            embeddings=embeddings,
            collection_name=collection_name,
            connection=connection,
            use_jsonb=True,
        )
