# Gestion des requêtes utilisateurs

import os
from langchain_local.vectorstore.embedding_model import VectorStorePG
from yaml import load, Loader


class QueryHandler:
    def __init__(
        self,
        config_path: str = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "config.yml"
        ),
    ):
        # Chargement de la configuration
        self.config = load(open(config_path), Loader)

        # Initialisation du stockage des embeddings
        self.vector_store = VectorStorePG(
            embeddings_model_name=self.config["vector_store"]["embedings_hf_model"],
            collection_name=self.config["vector_store"]["collection_name"],
            connection=self.config["vector_store"]["sql_db_url"],
        )

    def retrieve_documents(self, query: str):
        """Récupère et retourne les documents pertinents en fonction de la requête."""
        retrieved_docs = self.vector_store.query(query)

        if not retrieved_docs:
            return "Aucun document pertinent trouvé."

        docs_info = []

        for doc in retrieved_docs:
            filename = doc.metadata.get('dl_meta', {}).get('origin', {}).get('filename', 'Inconnu')
            page_no = None

            for item in doc.metadata.get('dl_meta', {}).get('doc_items', []):
                for prov in item.get('prov', []):
                    page_no = prov.get('page_no')

            page_info = f"Page {page_no}" if page_no else "Page inconnue"
            docs_info.append(f"{filename} ({page_info})")

        return docs_info


if __name__ == "__main__":
    query_handler = QueryHandler()
    response = query_handler.retrieve_documents("Effet photoélectrique")
    print("\n".join(response) if isinstance(response, list) else response)