# Fonction principale de retrieval avec LangChain

from abc import ABC, abstractmethod
from langchain_core.documents import Document
from langchain_core.vectorstores.base import VectorStore
from sklearn.feature_extraction.text import TfidfVectorizer
import time

class Retrieval(ABC):
    @abstractmethod
    def retrieve_docs(self, vector_store: VectorStore, query: str) -> list[Document]:
        pass

class NaiveRetrieval(Retrieval):
    def retrieve_docs(self, vector_store: VectorStore, query: str) -> list[Document]:
        retrieved_docs = vector_store.similarity_search(query)
        return retrieved_docs


class KeywordFilteredRAG(Retrieval):
    def retrieve_docs(self, vector_store: VectorStore, query: str, max_results: int = 3) -> list[Document]:
        print("retrieving docs with keyword filtering")
        retrieved_docs = []
        attempts = 0
        max_attempts = 3
        keywords = self.extract_keywords(query)  # Étape 1 : Extraire les mots-clés
        print(f"Extracted keywords: {keywords}")

        while attempts < max_attempts and len(retrieved_docs) < max_results:
            results = vector_store.similarity_search(query)  # Recherche initiale
            if not results:
                break  # Stop if no relevant documents are found

            # Étape 2 : Filtrer les documents en fonction des mots-clés extraits
            filtered_results = self.filter_documents_by_keywords(results, keywords)
            retrieved_docs.extend(filtered_results)

            # Étape 3 : Affiner la requête avec les mots-clés filtrés
            query = self.refine_query(query, filtered_results)
            attempts += 1

        return retrieved_docs[:max_results]

    def extract_keywords(self, query: str) -> list[str]:
        """Extraire des mots-clés de la requête à l'aide de TF-IDF ou d'une autre méthode."""
        # Utilisation de TF-IDF pour extraire les mots-clés
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([query])
        feature_names = vectorizer.get_feature_names_out()
        
        # Obtenir les mots-clés en fonction de leur score TF-IDF
        sorted_keywords = sorted(zip(feature_names, tfidf_matrix.sum(axis=0).A1), key=lambda x: x[1], reverse=True)
        keywords = [keyword for keyword, score in sorted_keywords[:5]]  # Garder les 5 mots-clés les plus importants
        return keywords

    def filter_documents_by_keywords(self, documents: list[Document], keywords: list[str]) -> list[Document]:
        """Filtrer les documents en fonction des mots-clés extraits."""
        filtered_docs = []
        for doc in documents:
            # Vérifier si le document contient l'un des mots-clés
            if any(keyword.lower() in doc.page_content.lower() for keyword in keywords):
                filtered_docs.append(doc)
        return filtered_docs

    def refine_query(self, query: str, filtered_results: list[Document]) -> str:
        """Affiner la requête avec les résultats filtrés."""
        relevant_context = " ".join(doc.page_content for doc in filtered_results)
        return f"{query} {relevant_context}".strip()