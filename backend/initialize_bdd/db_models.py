from datetime import datetime
from typing import List, Optional
from sqlmodel import Field, Relationship, SQLModel

# Table des documents
class Document(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    filename: str  # Nom du fichier
    modified_date: datetime  # Date de modification du document
    blob: Optional[bytes]  # Contenu binaire du document

# Table pour la relation Many-to-Many entre Chunk et Extrait
class ChunkExtrait(SQLModel, table=True):
    """Table d'association entre Chunk et Extrait"""
    chunk_id: Optional[int] = Field(default=None, foreign_key="chunk.id", primary_key=True)
    extrait_id: Optional[int] = Field(default=None, foreign_key="extrait.id", primary_key=True)

# Table des chunks
class Chunk(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    text: str  # Texte brut du chunk

    # Relation Many-to-Many avec Extrait via la table d'association
    extraits: List["Extrait"] = Relationship(back_populates="chunks", link_model=ChunkExtrait)

# Table des extraits
class Extrait(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    contenu_md: str  # Contenu Markdown de l'extrait
    contenu_plain_text: str  # Contenu texte brut de l'extrait
    page_no: int  # Numéro de la page
    page_height: float  # Hauteur de la page
    top: float  # Position verticale du texte
    bottom: float  # Position verticale du bas du texte
    document_id: int = Field(foreign_key="document.id")  # Clé étrangère vers le document

    # Relation Many-to-Many avec Chunk via la table ChunkExtrait
    chunks: List["Chunk"] = Relationship(back_populates="extraits", link_model=ChunkExtrait)