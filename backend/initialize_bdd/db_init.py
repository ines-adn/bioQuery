from initialize_bdd.db_models import Chunk, Document, Extrait
from sqlmodel import Field, Session, SQLModel, create_engine, select
from parsers._docling import DocumentParser  # Adapter selon tes modules
import os

SQLALCHEMY_DATABASE_URL = "postgresql://user:password@localhost:5432/mydatabase"  # ⚠️ À adapter


def get_session(engine):
    with Session(engine) as session:
        yield session


def createEngine():
    return create_engine(SQLALCHEMY_DATABASE_URL)


def create_db_and_tables():
    print("Creating database and tables")
    engine = createEngine()
    SQLModel.metadata.create_all(engine)


def insertElement(session, element: SQLModel):
    session.add(element)
    session.commit()
    session.refresh(element)
    return element


def insertExtract(session, extract_content, metadata, document_id):
    extract = Extrait(
        contenu_md=extract_content,  # Markdown ou autre format selon ton parser
        contenu_plain_text=extract_content,  # Adaptable selon les besoins
        page_no=metadata.get("page_no", None),
        top=metadata.get("top", None),
        bottom=metadata.get("bottom", None),
        page_height=metadata.get("page_height", None),
        document_id=document_id,
    )
    insertElement(session, extract)
    return extract


def insertDocument(session, document_parser):
    print("Inserting document to database")
    document = Document(
        filename=document_parser.filename,
        modified_date=document_parser.modified_date,
        blob=document_parser.get_blob_file(),
    )
    insertElement(session, document)
    return document


def insertChunk(session, chunk):
    query = select(Extrait).where(Extrait.id.in_(chunk.block_ids))
    extraits = session.exec(query).all()
    chunkDB = Chunk(text=chunk.plain_text, extraits=extraits)
    insertElement(session, chunkDB)
    return chunkDB


def insertDocumentExtractsToDatabase(document_parser):
    session = next(get_session(createEngine()))
    document = insertDocument(session, document_parser)

    print("Inserting extracts to database")
    for extract_content, metadata in document_parser.get_extracts():
        extractDB = insertExtract(session, extract_content, metadata, document.id)

    print("Extracts inserted successfully")


def process_corpus_documents(corpus_directory):
    for filename in os.listdir(corpus_directory):
        if filename.endswith(".pdf") or filename.endswith(".txt"):  # Adapter selon tes formats
            file_path = os.path.join(corpus_directory, filename)
            document_parser = DocumentParser(file_path)  # Adapter selon ton module
            insertDocumentExtractsToDatabase(document_parser)


if __name__ == "__main__":
    create_db_and_tables()
    corpus_directory = "/path/to/corpus/documents"  # ⚠️ À adapter
    process_corpus_documents(corpus_directory)