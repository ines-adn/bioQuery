import os
from enum import Enum

from docling.chunking import HybridChunker
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType


class FileExtensions(Enum):
    PDF = ".pdf"
    DOCX = ".docx"
    TXT = ".txt"
    JSON = ".json"
    CSV = ".csv"
    XLSX = ".xlsx"
    MD = ".md"


class DocLoaderDocling:
    def __init__(
        self,
        embeddings_model_name: str,
        export_type: ExportType = ExportType.DOC_CHUNKS,
    ):
        self.embeddings_model_name = embeddings_model_name
        self.export_type = export_type

    def get_splits_from_path(self, file_path: str):
        loader = DoclingLoader(
            file_path=file_path,
            export_type=self.export_type,
            chunker=HybridChunker(tokenizer=self.embeddings_model_name, max_tokens=512),
        )
        splits = loader.load()
        return splits

    def get_splits_from_folder(
        self, folder_path: str, extensions: list[FileExtensions], recursive: bool = True
    ):
        files = self._get_files_from_folder(folder_path, extensions, recursive)

        splits = []
        for file in files:
            splits.extend(self.get_splits_from_path(file))

        return splits

    @staticmethod
    def _get_files_from_folder(
        folder_path: str,
        extensions: list[FileExtensions] = [FileExtensions.PDF],
        recursive: bool = True,
    ):
        if not os.path.isdir(folder_path):
            raise NotADirectoryError(f"{folder_path} is not a directory")
        files = []

        for root, _, filenames in os.walk(folder_path):
            for filename in filenames:
                if filename.endswith(tuple(map(lambda x: x.value, extensions))):
                    files.append(os.path.join(root, filename))
            if not recursive:
                break

        return files


if __name__ == "__main__":
    docLoader = DocLoaderDocling(
        "HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1.5"
    )
    splits = docLoader.get_splits_from_folder(
        "dossier_test", [FileExtensions.PDF, FileExtensions.DOCX]
    )
    print(splits)
