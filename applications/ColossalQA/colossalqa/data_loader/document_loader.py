"""
Class for loading document type data
"""

import glob
from typing import List

from colossalqa.mylogging import get_logger
from langchain.document_loaders import (
    JSONLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
)
from langchain.document_loaders.csv_loader import CSVLoader

logger = get_logger()

SUPPORTED_DATA_FORMAT = [".csv", ".json", ".html", ".md", ".pdf", ".txt", ".jsonl"]


class DocumentLoader:
    """
    Load documents from different files into list of langchain Documents
    """

    def __init__(self, files: List, **kwargs) -> None:
        """
        Args:
            files: list of files (list[file path, name])
            **kwargs: keyword type arguments, useful for certain document types
        """
        self.data = {}
        self.kwargs = kwargs

        for item in files:
            path = item[0] if isinstance(item, list) else item
            logger.info(f"Loading data from {path}")
            self.load_data(path)
            logger.info("Data loaded")

        self.all_data = []
        for key in self.data:
            if isinstance(self.data[key], list):
                for item in self.data[key]:
                    if isinstance(item, list):
                        self.all_data.extend(item)
                    else:
                        self.all_data.append(item)

    def load_data(self, path: str) -> None:
        """
        Load data. Please refer to https://python.langchain.com/docs/modules/data_connection/document_loaders/
            for sepcific format requirements.
        Args:
            path: path to a file
                To load files with glob path, here are some examples.
                    Load all file from directory: folder1/folder2/*
                    Load all pdf file from directory: folder1/folder2/*.pdf
        """
        files = []

        # Handle glob expression
        try:
            files = glob.glob(path)
        except Exception as e:
            logger.error(e)
        if len(files) == 0:
            raise ValueError("Unsupported file/directory format. For directories, please use glob expression")
        elif len(files) == 1:
            path = files[0]
        else:
            for file in files:
                self.load_data(file)
            return

        # Load data if the path is a file
        logger.info(f"load {path}", verbose=True)
        if path.endswith(".csv"):
            # Load csv
            loader = CSVLoader(file_path=path, encoding="utf8")
            data = loader.load()
            self.data[path] = data
        elif path.endswith(".txt"):
            # Load txt
            loader = TextLoader(path, encoding="utf8")
            data = loader.load()
            self.data[path] = data
        elif path.endswith("html"):
            # Load html
            loader = UnstructuredHTMLLoader(path, encoding="utf8")
            data = loader.load()
            self.data[path] = data
        elif path.endswith("json"):
            # Load json
            loader = JSONLoader(
                file_path=path,
                jq_schema=self.kwargs.get("jq_schema", ".data[]"),
                content_key=self.kwargs.get("content_key", "content"),
                metadata_func=self.kwargs.get("metadata_func", None),
            )

            data = loader.load()
            self.data[path] = data
        elif path.endswith("jsonl"):
            # Load jsonl
            loader = JSONLoader(
                file_path=path, jq_schema=self.kwargs.get("jq_schema", ".data[].content"), json_lines=True
            )
            data = loader.load()
            self.data[path] = data
        elif path.endswith(".md"):
            # Load markdown
            loader = UnstructuredMarkdownLoader(path)
            data = loader.load()
            self.data[path] = data
        elif path.endswith(".pdf"):
            # Load pdf
            loader = PyPDFLoader(path)
            data = loader.load_and_split()
            self.data[path] = data
        else:
            if "." in path.split("/")[-1]:
                raise ValueError(f"Unsupported file format {path}. Supported formats: {SUPPORTED_DATA_FORMAT}")
            else:
                # May ba a directory, we strictly follow the glob path and will not load files in subdirectories
                pass
    
    def clear(self):
        """
        Clear loaded data.
        """
        self.data = {}
        self.kwargs = {}
        self.all_data = []
