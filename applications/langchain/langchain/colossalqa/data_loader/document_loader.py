'''
class for loading document type data
'''

import os
import logging
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import JSONLoader
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from typing import List, Union, Dict, Any
import glob


SUPPORTED_DATA_FORMAT = ['.csv','.xlsx', '.xls','.json','.html','.sql','.h5', '.hdf5','.parquet','.feather','.msgpack','.dta','.dta', 'txt']


class DocumentLoader:
    def __init__(self, files:List[List[str]], **kwargs) -> None:
        '''
        Args:
            files: list of files (list[file path, name])
            **kwargs: keyword type arguments, useful for certain document types 
        '''
        self.data = {}
        self.kwargs = kwargs
        for item in files:
            path = item[0]
            files = []
            try:
                files = glob.glob(path)
            except Exception:
                pass
            if len(files)==0:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"{path} doesn't exists")
                if not any([path.endswith(i) for i in SUPPORTED_DATA_FORMAT]):
                    raise TypeError(f"{path} not supported. Supported type {SUPPORTED_DATA_FORMAT}")
            
            print("loading data")
            self.load_data(path)
            print("data loaded")

        self.all_data = []
        for key in self.data:
            if isinstance(self.data[key], list):
                for item in self.data[key]:
                    if isinstance(item, list):
                        self.all_data.extend(item)
                    else:
                        self.all_data.append(item)

    def load_data(self, path:str) -> None:
        '''
        load data. Please refer to https://python.langchain.com/docs/modules/data_connection/document_loaders/ 
            for sepcific format requirements.
        Args:
            path: path to a file
        '''
        print(f"load {path}")
        if path.endswith('.csv'):
            # load csv
            loader = CSVLoader(file_path=path, encoding='utf8')
            data = loader.load()
            self.data[path] = data
        elif path.endswith('.txt'):
            # load txt
            loader = TextLoader(path, encoding='utf8')
            data = loader.load()
            self.data[path] = data
        elif path.endswith('html'):
            # load html
            loader = UnstructuredHTMLLoader(path, encoding='utf8')
            data = loader.load()
            self.data[path] = data
        elif path.endswith('json'):
            # load json
            loader = JSONLoader(file_path=path,jq_schema=self.kwargs.get('jq_schema','.data[].content'))
            data = loader.load()
            self.data[path] = data
        elif path.endswith('jsonl'):
            # load jsonl
            loader = JSONLoader(file_path=path,jq_schema=self.kwargs.get('jq_schema','.data[].content'),json_lines=True)
            data = loader.load()
            self.data[path] = data
        elif path.endswith(".md"):
            # load markdown
            loader = UnstructuredMarkdownLoader(path)
            data = loader.load()
            self.data[path] = data
        elif path.endswith(".pdf"):
            # load pdf
            loader = PyPDFLoader(path)
            data = loader.load_and_split()
            self.data[path] = data
        else:
            file = []
            try:
                files = glob.glob(path)
            except Exception:
                pass
            if len(files)==0:
                raise ValueError("Unsupported file/directory format. For directories, please use glob expression")
            else:
                loader = DirectoryLoader(path)
                self.data[path] = data
        
        
        




