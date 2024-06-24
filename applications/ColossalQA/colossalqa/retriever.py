"""
Code for custom retriver with incremental update
"""

import copy
import hashlib
import os
from collections import defaultdict
from typing import Any, Callable, Dict, List

from colossalqa.mylogging import get_logger
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.embeddings.base import Embeddings
from langchain.indexes import SQLRecordManager, index
from langchain.schema.retriever import BaseRetriever, Document
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.chroma import Chroma

logger = get_logger()


class CustomRetriever(BaseRetriever):
    """
    Custom retriever class with support for incremental update of indexes
    """

    vector_stores: Dict[str, VectorStore] = {}
    sql_index_database: Dict[str, str] = {}
    record_managers: Dict[str, SQLRecordManager] = {}
    sql_db_chains = []
    k = 3
    rephrase_handler: Callable = None
    buffer: Dict = []
    buffer_size: int = 5
    verbose: bool = False
    sql_file_path: str = None

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embeddings: Embeddings,
        **kwargs: Any,
    ) -> BaseRetriever:
        k = kwargs.pop("k", 3)
        cleanup = kwargs.pop("cleanup", "incremental")
        mode = kwargs.pop("mode", "by_source")
        ret = cls(k=k)
        ret.add_documents(documents, embedding=embeddings, cleanup=cleanup, mode=mode)
        return ret

    def add_documents(
        self,
        docs: Dict[str, Document] = [],
        cleanup: str = "incremental",
        mode: str = "by_source",
        embedding: Embeddings = None,
    ) -> None:
        """
        Add documents to retriever
        Args:
            docs: the documents to add
            cleanup: choose from "incremental" (update embeddings, skip existing embeddings) and "full" (destroy and rebuild retriever)
            mode: choose from "by source" (documents are grouped by source) and "merge" (documents are merged into one vector store)
        """
        if cleanup == "full":
            # Cleanup
            for source in self.vector_stores:
                os.remove(self.sql_index_database[source])
        # Add documents
        data_by_source = defaultdict(list)
        if mode == "by_source":
            for doc in docs:
                data_by_source[doc.metadata["source"]].append(doc)
        elif mode == "merge":
            data_by_source["merged"] = docs

        for source in data_by_source:
            if source not in self.vector_stores:
                hash_encoding = hashlib.sha3_224(source.encode()).hexdigest()
                if os.path.exists(f"{self.sql_file_path}/{hash_encoding}.db"):
                    # Remove the stale file
                    os.remove(f"{self.sql_file_path}/{hash_encoding}.db")
                # Create a new sql database to store indexes, sql files are stored in the same directory as the source file
                sql_path = f"sqlite:///{self.sql_file_path}/{hash_encoding}.db"
                # to record the sql database with their source as index
                self.sql_index_database[source] = f"{self.sql_file_path}/{hash_encoding}.db"

                self.vector_stores[source] = Chroma(embedding_function=embedding, collection_name=hash_encoding)
                self.record_managers[source] = SQLRecordManager(source, db_url=sql_path)
                self.record_managers[source].create_schema()
            index(
                data_by_source[source],
                self.record_managers[source],
                self.vector_stores[source],
                cleanup=cleanup,
                source_id_key="source",
            )

    def clear_documents(self):
        """Clear all document vectors from database"""
        for source in self.vector_stores:
            index([], self.record_managers[source], self.vector_stores[source], cleanup="full", source_id_key="source")
        self.vector_stores = {}
        self.sql_index_database = {}
        self.record_managers = {}

    def __del__(self):
        for source in self.sql_index_database:
            if os.path.exists(self.sql_index_database[source]):
                os.remove(self.sql_index_database[source])

    def set_sql_database_chain(self, db_chains) -> None:
        """
        set sql agent chain to retrieve information from sql database
        Not used in this version
        """
        self.sql_db_chains = db_chains

    def set_rephrase_handler(self, handler: Callable = None) -> None:
        """
        Set a handler to preprocess the input str before feed into the retriever
        """
        self.rephrase_handler = handler

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun = None,
        score_threshold: float = None,
        return_scores: bool = False,
    ) -> List[Document]:
        """
        This function is called by the retriever to get the relevant documents.
        recent vistied queries are stored in buffer, if the query is in buffer, return the documents directly

        Args:
            query: the query to be searched
            run_manager: the callback manager for retriever run
        Returns:
            documents: the relevant documents
        """
        for buffered_doc in self.buffer:
            if buffered_doc[0] == query:
                return buffered_doc[1]
        query_ = str(query)
        # Use your existing retriever to get the documents
        if self.rephrase_handler:
            query = self.rephrase_handler(query)
        documents = []
        for k in self.vector_stores:
            # Retrieve documents from each retriever
            vectorstore = self.vector_stores[k]
            documents.extend(vectorstore.similarity_search_with_score(query, self.k, score_threshold=score_threshold))
            # print(documents)
        # Return the top k documents among all retrievers
        documents = sorted(documents, key=lambda x: x[1], reverse=False)[: self.k]
        if return_scores:
            # Return score
            documents = copy.deepcopy(documents)
            for doc in documents:
                doc[0].metadata["score"] = doc[1]
        documents = [doc[0] for doc in documents]
        # Retrieve documents from sql database (not applicable for the local chains)
        for sql_chain in self.sql_db_chains:
            documents.append(
                Document(
                    page_content=f"Query: {query}  Answer: {sql_chain.run(query)}", metadata={"source": "sql_query"}
                )
            )
        if len(self.buffer) < self.buffer_size:
            self.buffer.append([query_, documents])
        else:
            self.buffer.pop(0)
            self.buffer.append([query_, documents])
        logger.info(f"retrieved documents:\n{str(documents)}", verbose=self.verbose)
        return documents
