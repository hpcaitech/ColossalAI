from langchain.schema.retriever import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from typing import List, Callable, Dict, Any

class CustomRetriever(BaseRetriever):
    retriever: BaseRetriever = None
    sql_db_chains = []
    k = 3
    rephrase_handler:Callable = None
    buffer: Dict = []
    buffer_size: int = 5

    def set_retriever(self, retriever:BaseRetriever):
        '''update retriever. Useful when you want to change the supporting documents'''
        self.retriever = retriever

    def set_k(self, k:int=3):
        '''update k, k is the number of document to retrieve'''
        self.k = k

    def set_sql_database_chain(self, db_chains):
        '''
        set sql agent chain to retrieve information from sql database
        Not used in this version
        '''
        self.sql_db_chains = db_chains

    def set_rephrase_handler(self, handler:Callable=None):
        '''
        Set a handler to preprocess the input str before feed into the retriever
        '''
        self.rephrase_handler = handler

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun=None
    ) -> List[Document]:
        '''
        This function is called by the retriever to get the relevant documents.
        recent vistied queries are stored in buffer, if the query is in buffer, return the documents directly
        
        Args:
            query: the query to be searched
            run_manager: the callback manager for retriever run
        Returns:
            documents: the relevant documents
        '''
        for buffered_doc in self.buffer:
            if buffered_doc[0] == query:
                return buffered_doc[1]
        query_ = str(query)
        # Use your existing retriever to get the documents
        if self.rephrase_handler:
            query = self.rephrase_handler(query)
        documents = []
        documents.extend(self.retriever.get_relevant_documents(query, callbacks=run_manager.get_child() if run_manager else None))
        for sql_chain in self.sql_db_chains:
            documents.append(Document(page_content = f"Query: {query}  Answer: {sql_chain.run(query)}", metadata={"source": "sql_query"}))
        if len(self.buffer)<self.buffer_size:
            self.buffer.append([query_, documents])
        else:
            self.buffer.pop(0)
            self.buffer.append([query_, documents])
        print("retrieved documents:")
        print(documents)  
        return documents