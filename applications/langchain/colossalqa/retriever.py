from langchain.schema.retriever import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from typing import List, Callable, Dict, Any, Union

class CustomRetriever(BaseRetriever):
    retriever: List[BaseRetriever] = None
    sql_db_chains = []
    k = 3
    rephrase_handler:Callable = None
    buffer: Dict = []
    buffer_size: int = 5

    def set_retriever(self, retriever:Union[BaseRetriever, List[BaseRetriever]]):
        '''update retriever. Useful when you want to change the supporting documents'''
        self.retriever = [retriever] if isinstance(retriever, BaseRetriever) else retriever

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
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun=None, 
        score_threshold: float = None
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
        for retriever in self.retriever:
            # retrieve documents from each retriever
            k = retriever.search_kwargs['k'] if 'k' in retriever.search_kwargs else self.k
            documents.extend(retriever.vectorstore.similarity_search_with_relevance_scores(query, k, score_threshold=score_threshold))
        # return the top k documents among all retrievers
        documents = sorted(documents, key=lambda x: x[1], reverse=True)[:self.k]
        documents = [doc[0] for doc in documents]
        # retrieve documents from sql database (not applicable for the local chains)
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