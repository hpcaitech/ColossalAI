'''
Multilingual retrieval based conversation system
'''

import colossalqa.retrieval_conversation_en as en_utils
import colossalqa.retrieval_conversation_zh as zh_utils
from typing import Dict, Any, List, Tuple
from langchain.vectorstores import Chroma
from colossalqa.data_loader.document_loader import DocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from colossalqa.text_splitter import NeuralTextSplitter
from colossalqa.chain.retrieval_qa.base import RetrievalQA
from colossalqa.memory import ConversationBufferWithSummary
from colossalqa.retriever import CustomRetriever
from langchain.embeddings import HuggingFaceEmbeddings
from colossalqa.utils import detect_lang_naive

class ChineseRetrievalConversation:
    def __init__(self, retriever) -> None:
        '''
        setup retrieval qa chain for Chinese retrieval based QA
        '''
        self.retriever = retriever
        self.retriever.set_rephrase_handler(zh_utils.disambiguity)
        self.memory = zh_utils.memory
        self.memory.initiate_document_retrieval_chain(zh_utils.llm, 
                                    zh_utils.PROMPT_RETRIEVAL_QA_ZH, self.retriever, chain_type_kwargs={'chat_history':'', })
        self.retrieval_chain = RetrievalQA.from_chain_type(llm=zh_utils.llm, verbose=False, chain_type="stuff", 
                retriever=self.retriever, chain_type_kwargs={"prompt": zh_utils.PROMPT_RETRIEVAL_QA_ZH,"memory":self.memory },
                llm_kwargs={'max_new_tokens':50, 'temperature':0.75, 'do_sample':True})
  
    @classmethod
    def from_retriever(cls, retriever) -> "ChineseRetrievalConversation":
        return cls(retriever)

    def run(self, user_input: str, memory: ConversationBufferWithSummary) -> Tuple[str, ConversationBufferWithSummary]:
        if memory:
            # TODO add translation chain here
            self.memory.buffered_history.messages = memory.buffered_history.messages
            self.memory.summarized_history.messages = memory.summarized_history.messages
            self.memory.summarized_history_temp.messages = memory.summarized_history_temp.messages
        return self.retrieval_chain.run(query=user_input, stop=[self.memory.human_prefix+': ']).split('\n')[0], self.memory
    
class EnglishRetrievalConversation:
    def __init__(self, retriever) -> None:
        '''
        setup retrieval qa chain for English retrieval based QA
        '''
        self.retriever = retriever
        self.retriever.set_rephrase_handler(en_utils.disambiguity)
        self.memory = en_utils.memory
        self.memory.initiate_document_retrieval_chain(en_utils.llm, en_utils.PROMPT_RETRIEVAL_QA_EN, self.retriever,
                chain_type_kwargs={'chat_history':'', })
        self.retrieval_chain = RetrievalQA.from_chain_type(llm=en_utils.llm, verbose=False, chain_type="stuff", 
                retriever=self.retriever, chain_type_kwargs={"prompt": en_utils.PROMPT_RETRIEVAL_QA_EN,"memory":self.memory },
                llm_kwargs={'max_new_tokens':50, 'temperature':0.75, 'do_sample':True})
  
    @classmethod
    def from_retriever(cls, retriever) -> "EnglishRetrievalConversation":
        return cls(retriever)

    def run(self, user_input: str, memory: ConversationBufferWithSummary) -> Tuple[str, ConversationBufferWithSummary]:
        if memory:
            # TODO add translation chain here
            self.memory.buffered_history.messages = memory.buffered_history.messages
            self.memory.summarized_history.messages = memory.summarized_history.messages
            self.memory.summarized_history_temp.messages = memory.summarized_history_temp.messages
        return self.retrieval_chain.run(query=user_input, stop=[self.memory.human_prefix+': ']).split('\n')[0], self.memory
    
class UniversalRetrievalConversation:
    def __init__(self, embedding_model_path: str="moka-ai/m3e-base", embedding_model_device: str="cpu", 
                 files_zh:List[List[str]]=None,
                files_en:List[List[str]]=None) -> None:
        '''
        Warpper for multilingual retrieval qa class (Chinese + English)
        Args:
            embedding_model_path: local or huggingface embedding model
            embedding_model_device: 
            files_zh: [[file_path, name_of_file],...] defines the files used as supporting documents for Chinese retrieval QA
            files_en: [[file_path, name_of_file],...] defines the files used as supporting documents for English retrieval QA
        '''
        self.embedding = HuggingFaceEmbeddings(model_name=embedding_model_path,
                           model_kwargs={'device': embedding_model_device},
                           encode_kwargs={'normalize_embeddings': False})

        docs_zh = self.load_supporting_docs(files=files_zh)
        # create vector store for Chinese
        vectordb_zh = Chroma.from_documents(documents=docs_zh, embedding=self.embedding, collection_name='doc_zh')
        print(f"Number of supporting documents: {vectordb_zh._collection.count()}")
        # create retriever for Chinese
        retriever_zh=vectordb_zh.as_retriever(search_kwargs={"k":3})
        self.information_retriever_zh = CustomRetriever(k=3)
        self.information_retriever_zh.set_retriever(retriever=retriever_zh)

        docs_en = self.load_supporting_docs(files=files_en)
        # create vector store for English
        vectordb_en = Chroma.from_documents(documents=docs_en, embedding=self.embedding, collection_name='doc_en')
        print(f"Number of supporting documents: {vectordb_en._collection.count()}")
        # create retriever for English
        retriever_en=vectordb_en.as_retriever(search_kwargs={"k":3})
        self.information_retriever_en = CustomRetriever(k=3)
        self.information_retriever_en.set_retriever(retriever=retriever_en)

        self.chinese_retrieval_conversation = ChineseRetrievalConversation.from_retriever(self.information_retriever_zh)
        self.english_retrieval_conversation = EnglishRetrievalConversation.from_retriever(self.information_retriever_en)
        self.memory = None

    def load_supporting_docs(self, files:List[List[str]]=None):
        '''
        load supporting documents, currently, all documents will be stored in one vector store
        '''
        documents = []
        if files:
            for file in files:
                retriever_data = DocumentLoader([file]).all_data
                text_splitter = NeuralTextSplitter()
                splits = text_splitter.split_documents(retriever_data)
                documents.extend(splits)
            # create vector store
            vectordb = Chroma.from_documents(documents=documents, embedding=self.embedding)
            print(f"Number of supporting documents: {vectordb._collection.count()}")
        else:
            while True:
                file = input("Select a file to load or enter Esc to exit:")
                if file=='Esc':
                    break
                data_name = input("Enter a short description of the data:")
                retriever_data = DocumentLoader([[file, data_name.replace(' ', '_')]]).all_data

                # Split
                # text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
                text_splitter = NeuralTextSplitter()
                splits = text_splitter.split_documents(retriever_data)
                documents.extend(splits)
        return documents

    def start_test_session(self):
        '''
        simple multilingual session for testing purpose, with naive language selection mechanism
        '''
        while True:
            user_input = input("User: ")
            print(f"User: {user_input}")
            lang = detect_lang_naive(user_input)
            if 'END' == user_input:
                print("Agent: Happy to chat with you ：)")
                break    
            if "zh"==lang:
                agent_response, self.memory = self.chinese_retrieval_conversation.run(user_input, self.memory)
            else:
                agent_response, self.memory = self.english_retrieval_conversation.run(user_input, self.memory)
            agent_response = agent_response.split('\n')[0]
            print(f"Agent: {agent_response}")

    def run(self, user_input: str, which_language = str):
        '''
        Generate the response given the user input and a str indicates the language requirement of the output string
        '''
        assert which_language in ['zh', 'en']
        if which_language == 'zh':
            agent_response, self.memory = self.chinese_retrieval_conversation.run(user_input, self.memory)
        else:
            agent_response, self.memory = self.english_retrieval_conversation.run(user_input, self.memory)
        return agent_response.split('\n')[0]
    
if __name__ == '__main__':
    session = UniversalRetrievalConversation()
    session.start_test_session()
        