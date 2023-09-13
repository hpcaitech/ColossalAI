import pytest
import os
from colossalqa.memory import ConversationBufferWithSummary
from colossalqa.local.llm import CoatiLLM, CoatiAPI
from colossalqa.data_loader.document_loader import DocumentLoader
from colossalqa.text_splitter import NeuralTextSplitter
from langchain.vectorstores import Chroma
from colossalqa.retriever import CustomRetriever
from langchain.embeddings import HuggingFaceEmbeddings
from colossalqa.prompt.prompt import PROMPT_RETRIEVAL_QA_ZH
from langchain.chains import RetrievalQA

def test_memory_long():
    model_path = os.environ.get('LLAMA2_PATH')
    data_path = os.environ.get('TEST_DATA_PATH_EN')
    coati_api = CoatiAPI('llama', model_path)
    llm = CoatiLLM(n=4, api=coati_api)
    memory = ConversationBufferWithSummary(llm=llm,
        llm_kwargs={'max_new_tokens':50, 'temperature':0.6, 'do_sample':True})
    retriever_data = DocumentLoader([[data_path, 'company information']]).all_data

    # Split
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    text_splitter = NeuralTextSplitter()
    splits = text_splitter.split_documents(retriever_data)

    embedding =  HuggingFaceEmbeddings(model_name="moka-ai/m3e-base",
                           model_kwargs={'device': 'cpu'},encode_kwargs={'normalize_embeddings': False})

    # create vector store
    vectordb = Chroma.from_documents(documents=splits, embedding=embedding)

    # create retriever    
    retriever=vectordb.as_retriever(search_kwargs={"k":3})
    information_retriever = CustomRetriever()
    information_retriever.set_retriever(retriever=retriever)
    information_retriever.set_k(k=3)

    memory.initiate_document_retrieval_chain(llm, PROMPT_RETRIEVAL_QA_ZH, information_retriever, 
        chain_type_kwargs={'chat_history':'', })
    
    # this keep the prompt length excluding dialogues the same
    docs = information_retriever._get_relevant_documents("this is a test input.")
    prompt_length = memory.chain.prompt_length(docs, **{'question':"this is a test input.", 'chat_history':""})
    remain = 600 - prompt_length
    have_summarization_flag = False
    for i in range(40):
        chat_history = memory.load_memory_variables({'question':"this is a test input.", "input_documents":docs})['chat_history']
        assert memory.get_conversation_length()<=remain
        memory.save_context({'question':"this is a test input."}, {"output":"this is a test output."})
        if "A summarization of historical conversation:" in chat_history:
            have_summarization_flag = True
    assert have_summarization_flag==True

def test_memory_short():
    model_path = os.environ.get('LLAMA2_PATH')
    data_path = os.environ.get('TEST_DATA_PATH_EN')
    coati_api = CoatiAPI('llama', model_path)
    llm = CoatiLLM(n=4, api=coati_api)
    memory = ConversationBufferWithSummary(llm=llm,
        llm_kwargs={'max_new_tokens':50, 'temperature':0.6, 'do_sample':True})
    retriever_data = DocumentLoader([[data_path, 'company information']]).all_data

    # Split
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    text_splitter = NeuralTextSplitter()
    splits = text_splitter.split_documents(retriever_data)

    embedding =  HuggingFaceEmbeddings(model_name="moka-ai/m3e-base",
                           model_kwargs={'device': 'cpu'},encode_kwargs={'normalize_embeddings': False})

    # create vector store
    vectordb = Chroma.from_documents(documents=splits, embedding=embedding)

    # create retriever    
    retriever=vectordb.as_retriever(search_kwargs={"k":3})
    information_retriever = CustomRetriever()
    information_retriever.set_retriever(retriever=retriever)
    information_retriever.set_k(k=3)

    memory.initiate_document_retrieval_chain(llm, PROMPT_RETRIEVAL_QA_ZH, information_retriever, 
        chain_type_kwargs={'chat_history':'', })
    
    # this keep the prompt length excluding dialogues the same
    docs = information_retriever._get_relevant_documents("this is a test input.")

    for i in range(4):
        chat_history = memory.load_memory_variables({'question':"this is a test input.", "input_documents":docs})['chat_history']
        assert chat_history.count('AI: this is a test output.')==i
        assert chat_history.count("Human: this is a test input.")==i
        memory.save_context({'question':"this is a test input."}, {"output":"this is a test output."})

if __name__=="__main__":
    test_memory_short()
    test_memory_long()