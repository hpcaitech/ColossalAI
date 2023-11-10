"""
Script for Chinese retrieval based conversation system backed by ChatGLM
"""
import argparse
import os

from colossalqa.chain.retrieval_qa.base import RetrievalQA
from colossalqa.data_loader.document_loader import DocumentLoader
from colossalqa.local.llm import ColossalAPI, ColossalLLM
from colossalqa.memory import ConversationBufferWithSummary
from colossalqa.prompt.prompt import (
    PROMPT_DISAMBIGUATE_ZH,
    PROMPT_RETRIEVAL_QA_ZH,
    SUMMARY_PROMPT_ZH,
    ZH_RETRIEVAL_QA_REJECTION_ANSWER,
    ZH_RETRIEVAL_QA_TRIGGER_KEYWORDS,
)
from colossalqa.retriever import CustomRetriever
from colossalqa.text_splitter import ChineseTextSplitter
from langchain import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Chinese retrieval based conversation system backed by ChatGLM2")
    parser.add_argument("--model_path", type=str, default=None, help="path to the model")
    parser.add_argument("--model_name", type=str, default=None, help="name of the model")
    parser.add_argument(
        "--sql_file_path", type=str, default=None, help="path to the a empty folder for storing sql files for indexing"
    )

    args = parser.parse_args()

    if not os.path.exists(args.sql_file_path):
        os.makedirs(args.sql_file_path)

    colossal_api = ColossalAPI.get_api(args.model_name, args.model_path)
    llm = ColossalLLM(n=1, api=colossal_api)

    # Setup embedding model locally
    embedding = HuggingFaceEmbeddings(
        model_name="moka-ai/m3e-base", model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": False}
    )
    # Define the retriever
    information_retriever = CustomRetriever(k=3, sql_file_path=args.sql_file_path, verbose=True)

    # Define memory with summarization ability
    memory = ConversationBufferWithSummary(
        llm=llm,
        prompt=SUMMARY_PROMPT_ZH,
        human_prefix="用户",
        ai_prefix="Assistant",
        max_tokens=2000,
        llm_kwargs={"max_new_tokens": 50, "temperature": 0.6, "do_sample": True},
    )

    # Define the chain to preprocess the input
    # Disambiguate the input. e.g. "What is the capital of that country?" -> "What is the capital of France?"
    llm_chain_disambiguate = LLMChain(
        llm=llm, prompt=PROMPT_DISAMBIGUATE_ZH, llm_kwargs={"max_new_tokens": 30, "temperature": 0.6, "do_sample": True}
    )

    def disambiguity(input: str):
        out = llm_chain_disambiguate.run(input=input, chat_history=memory.buffer, stop=["\n"])
        return out.split("\n")[0]

    # Load data to vector store
    print("Select files for constructing retriever")
    documents = []
    while True:
        file = input("Enter a file path or press Enter directory without input to exit:").strip()
        if file == "":
            break
        data_name = input("Enter a short description of the data:")
        retriever_data = DocumentLoader([[file, data_name.replace(" ", "_")]]).all_data

        # Split
        text_splitter = ChineseTextSplitter()
        splits = text_splitter.split_documents(retriever_data)
        documents.extend(splits)
    # Create retriever
    information_retriever.add_documents(docs=documents, cleanup="incremental", mode="by_source", embedding=embedding)

    # Set document retrieval chain, we need this chain to calculate prompt length
    memory.initiate_document_retrieval_chain(llm, PROMPT_RETRIEVAL_QA_ZH, information_retriever)

    # Define retrieval chain
    llm_chain = RetrievalQA.from_chain_type(
        llm=llm,
        verbose=False,
        chain_type="stuff",
        retriever=information_retriever,
        chain_type_kwargs={"prompt": PROMPT_RETRIEVAL_QA_ZH, "memory": memory},
        llm_kwargs={"max_new_tokens": 150, "temperature": 0.6, "do_sample": True},
    )

    # Set disambiguity handler
    information_retriever.set_rephrase_handler(disambiguity)

    # Start conversation
    while True:
        user_input = input("User: ")
        if "END" == user_input:
            print("Agent: Happy to chat with you ：)")
            break
        agent_response = llm_chain.run(
            query=user_input,
            stop=["</答案>"],
            doc_prefix="支持文档",
            rejection_trigger_keywrods=ZH_RETRIEVAL_QA_TRIGGER_KEYWORDS,
            rejection_answer=ZH_RETRIEVAL_QA_REJECTION_ANSWER,
        )
        print(f"Agent: {agent_response}")
