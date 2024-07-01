"""
Script for English retrieval based conversation system backed by LLaMa2
"""

import argparse
import os

from colossalqa.chain.retrieval_qa.base import RetrievalQA
from colossalqa.data_loader.document_loader import DocumentLoader
from colossalqa.local.llm import ColossalAPI, ColossalLLM
from colossalqa.prompt.prompt import PROMPT_RETRIEVAL_CLASSIFICATION_USE_CASE_ZH
from colossalqa.retriever import CustomRetriever
from colossalqa.text_splitter import ChineseTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="English retrieval based conversation system backed by LLaMa2")
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

    # Define the retriever
    information_retriever = CustomRetriever(k=2, sql_file_path=args.sql_file_path, verbose=True)

    # Setup embedding model locally
    embedding = HuggingFaceEmbeddings(
        model_name="moka-ai/m3e-base", model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": False}
    )

    # Load data to vector store
    print("Select files for constructing retriever")
    documents = []

    # define metadata function which is used to format the prompt with value in metadata instead of key,
    # the later is langchain's default behavior
    def metadata_func(data_sample, additional_fields):
        """
        metadata_func (Callable[Dict, Dict]): A function that takes in the JSON
                object extracted by the jq_schema and the default metadata and returns
                a dict of the updated metadata.

        To use key-value format, the metadata_func should be defined as follows:
            metadata = {'value': 'a string to be used to format the prompt', 'is_key_value_mapping': True}
        """
        metadata = {}
        metadata["value"] = f"Question: {data_sample['key']}\nAnswer:{data_sample['value']}"
        metadata["is_key_value_mapping"] = True
        assert "value" not in additional_fields
        assert "is_key_value_mapping" not in additional_fields
        metadata.update(additional_fields)
        return metadata

    retriever_data = DocumentLoader(
        [["../data/data_sample/custom_service_classification.json", "CustomerServiceDemo"]],
        content_key="key",
        metadata_func=metadata_func,
    ).all_data

    # Split
    text_splitter = ChineseTextSplitter()
    splits = text_splitter.split_documents(retriever_data)
    documents.extend(splits)

    # Create retriever
    information_retriever.add_documents(docs=documents, cleanup="incremental", mode="by_source", embedding=embedding)

    # Define retrieval chain
    retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm,
        verbose=True,
        chain_type="stuff",
        retriever=information_retriever,
        chain_type_kwargs={"prompt": PROMPT_RETRIEVAL_CLASSIFICATION_USE_CASE_ZH},
        llm_kwargs={"max_new_tokens": 50, "temperature": 0.75, "do_sample": True},
    )
    # Set disambiguity handler

    # Start conversation
    while True:
        user_input = input("User: ")
        if "END" == user_input:
            print("Agent: Happy to chat with you ：)")
            break
        # 要使用和custom_service_classification.json 里的key 类似的句子做输入
        agent_response = retrieval_chain.run(query=user_input, stop=["Human: "])
        agent_response = agent_response.split("\n")[0]
        print(f"Agent: {agent_response}")
