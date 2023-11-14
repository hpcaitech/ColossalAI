"""
Script for English retrieval based conversation system backed by LLaMa2
"""
import argparse
import json
import os

from colossalqa.chain.retrieval_qa.base import RetrievalQA
from colossalqa.data_loader.document_loader import DocumentLoader
from colossalqa.local.llm import ColossalAPI, ColossalLLM
from colossalqa.memory import ConversationBufferWithSummary
from colossalqa.prompt.prompt import (
    EN_RETRIEVAL_QA_REJECTION_ANSWER,
    EN_RETRIEVAL_QA_TRIGGER_KEYWORDS,
    PROMPT_DISAMBIGUATE_EN,
    PROMPT_RETRIEVAL_QA_EN,
)
from colossalqa.retriever import CustomRetriever
from langchain import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
    information_retriever = CustomRetriever(k=3, sql_file_path=args.sql_file_path, verbose=True)

    # Setup embedding model locally
    embedding = HuggingFaceEmbeddings(
        model_name="moka-ai/m3e-base", model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": False}
    )

    # Define memory with summarization ability
    memory = ConversationBufferWithSummary(
        llm=llm, max_tokens=2000, llm_kwargs={"max_new_tokens": 50, "temperature": 0.6, "do_sample": True}
    )

    # Define the chain to preprocess the input
    # Disambiguate the input. e.g. "What is the capital of that country?" -> "What is the capital of France?"
    llm_chain_disambiguate = LLMChain(
        llm=llm, prompt=PROMPT_DISAMBIGUATE_EN, llm_kwargs={"max_new_tokens": 30, "temperature": 0.6, "do_sample": True}
    )

    def disambiguity(input):
        out = llm_chain_disambiguate.run(input=input, chat_history=memory.buffer, stop=["\n"])
        return out.split("\n")[0]

    # Load data to vector store
    print("Select files for constructing retriever")
    documents = []

    # preprocess data
    if not os.path.exists("../data/data_sample/custom_service_preprocessed.json"):
        if not os.path.exists("../data/data_sample/custom_service.json"):
            raise ValueError(
                "custom_service.json not found, please download the data from HuggingFace Datasets: qgyd2021/e_commerce_customer_service"
            )
        data = json.load(open("../data/data_sample/custom_service.json", "r", encoding="utf8"))
        preprocessed = []
        for row in data["rows"]:
            preprocessed.append({"key": row["row"]["query"], "value": row["row"]["response"]})
        data = {}
        data["data"] = preprocessed
        with open("../data/data_sample/custom_service_preprocessed.json", "w", encoding="utf8") as f:
            json.dump(data, f, ensure_ascii=False)

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
        [["../data/data_sample/custom_service_preprocessed.json", "CustomerServiceDemo"]],
        content_key="key",
        metadata_func=metadata_func,
    ).all_data

    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    splits = text_splitter.split_documents(retriever_data)
    documents.extend(splits)

    # Create retriever
    information_retriever.add_documents(docs=documents, cleanup="incremental", mode="by_source", embedding=embedding)

    # Set document retrieval chain, we need this chain to calculate prompt length
    memory.initiate_document_retrieval_chain(
        llm,
        PROMPT_RETRIEVAL_QA_EN,
        information_retriever,
        chain_type_kwargs={
            "chat_history": "",
        },
    )

    # Define retrieval chain
    retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm,
        verbose=False,
        chain_type="stuff",
        retriever=information_retriever,
        chain_type_kwargs={"prompt": PROMPT_RETRIEVAL_QA_EN, "memory": memory},
        llm_kwargs={"max_new_tokens": 50, "temperature": 0.75, "do_sample": True},
    )
    # Set disambiguity handler
    information_retriever.set_rephrase_handler(disambiguity)
    # Start conversation
    while True:
        user_input = input("User: ")
        if "END" == user_input:
            print("Agent: Happy to chat with you ：)")
            break
        agent_response = retrieval_chain.run(
            query=user_input,
            stop=["Human: "],
            rejection_trigger_keywrods=EN_RETRIEVAL_QA_TRIGGER_KEYWORDS,
            rejection_answer=EN_RETRIEVAL_QA_REJECTION_ANSWER,
        )
        agent_response = agent_response.split("\n")[0]
        print(f"Agent: {agent_response}")
