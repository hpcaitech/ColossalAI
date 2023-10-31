import argparse
import os

from colossalqa.chain.retrieval_qa.base import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from colossalqa.mylogging import get_logger
from colossalqa.text_splitter import ChineseTextSplitter
from colossalqa.data_loader.document_loader import DocumentLoader
from colossalqa.retriever import CustomRetriever
from typing import Any, List, Optional, Dict, Tuple
from colossalqa.local.llm import ColossalAPI, ColossalLLM
from colossalqa.memory import ConversationBufferWithSummary
from colossalqa.prompt.prompt import PROMPT_DISAMBIGUATE_ZH, PROMPT_RETRIEVAL_QA_ZH, SUMMARY_PROMPT_ZH
from langchain import LLMChain
from pangu_llm import Pangu

logger = get_logger()

RAG_CFG = {
    "retri_top_k": 3,
    "retri_kb_file_path": "./",
    "verbose": True,
    "mem_summary_prompt": SUMMARY_PROMPT_ZH,
    "mem_human_prefix": "用户",
    "mem_ai_prefix": "AI",
    "mem_max_tokens": 2000,
    "mem_llm_kwargs": {"max_new_tokens": 50, "temperature": 0.6, "do_sample": True},
    "disambig_prompt": PROMPT_DISAMBIGUATE_ZH,
    "disambig_llm_kwargs": {"max_new_tokens": 30, "temperature": 0.6, "do_sample": True},
    "embed_model_name": "moka-ai/m3e-base",
    "embed_model_device": {"device": "cpu"},
    "gen_llm_kwargs": {"max_new_tokens": 50, "temperature": 0.75, "do_sample": True},
    "gen_qa_prompt": PROMPT_RETRIEVAL_QA_ZH
}

class RAG_Conversation:
    def __init__(
        self,
        llm,
        rag_config,
    ) -> None:
        self.llm = llm
        self.rag_config = rag_config
        self.set_embed_model(**self.rag_config)
        self.set_text_splitter(**self.rag_config)
        self.set_memory(**self.rag_config)
        self.set_info_retriever(**self.rag_config)
        self.set_rag_chain(**self.rag_config)
        if self.rag_config.get("disambig_prompt", None):
            self.set_disambig_retriv(**self.rag_config)

    def set_embed_model(self, **kwargs):
        self.embed_model = HuggingFaceEmbeddings(
            model_name=kwargs['embed_model_name'],
            model_kwargs=kwargs['embed_model_device'],
            encode_kwargs={"normalize_embeddings": False}
        )
    
    def set_text_splitter(self, **kwargs):
        self.text_splitter = ChineseTextSplitter()

    def set_memory(self, **kwargs):
        # Initialize memory with summarization ability
        self.memory = ConversationBufferWithSummary(
            llm=self.llm,
            prompt=kwargs["mem_summary_prompt"],
            human_prefix=kwargs["mem_human_prefix"],
            ai_prefix=kwargs["mem_ai_prefix"],
            max_tokens=kwargs["mem_max_tokens"],
            llm_kwargs=kwargs["mem_llm_kwargs"],
        )

    def set_info_retriever(self, **kwargs):
        self.info_retriever = CustomRetriever(k=kwargs["retri_top_k"], 
                                              sql_file_path=kwargs["retri_kb_file_path"], 
                                              verbose=kwargs['verbose'])
    
    def set_rag_chain(self, **kwargs):
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            verbose=kwargs['verbose'],
            chain_type="stuff",
            retriever=self.info_retriever,
            chain_type_kwargs={"prompt": kwargs["gen_qa_prompt"], "memory": self.memory},
            llm_kwargs=kwargs["gen_llm_kwargs"],
        )

    def split_docs(self, documents):
        doc_splits = self.text_splitter.split_documents(documents)
        return doc_splits

    def set_disambig_retriv(self, **kwargs):
        self.llm_chain_disambiguate = LLMChain(
            llm=self.llm,
            prompt=kwargs["disambig_prompt"],
            llm_kwargs=kwargs["disambig_llm_kwargs"]   
        )
        def disambiguity(input: str):
            out = self.llm_chain_disambiguate.run(input=input, chat_history=self.memory.buffer, stop=["\n"])
            return out.split("\n")[0]
        self.info_retriever.set_rephrase_handler(disambiguity)

    def load_doc_from_console(self, json_parse_args: Dict={}):
        documents = []
        print("Select files for constructing Chinese retriever")
        while True:
            file = input("Enter a file path or press Enter directly without input to exit:").strip()
            if file == "":
                break
            data_name = input("Enter a short description of the data:")
            docs = DocumentLoader([[file, data_name.replace(" ", "_")]], **json_parse_args).all_data
            documents.extend(docs)
        self.documents = documents
        self.split_docs_and_add_to_mem(**self.rag_config)
    
    def split_docs_and_add_to_mem(self, **kwargs):
        self.doc_splits = self.split_docs(self.documents)
        self.info_retriever.add_documents(docs=self.doc_splits, cleanup="incremental",
                                          mode="by_source", embedding=self.embed_model)
        self.memory.initiate_document_retrieval_chain(self.llm, 
                                                      kwargs["gen_qa_prompt"],
                                                      self.info_retriever)
    
    def run(self, user_input: str, memory: ConversationBufferWithSummary) -> Tuple[str, ConversationBufferWithSummary]:
        if memory:
            memory.buffered_history.messages = memory.buffered_history.messages
            memory.summarized_history_temp.messages = memory.summarized_history_temp.messages
        result = self.rag_chain.run(query=user_input, stop=[memory.human_prefix + ": "])
        return result.split("\n")[0], memory
    
    def start_test_session(self):
        """
        Simple session for testing purpose
        """
        while True:
            user_input = input("User: ")
            if "END" == user_input:
                print("Agent: Happy to chat with you :)")
                break
            agent_response, self.memory = self.run(user_input, self.memory)
            print(f"Agent: {agent_response}")

if __name__ == "__main__":
    # Parse arguments
    # parser = argparse.ArgumentParser(description="Chinese retrieval based conversation system backed by ChatGLM2")
    # parser.add_argument("--model_path", type=str, default=None, help="path to the model")
    # parser.add_argument("--model_name", type=str, default=None, help="name of the model")
    # parser.add_argument(
    #     "--sql_file_path", type=str, default=None, help="path to the a empty folder for storing sql files for indexing"
    # )
    
    
    # args = parser.parse_args()
    # if not os.path.exists(args.sql_file_path):
    #     os.makedirs(args.sql_file_path)
    #     RAG_CFG['retri_kb_file_path'] = args.sql_file_path

    os.environ["URL"] = "https://pangu.cn-southwest-2.myhuaweicloud.com/v1/infers/7c996ea5-a08d-4480-9b47-29b71df679ce/v1/ddc28c926972441592db4a9052389ad1/deployments/abd45976-f351-4ebc-af2b-20bb4bbebf48/text/completions"
    os.environ["USERNAME"] = "ZhengZian"
    os.environ["PASSWORD"] = "zheng19991231"
    os.environ["DOMAIN_NAME"] = "luchen1627"

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
    json_parse_args = {"jq_schema": ".data[]", "content_key": "key", "metadata_func": metadata_func}
    
    llm = Pangu(id=1)
    llm.set_auth_config()

    rag = RAG_Conversation(llm, RAG_CFG)
    rag.load_doc_from_console(json_parse_args)
    rag.start_test_session()

    