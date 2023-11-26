from typing import Dict, Tuple

from colossalqa.chain.retrieval_qa.base import RetrievalQA
from colossalqa.data_loader.document_loader import DocumentLoader
from colossalqa.memory import ConversationBufferWithSummary
from colossalqa.mylogging import get_logger
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

logger = get_logger()

DEFAULT_RAG_CFG = {
    "retri_top_k": 3,
    "retri_kb_file_path": "./",
    "verbose": True,
    "mem_summary_prompt": SUMMARY_PROMPT_ZH,
    "mem_human_prefix": "用户",
    "mem_ai_prefix": "Assistant",
    "mem_max_tokens": 2000,
    "mem_llm_kwargs": {"max_new_tokens": 50, "temperature": 1, "do_sample": True},
    "disambig_prompt": PROMPT_DISAMBIGUATE_ZH,
    "disambig_llm_kwargs": {"max_new_tokens": 30, "temperature": 1, "do_sample": True},
    "embed_model_name_or_path": "moka-ai/m3e-base",
    "embed_model_device": {"device": "cpu"},
    "gen_llm_kwargs": {"max_new_tokens": 100, "temperature": 1, "do_sample": True},
    "gen_qa_prompt": PROMPT_RETRIEVAL_QA_ZH,
}


class RAG_ChatBot:
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
            model_name=kwargs["embed_model_name_or_path"],
            model_kwargs=kwargs["embed_model_device"],
            encode_kwargs={"normalize_embeddings": False},
        )

    def set_text_splitter(self, **kwargs):
        # Initialize text_splitter
        self.text_splitter = ChineseTextSplitter()

    def set_memory(self, **kwargs):
        params = {"llm_kwargs": kwargs["mem_llm_kwargs"]} if kwargs.get("mem_llm_kwargs", None) else {}
        # Initialize memory with summarization ability
        self.memory = ConversationBufferWithSummary(
            llm=self.llm,
            prompt=kwargs["mem_summary_prompt"],
            human_prefix=kwargs["mem_human_prefix"],
            ai_prefix=kwargs["mem_ai_prefix"],
            max_tokens=kwargs["mem_max_tokens"],
            **params,
        )

    def set_info_retriever(self, **kwargs):
        self.info_retriever = CustomRetriever(
            k=kwargs["retri_top_k"], sql_file_path=kwargs["retri_kb_file_path"], verbose=kwargs["verbose"]
        )

    def set_rag_chain(self, **kwargs):
        params = {"llm_kwargs": kwargs["gen_llm_kwargs"]} if kwargs.get("gen_llm_kwargs", None) else {}
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            verbose=kwargs["verbose"],
            chain_type="stuff",
            retriever=self.info_retriever,
            chain_type_kwargs={"prompt": kwargs["gen_qa_prompt"], "memory": self.memory},
            **params,
        )

    def split_docs(self, documents):
        doc_splits = self.text_splitter.split_documents(documents)
        return doc_splits

    def set_disambig_retriv(self, **kwargs):
        params = {"llm_kwargs": kwargs["disambig_llm_kwargs"]} if kwargs.get("disambig_llm_kwargs", None) else {}
        self.llm_chain_disambiguate = LLMChain(llm=self.llm, prompt=kwargs["disambig_prompt"], **params)

        def disambiguity(input: str):
            out = self.llm_chain_disambiguate.run(input=input, chat_history=self.memory.buffer, stop=["\n"])
            return out.split("\n")[0]

        self.info_retriever.set_rephrase_handler(disambiguity)

    def load_doc_from_console(self, json_parse_args: Dict = {}):
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

    def load_doc_from_files(self, files, data_name="default_kb", json_parse_args: Dict = {}):
        documents = []
        for file in files:
            docs = DocumentLoader([[file, data_name.replace(" ", "_")]], **json_parse_args).all_data
            documents.extend(docs)
        self.documents = documents
        self.split_docs_and_add_to_mem(**self.rag_config)

    def split_docs_and_add_to_mem(self, **kwargs):
        self.doc_splits = self.split_docs(self.documents)
        self.info_retriever.add_documents(
            docs=self.doc_splits, cleanup="incremental", mode="by_source", embedding=self.embed_model
        )
        self.memory.initiate_document_retrieval_chain(self.llm, kwargs["gen_qa_prompt"], self.info_retriever)

    def reset_config(self, rag_config):
        self.rag_config = rag_config
        self.set_embed_model(**self.rag_config)
        self.set_text_splitter(**self.rag_config)
        self.set_memory(**self.rag_config)
        self.set_info_retriever(**self.rag_config)
        self.set_rag_chain(**self.rag_config)
        if self.rag_config.get("disambig_prompt", None):
            self.set_disambig_retriv(**self.rag_config)

    def run(self, user_input: str, memory: ConversationBufferWithSummary) -> Tuple[str, ConversationBufferWithSummary]:
        if memory:
            memory.buffered_history.messages = memory.buffered_history.messages
            memory.summarized_history_temp.messages = memory.summarized_history_temp.messages
        result = self.rag_chain.run(
            query=user_input,
            stop=[memory.human_prefix + ": "],
            rejection_trigger_keywrods=ZH_RETRIEVAL_QA_TRIGGER_KEYWORDS,
            rejection_answer=ZH_RETRIEVAL_QA_REJECTION_ANSWER,
        )
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
    # Initialize an Langchain LLM(here we use ChatGPT as an example)
    from langchain.llms import OpenAI

    llm = OpenAI(openai_api_key="YOUR_OPENAI_API_KEY")

    # chatgpt cannot control temperature, do_sample, etc.
    DEFAULT_RAG_CFG["mem_llm_kwargs"] = None
    DEFAULT_RAG_CFG["disambig_llm_kwargs"] = None
    DEFAULT_RAG_CFG["gen_llm_kwargs"] = None

    rag = RAG_ChatBot(llm, DEFAULT_RAG_CFG)
    rag.load_doc_from_console()
    rag.start_test_session()
