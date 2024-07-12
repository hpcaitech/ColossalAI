"""
Multilingual retrieval based conversation system
"""

from typing import List

from colossalqa.data_loader.document_loader import DocumentLoader
from colossalqa.mylogging import get_logger
from colossalqa.retrieval_conversation_en import EnglishRetrievalConversation
from colossalqa.retrieval_conversation_zh import ChineseRetrievalConversation
from colossalqa.retriever import CustomRetriever
from colossalqa.text_splitter import ChineseTextSplitter
from colossalqa.utils import detect_lang_naive
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter

logger = get_logger()


class UniversalRetrievalConversation:
    """
    Wrapper class for bilingual retrieval conversation system
    """

    def __init__(
        self,
        embedding_model_path: str = "moka-ai/m3e-base",
        embedding_model_device: str = "cpu",
        zh_model_path: str = None,
        zh_model_name: str = None,
        en_model_path: str = None,
        en_model_name: str = None,
        sql_file_path: str = None,
        files_zh: List[List[str]] = None,
        files_en: List[List[str]] = None,
        text_splitter_chunk_size=100,
        text_splitter_chunk_overlap=10,
    ) -> None:
        """
        Wrapper for multilingual retrieval qa class (Chinese + English)
        Args:
            embedding_model_path: local or huggingface embedding model
            embedding_model_device:
            files_zh: [[file_path, name_of_file, separator],...] defines the files used as supporting documents for Chinese retrieval QA
            files_en: [[file_path, name_of_file, separator],...] defines the files used as supporting documents for English retrieval QA
        """
        self.embedding = HuggingFaceEmbeddings(
            model_name=embedding_model_path,
            model_kwargs={"device": embedding_model_device},
            encode_kwargs={"normalize_embeddings": False},
        )
        print("Select files for constructing Chinese retriever")
        docs_zh = self.load_supporting_docs(
            files=files_zh,
            text_splitter=ChineseTextSplitter(
                chunk_size=text_splitter_chunk_size, chunk_overlap=text_splitter_chunk_overlap
            ),
        )
        # Create retriever
        self.information_retriever_zh = CustomRetriever(
            k=3, sql_file_path=sql_file_path.replace(".db", "_zh.db"), verbose=True
        )
        self.information_retriever_zh.add_documents(
            docs=docs_zh, cleanup="incremental", mode="by_source", embedding=self.embedding
        )

        print("Select files for constructing English retriever")
        docs_en = self.load_supporting_docs(
            files=files_en,
            text_splitter=RecursiveCharacterTextSplitter(
                chunk_size=text_splitter_chunk_size, chunk_overlap=text_splitter_chunk_overlap
            ),
        )
        # Create retriever
        self.information_retriever_en = CustomRetriever(
            k=3, sql_file_path=sql_file_path.replace(".db", "_en.db"), verbose=True
        )
        self.information_retriever_en.add_documents(
            docs=docs_en, cleanup="incremental", mode="by_source", embedding=self.embedding
        )

        self.chinese_retrieval_conversation = ChineseRetrievalConversation.from_retriever(
            self.information_retriever_zh, model_path=zh_model_path, model_name=zh_model_name
        )
        self.english_retrieval_conversation = EnglishRetrievalConversation.from_retriever(
            self.information_retriever_en, model_path=en_model_path, model_name=en_model_name
        )
        self.memory = None

    def load_supporting_docs(self, files: List[List[str]] = None, text_splitter: TextSplitter = None):
        """
        Load supporting documents, currently, all documents will be stored in one vector store
        """
        documents = []
        if files:
            for file in files:
                retriever_data = DocumentLoader([[file["data_path"], file["name"]]]).all_data
                splits = text_splitter.split_documents(retriever_data)
                documents.extend(splits)
        else:
            while True:
                file = input("Select a file to load or press Enter to exit:")
                if file == "":
                    break
                data_name = input("Enter a short description of the data:")
                separator = input(
                    "Enter a separator to force separating text into chunks, if no separator is given, the default separator is '\\n\\n', press ENTER directly to skip:"
                )
                separator = separator if separator != "" else "\n\n"
                retriever_data = DocumentLoader([[file, data_name.replace(" ", "_")]]).all_data

                # Split
                splits = text_splitter.split_documents(retriever_data)
                documents.extend(splits)
        return documents

    def start_test_session(self):
        """
        Simple multilingual session for testing purpose, with naive language selection mechanism
        """
        while True:
            user_input = input("User: ")
            lang = detect_lang_naive(user_input)
            if "END" == user_input:
                print("Agent: Happy to chat with you ：)")
                break
            agent_response = self.run(user_input, which_language=lang)
            print(f"Agent: {agent_response}")

    def run(self, user_input: str, which_language=str):
        """
        Generate the response given the user input and a str indicates the language requirement of the output string
        """
        assert which_language in ["zh", "en"]
        if which_language == "zh":
            agent_response, self.memory = self.chinese_retrieval_conversation.run(user_input, self.memory)
        else:
            agent_response, self.memory = self.english_retrieval_conversation.run(user_input, self.memory)
        return agent_response.split("\n")[0]
