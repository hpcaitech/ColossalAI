"""
Script for Chinese retrieval based conversation system backed by ChatGLM
"""

from typing import Tuple

from colossalqa.chain.retrieval_qa.base import RetrievalQA
from colossalqa.local.llm import ColossalAPI, ColossalLLM
from colossalqa.memory import ConversationBufferWithSummary
from colossalqa.mylogging import get_logger
from colossalqa.prompt.prompt import PROMPT_DISAMBIGUATE_EN, PROMPT_RETRIEVAL_QA_EN
from colossalqa.retriever import CustomRetriever
from langchain import LLMChain

logger = get_logger()


class EnglishRetrievalConversation:
    """
    Wrapper class for Chinese retrieval conversation system
    """

    def __init__(self, retriever: CustomRetriever, model_path: str, model_name: str) -> None:
        """
        Setup retrieval qa chain for Chinese retrieval based QA
        """
        logger.info(f"model_name: {model_name}; model_path: {model_path}", verbose=True)
        colossal_api = ColossalAPI.get_api(model_name, model_path)
        self.llm = ColossalLLM(n=1, api=colossal_api)

        # Define the retriever
        self.retriever = retriever

        # Define the chain to preprocess the input
        # Disambiguate the input. e.g. "What is the capital of that country?" -> "What is the capital of France?"
        # Prompt is summarization prompt
        self.llm_chain_disambiguate = LLMChain(
            llm=self.llm,
            prompt=PROMPT_DISAMBIGUATE_EN,
            llm_kwargs={"max_new_tokens": 30, "temperature": 0.6, "do_sample": True},
        )

        self.retriever.set_rephrase_handler(self.disambiguity)
        # Define memory with summarization ability
        self.memory = ConversationBufferWithSummary(
            llm=self.llm, llm_kwargs={"max_new_tokens": 50, "temperature": 0.6, "do_sample": True}
        )
        self.memory.initiate_document_retrieval_chain(
            self.llm,
            PROMPT_RETRIEVAL_QA_EN,
            self.retriever,
            chain_type_kwargs={
                "chat_history": "",
            },
        )
        self.retrieval_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            verbose=False,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": PROMPT_RETRIEVAL_QA_EN, "memory": self.memory},
            llm_kwargs={"max_new_tokens": 50, "temperature": 0.75, "do_sample": True},
        )

    def disambiguity(self, input: str):
        out = self.llm_chain_disambiguate.run(input=input, chat_history=self.memory.buffer, stop=["\n"])
        return out.split("\n")[0]

    @classmethod
    def from_retriever(
        cls, retriever: CustomRetriever, model_path: str, model_name: str
    ) -> "EnglishRetrievalConversation":
        return cls(retriever, model_path, model_name)

    def run(self, user_input: str, memory: ConversationBufferWithSummary) -> Tuple[str, ConversationBufferWithSummary]:
        if memory:
            # TODO add translation chain here
            self.memory.buffered_history.messages = memory.buffered_history.messages
            self.memory.summarized_history_temp.messages = memory.summarized_history_temp.messages
        return (
            self.retrieval_chain.run(
                query=user_input,
                stop=[self.memory.human_prefix + ": "],
                rejection_trigger_keywords=["cannot answer the question"],
                rejection_answer="Sorry, this question cannot be answered based on the information provided.",
            ).split("\n")[0],
            self.memory,
        )
