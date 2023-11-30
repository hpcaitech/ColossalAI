"""
Custom SummarizerMixin base class and ConversationSummaryMemory class

Modified from Original Source

This code is based on LangChain Ai's langchain, which can be found at
https://github.com/langchain-ai/langchain
The original code is licensed under the MIT license.
"""

from __future__ import annotations

from typing import Any, Dict, List, Type

from langchain.chains.llm import LLMChain
from langchain.memory.chat_memory import BaseChatMemory
from langchain.memory.prompt import SUMMARY_PROMPT
from langchain.pydantic_v1 import BaseModel, root_validator
from langchain.schema import BaseChatMessageHistory, BasePromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import BaseMessage, SystemMessage, get_buffer_string


class SummarizerMixin(BaseModel):
    """
    Mixin for summarizer.
    """

    human_prefix: str = "Human"
    ai_prefix: str = "Assistant"
    llm: BaseLanguageModel
    prompt: BasePromptTemplate = SUMMARY_PROMPT
    summary_message_cls: Type[BaseMessage] = SystemMessage
    llm_kwargs: Dict = {}

    def predict_new_summary(self, messages: List[BaseMessage], existing_summary: str, stop: List = []) -> str:
        """
        Recursively summarize a conversation by generating a new summary using
        the last round of conversation and the existing summary.
        """
        new_lines = get_buffer_string(
            messages,
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )

        chain = LLMChain(llm=self.llm, prompt=self.prompt, llm_kwargs=self.llm_kwargs)
        return chain.predict(summary=existing_summary, new_lines=new_lines, stop=stop)


class ConversationSummaryMemory(BaseChatMemory, SummarizerMixin):
    """Conversation summarizer to chat memory."""

    buffer: str = ""
    memory_key: str = "history"

    @classmethod
    def from_messages(
        cls,
        llm: BaseLanguageModel,
        chat_memory: BaseChatMessageHistory,
        summarize_step: int = 2,
        **kwargs: Any,
    ) -> ConversationSummaryMemory:
        obj = cls(llm=llm, chat_memory=chat_memory, **kwargs)
        for i in range(0, len(obj.chat_memory.messages), summarize_step):
            obj.buffer = obj.predict_new_summary(obj.chat_memory.messages[i : i + summarize_step], obj.buffer)
        return obj

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables."""
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history buffer."""
        if self.return_messages:
            buffer: Any = [self.summary_message_cls(content=self.buffer)]
        else:
            buffer = self.buffer
        return {self.memory_key: buffer}

    @root_validator()
    def validate_prompt_input_variables(cls, values: Dict) -> Dict:
        """Validate that prompt input variables are consistent."""
        prompt_variables = values["prompt"].input_variables
        expected_keys = {"summary", "new_lines"}
        if expected_keys != set(prompt_variables):
            raise ValueError(
                "Got unexpected prompt input variables. The prompt expects "
                f"{prompt_variables}, but it should have {expected_keys}."
            )
        return values

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        super().save_context(inputs, outputs)
        self.buffer = self.predict_new_summary(self.chat_memory.messages[-2:], self.buffer)

    def clear(self) -> None:
        """Clear memory contents."""
        super().clear()
        self.buffer = ""
