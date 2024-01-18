"""
Implement a memory class for storing conversation history
Support long term and short term memory
"""
from typing import Any, Dict, List

from colossalqa.chain.memory.summary import ConversationSummaryMemory
from colossalqa.chain.retrieval_qa.load_chain import load_qa_chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.schema import BaseChatMessageHistory
from langchain.schema.messages import BaseMessage
from langchain.schema.retriever import BaseRetriever
from pydantic import Field


class ConversationBufferWithSummary(ConversationSummaryMemory):
    """Memory class for storing information about entities."""

    # Define dictionary to store information about entities.
    # Store the most recent conversation history
    buffered_history: BaseChatMessageHistory = Field(default_factory=ChatMessageHistory)
    # Temp buffer
    summarized_history_temp: BaseChatMessageHistory = Field(default_factory=ChatMessageHistory)
    human_prefix: str = "Human"
    ai_prefix: str = "Assistant"
    buffer: str = ""  # Formated conversation in str
    existing_summary: str = ""  # Summarization of stale converstion in str
    # Define key to pass information about entities into prompt.
    memory_key: str = "chat_history"
    input_key: str = "question"
    retriever: BaseRetriever = None
    max_tokens: int = 2000
    chain: BaseCombineDocumentsChain = None
    input_chain_type_kwargs: List = {}

    @property
    def buffer(self) -> Any:
        """String buffer of memory."""
        return self.buffer_as_messages if self.return_messages else self.buffer_as_str

    @property
    def buffer_as_str(self) -> str:
        """Exposes the buffer as a string in case return_messages is True."""
        self.buffer = self.format_dialogue()
        return self.buffer

    @property
    def buffer_as_messages(self) -> List[BaseMessage]:
        """Exposes the buffer as a list of messages in case return_messages is False."""
        return self.buffered_history.messages

    def clear(self):
        """Clear all the memory"""
        self.buffered_history.clear()
        self.summarized_history_temp.clear()

    def initiate_document_retrieval_chain(
        self, llm: Any, prompt_template: Any, retriever: Any, chain_type_kwargs: Dict[str, Any] = {}
    ) -> None:
        """
        Since we need to calculate the length of the prompt, we need to initiate a retrieval chain
        to calculate the length of the prompt.
        Args:
            llm: the language model for the retrieval chain (we won't actually return the output)
            prompt_template: the prompt template for constructing the retrieval chain
            retriever: the retriever for the retrieval chain
            max_tokens: the max length of the prompt (not include the output)
            chain_type_kwargs: the kwargs for the retrieval chain
            memory_key: the key for the chat history
            input_key: the key for the input query
        """
        self.retriever = retriever
        input_chain_type_kwargs = {k: v for k, v in chain_type_kwargs.items() if k not in [self.memory_key]}
        self.input_chain_type_kwargs = input_chain_type_kwargs
        self.chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt_template, **self.input_chain_type_kwargs)

    @property
    def memory_variables(self) -> List[str]:
        """Define the variables we are providing to the prompt."""
        return [self.memory_key]

    def format_dialogue(self, lang: str = "en") -> str:
        """Format memory into two parts--- summarization of historical conversation and most recent conversation"""
        if len(self.summarized_history_temp.messages) != 0:
            for i in range(int(len(self.summarized_history_temp.messages) / 2)):
                self.existing_summary = (
                    self.predict_new_summary(
                        self.summarized_history_temp.messages[i * 2 : i * 2 + 2], self.existing_summary, stop=["\n\n"]
                    )
                    .strip()
                    .split("\n")[0]
                    .strip()
                )
            for i in range(int(len(self.summarized_history_temp.messages) / 2)):
                self.summarized_history_temp.messages.pop(0)
                self.summarized_history_temp.messages.pop(0)
        conversation_buffer = []
        for t in self.buffered_history.messages:
            if t.type == "human":
                prefix = self.human_prefix
            else:
                prefix = self.ai_prefix
            conversation_buffer.append(prefix + ": " + t.content)
        conversation_buffer = "\n".join(conversation_buffer)
        if len(self.existing_summary) > 0:
            if lang == "en":
                message = f"A summarization of historical conversation:\n{self.existing_summary}\nMost recent conversation:\n{conversation_buffer}"
            elif lang == "zh":
                message = f"历史对话概要:\n{self.existing_summary}\n最近的对话:\n{conversation_buffer}"
            else:
                raise ValueError("Unsupported language")
            return message
        else:
            message = conversation_buffer
            return message

    def get_conversation_length(self):
        """Get the length of the formatted conversation"""
        prompt = self.format_dialogue()
        length = self.llm.get_num_tokens(prompt)
        return length

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Load the memory variables.
        Summarize oversize conversation to fit into the length constraint defined by max_tokene
        Args:
            inputs: the kwargs of the chain of your definition
        Returns:
            a dict that maps from memory key to the formated dialogue
            the formated dialogue has the following format
            if conversation is too long:
                A summarization of historical conversation:
                {summarization}
                Most recent conversation:
                Human: XXX
                Assistant: XXX
                ...
            otherwise
                Human: XXX
                Assistant: XXX
                ...
        """
        # Calculate remain length
        if "input_documents" in inputs:
            # Run in a retrieval qa chain
            docs = inputs["input_documents"]
        else:
            # For test
            docs = self.retriever.get_relevant_documents(inputs[self.input_key])
        inputs[self.memory_key] = ""
        inputs = {k: v for k, v in inputs.items() if k in [self.chain.input_key, self.input_key, self.memory_key]}
        prompt_length = self.chain.prompt_length(docs, **inputs)
        remain = self.max_tokens - prompt_length
        while self.get_conversation_length() > remain:
            if len(self.buffered_history.messages) <= 2:
                raise RuntimeError("Exceed max_tokens, trunk size of retrieved documents is too large")
            temp = self.buffered_history.messages.pop(0)
            self.summarized_history_temp.messages.append(temp)
            temp = self.buffered_history.messages.pop(0)
            self.summarized_history_temp.messages.append(temp)
        return {self.memory_key: self.format_dialogue()}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        input_str, output_str = self._get_input_output(inputs, outputs)
        self.buffered_history.add_user_message(input_str.strip())
        self.buffered_history.add_ai_message(output_str.strip())
