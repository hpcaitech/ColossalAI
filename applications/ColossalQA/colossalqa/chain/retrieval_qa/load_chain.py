"""
Load question answering chains.
For now, only the stuffed chain is modified

Modified from Original Source

This code is based on LangChain Ai's langchain, which can be found at
https://github.com/langchain-ai/langchain
The original code is licensed under the MIT license.
"""
import copy
from typing import Any, Mapping, Optional, Protocol

from colossalqa.chain.retrieval_qa.stuff import CustomStuffDocumentsChain
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.manager import Callbacks
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import stuff_prompt
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.prompt_template import BasePromptTemplate


class LoadingCallable(Protocol):
    """Interface for loading the combine documents chain."""

    def __call__(self, llm: BaseLanguageModel, **kwargs: Any) -> BaseCombineDocumentsChain:
        """Callable to load the combine documents chain."""


def _load_stuff_chain(
    llm: BaseLanguageModel,
    prompt: Optional[BasePromptTemplate] = None,
    document_variable_name: str = "context",
    verbose: Optional[bool] = None,
    callback_manager: Optional[BaseCallbackManager] = None,
    callbacks: Callbacks = None,
    **kwargs: Any,
) -> CustomStuffDocumentsChain:
    _prompt = prompt or stuff_prompt.PROMPT_SELECTOR.get_prompt(llm)
    if "llm_kwargs" in kwargs:
        llm_kwargs = copy.deepcopy(kwargs["llm_kwargs"])
        del kwargs["llm_kwargs"]
    else:
        llm_kwargs = {}
    llm_chain = LLMChain(
        llm=llm,
        prompt=_prompt,
        verbose=verbose,
        callback_manager=callback_manager,
        callbacks=callbacks,
        llm_kwargs=llm_kwargs,
    )
    return CustomStuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name=document_variable_name,
        verbose=verbose,
        callback_manager=callback_manager,
        callbacks=callbacks,
        **kwargs,
    )


def load_qa_chain(
    llm: BaseLanguageModel,
    chain_type: str = "stuff",
    verbose: Optional[bool] = None,
    callback_manager: Optional[BaseCallbackManager] = None,
    **kwargs: Any,
) -> BaseCombineDocumentsChain:
    """Load question answering chain.

    Args:
        llm: Language Model to use in the chain.
        chain_type: Type of document combining chain to use. Should be one of "stuff",
            "map_reduce", "map_rerank", and "refine".
        verbose: Whether chains should be run in verbose mode or not. Note that this
            applies to all chains that make up the final chain.
        callback_manager: Callback manager to use for the chain.

    Returns:
        A chain to use for question answering.
    """
    loader_mapping: Mapping[str, LoadingCallable] = {"stuff": _load_stuff_chain}
    if chain_type not in loader_mapping:
        raise ValueError(f"Got unsupported chain type: {chain_type}. " f"Should be one of {loader_mapping.keys()}")
    return loader_mapping[chain_type](llm, verbose=verbose, callback_manager=callback_manager, **kwargs)
