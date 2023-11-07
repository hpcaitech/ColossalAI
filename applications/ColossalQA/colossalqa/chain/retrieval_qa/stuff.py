"""
Chain that combines documents by stuffing into context

Modified from Original Source

This code is based on LangChain Ai's langchain, which can be found at
https://github.com/langchain-ai/langchain
The original code is licensed under the MIT license.
"""
import copy
from typing import Any, List

from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.docstore.document import Document
from langchain.schema import format_document


class CustomStuffDocumentsChain(StuffDocumentsChain):
    """Chain that combines documents by stuffing into context.

    This chain takes a list of documents and first combines them into a single string.
    It does this by formatting each document into a string with the `document_prompt`
    and then joining them together with `document_separator`. It then adds that new
    string to the inputs with the variable name set by `document_variable_name`.
    Those inputs are then passed to the `llm_chain`.

    Example:
        .. code-block:: python

            from langchain.chains import StuffDocumentsChain, LLMChain
            from langchain.prompts import PromptTemplate
            from langchain.llms import OpenAI

            # This controls how each document will be formatted. Specifically,
            # it will be passed to `format_document` - see that function for more
            # details.
            document_prompt = PromptTemplate(
                input_variables=["page_content"],
                 template="{page_content}"
            )
            document_variable_name = "context"
            llm = OpenAI()
            # The prompt here should take as an input variable the
            # `document_variable_name`
            prompt = PromptTemplate.from_template(
                "Summarize this content: {context}"
            )
            llm_chain = LLMChain(llm=llm, prompt=prompt)
            chain = StuffDocumentsChain(
                llm_chain=llm_chain,
                document_prompt=document_prompt,
                document_variable_name=document_variable_name
            )
    """

    def _get_inputs(self, docs: List[Document], **kwargs: Any) -> dict:
        """Construct inputs from kwargs and docs.

        Format and the join all the documents together into one input with name
        `self.document_variable_name`. The pluck any additional variables
        from **kwargs.

        Args:
            docs: List of documents to format and then join into single input
            **kwargs: additional inputs to chain, will pluck any other required
                arguments from here.

        Returns:
            dictionary of inputs to LLMChain
        """
        # Format each document according to the prompt

        # if the document is in the key-value format has a 'is_key_value_mapping'=True in meta_data and has 'value' in metadata
        # use the value to replace the key
        doc_prefix = kwargs.get("doc_prefix", "Supporting Document")
        docs_ = []
        for id, doc in enumerate(docs):
            doc_ = copy.deepcopy(doc)
            if doc_.metadata.get("is_key_value_mapping", False) and "value" in doc_.metadata:
                doc_.page_content = str(doc_.metadata["value"])
            prefix = doc_prefix + str(id)
            doc_.page_content = str(prefix + ":" + (" " if doc_.page_content[0] != " " else "") + doc_.page_content)
            docs_.append(doc_)

        doc_strings = [format_document(doc, self.document_prompt) for doc in docs_]
        arg_list = ["stop", "temperature", "top_k", "top_p", "max_new_tokens"]
        arg_list.extend(self.llm_chain.prompt.input_variables)
        # Join the documents together to put them in the prompt.
        inputs = {k: v for k, v in kwargs.items() if k in arg_list}
        inputs[self.document_variable_name] = self.document_separator.join(doc_strings)
        return inputs
