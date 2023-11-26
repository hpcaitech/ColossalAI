"""
Multilingual retrieval based conversation system backed by ChatGPT
"""

import argparse
import os

from colossalqa.data_loader.document_loader import DocumentLoader
from colossalqa.memory import ConversationBufferWithSummary
from colossalqa.retriever import CustomRetriever
from langchain import LLMChain
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multilingual retrieval based conversation system backed by ChatGPT")
    parser.add_argument("--open_ai_key_path", type=str, default=None, help="path to the model")
    parser.add_argument(
        "--sql_file_path", type=str, default=None, help="path to the a empty folder for storing sql files for indexing"
    )

    args = parser.parse_args()

    if not os.path.exists(args.sql_file_path):
        os.makedirs(args.sql_file_path)

    # Setup openai key
    # Set env var OPENAI_API_KEY or load from a file
    openai_key = open(args.open_ai_key_path).read()
    os.environ["OPENAI_API_KEY"] = openai_key

    llm = OpenAI(temperature=0.6)

    information_retriever = CustomRetriever(k=3, sql_file_path=args.sql_file_path, verbose=True)
    # VectorDB
    embedding = HuggingFaceEmbeddings(
        model_name="moka-ai/m3e-base", model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": False}
    )

    # Define memory with summarization ability
    memory = ConversationBufferWithSummary(llm=llm)

    # Load data to vector store
    print("Select files for constructing retriever")
    documents = []
    while True:
        file = input("Enter a file path or press Enter directory without input to exit:").strip()
        if file == "":
            break
        data_name = input("Enter a short description of the data:")
        retriever_data = DocumentLoader([[file, data_name.replace(" ", "_")]]).all_data

        # Split
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
        splits = text_splitter.split_documents(retriever_data)
        documents.extend(splits)
    # Create retriever
    information_retriever.add_documents(docs=documents, cleanup="incremental", mode="by_source", embedding=embedding)

    prompt_template = """Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
    If the answer cannot be infered based on the given context, please don't share false information.
    Use the context and chat history to respond to the human's input at the end or carry on the conversation. You should generate one response only. No following up is needed.

    context:
    {context}

    chat history
    {chat_history}

    Human: {question}
    Assistant:"""

    prompt_template_disambiguate = """You are a helpful, respectful and honest assistant. You always follow the instruction.
    Please replace any ambiguous references in the given sentence with the specific names or entities mentioned in the chat history or just output the original sentence if no chat history is provided or if the sentence doesn't contain ambiguous references. Your output should be the disambiguated sentence itself (in the same line as "disambiguated sentence:") and contain nothing else.

    Here is an example:
    Chat history:
    Human: I have a friend, Mike. Do you know him?
    Assistant: Yes, I know a person named Mike

    sentence: What's his favorite food?
    disambiguated sentence: What's Mike's favorite food?
    END OF EXAMPLE

    Chat history:
    {chat_history}

    sentence: {input}
    disambiguated sentence:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["question", "chat_history", "context"])

    memory.initiate_document_retrieval_chain(
        llm,
        PROMPT,
        information_retriever,
        chain_type_kwargs={
            "chat_history": "",
        },
    )

    PROMPT_DISAMBIGUATE = PromptTemplate(
        template=prompt_template_disambiguate, input_variables=["chat_history", "input"]
    )

    llm_chain = RetrievalQA.from_chain_type(
        llm=llm,
        verbose=False,
        chain_type="stuff",
        retriever=information_retriever,
        chain_type_kwargs={"prompt": PROMPT, "memory": memory},
    )
    llm_chain_disambiguate = LLMChain(llm=llm, prompt=PROMPT_DISAMBIGUATE)

    def disambiguity(input):
        out = llm_chain_disambiguate.run({"input": input, "chat_history": memory.buffer})
        return out.split("\n")[0]

    information_retriever.set_rephrase_handler(disambiguity)

    while True:
        user_input = input("User: ")
        if " end " in user_input:
            print("Agent: Happy to chat with you ：)")
            break
        agent_response = llm_chain.run(user_input)
        agent_response = agent_response.split("\n")[0]
        print(f"Agent: {agent_response}")
