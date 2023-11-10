import os

from colossalqa.data_loader.document_loader import DocumentLoader
from colossalqa.local.llm import ColossalAPI, ColossalLLM
from colossalqa.memory import ConversationBufferWithSummary
from colossalqa.prompt.prompt import PROMPT_RETRIEVAL_QA_ZH
from colossalqa.retriever import CustomRetriever
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


def test_memory_long():
    model_path = os.environ.get("EN_MODEL_PATH")
    data_path = os.environ.get("TEST_DATA_PATH_EN")
    model_name = os.environ.get("EN_MODEL_NAME")
    sql_file_path = os.environ.get("SQL_FILE_PATH")

    if not os.path.exists(sql_file_path):
        os.makedirs(sql_file_path)

    colossal_api = ColossalAPI.get_api(model_name, model_path)
    llm = ColossalLLM(n=4, api=colossal_api)
    memory = ConversationBufferWithSummary(
        llm=llm, max_tokens=600, llm_kwargs={"max_new_tokens": 50, "temperature": 0.6, "do_sample": True}
    )
    retriever_data = DocumentLoader([[data_path, "company information"]]).all_data

    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    splits = text_splitter.split_documents(retriever_data)

    embedding = HuggingFaceEmbeddings(
        model_name="moka-ai/m3e-base", model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": False}
    )

    # Create retriever
    information_retriever = CustomRetriever(k=3, sql_file_path=sql_file_path)
    information_retriever.add_documents(docs=splits, cleanup="incremental", mode="by_source", embedding=embedding)

    memory.initiate_document_retrieval_chain(
        llm,
        PROMPT_RETRIEVAL_QA_ZH,
        information_retriever,
        chain_type_kwargs={
            "chat_history": "",
        },
    )

    # This keep the prompt length excluding dialogues the same
    docs = information_retriever.get_relevant_documents("this is a test input.")
    prompt_length = memory.chain.prompt_length(docs, **{"question": "this is a test input.", "chat_history": ""})
    remain = 600 - prompt_length
    have_summarization_flag = False
    for i in range(40):
        chat_history = memory.load_memory_variables({"question": "this is a test input.", "input_documents": docs})[
            "chat_history"
        ]

        assert memory.get_conversation_length() <= remain
        memory.save_context({"question": "this is a test input."}, {"output": "this is a test output."})
        if "A summarization of historical conversation:" in chat_history:
            have_summarization_flag = True
    assert have_summarization_flag == True


def test_memory_short():
    model_path = os.environ.get("EN_MODEL_PATH")
    data_path = os.environ.get("TEST_DATA_PATH_EN")
    model_name = os.environ.get("EN_MODEL_NAME")
    sql_file_path = os.environ.get("SQL_FILE_PATH")

    if not os.path.exists(sql_file_path):
        os.makedirs(sql_file_path)

    colossal_api = ColossalAPI.get_api(model_name, model_path)
    llm = ColossalLLM(n=4, api=colossal_api)
    memory = ConversationBufferWithSummary(
        llm=llm, llm_kwargs={"max_new_tokens": 50, "temperature": 0.6, "do_sample": True}
    )
    retriever_data = DocumentLoader([[data_path, "company information"]]).all_data

    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    splits = text_splitter.split_documents(retriever_data)

    embedding = HuggingFaceEmbeddings(
        model_name="moka-ai/m3e-base", model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": False}
    )

    # create retriever
    information_retriever = CustomRetriever(k=3, sql_file_path=sql_file_path)
    information_retriever.add_documents(docs=splits, cleanup="incremental", mode="by_source", embedding=embedding)

    memory.initiate_document_retrieval_chain(
        llm,
        PROMPT_RETRIEVAL_QA_ZH,
        information_retriever,
        chain_type_kwargs={
            "chat_history": "",
        },
    )

    # This keep the prompt length excluding dialogues the same
    docs = information_retriever.get_relevant_documents("this is a test input.", return_scores=True)

    for i in range(4):
        chat_history = memory.load_memory_variables({"question": "this is a test input.", "input_documents": docs})[
            "chat_history"
        ]
        assert chat_history.count("Assistant: this is a test output.") == i
        assert chat_history.count("Human: this is a test input.") == i
        memory.save_context({"question": "this is a test input."}, {"output": "this is a test output."})


if __name__ == "__main__":
    test_memory_short()
    test_memory_long()
