import os

from colossalqa.retrieval_conversation_universal import UniversalRetrievalConversation


def test_en_retrievalQA():
    data_path_en = os.environ.get("TEST_DATA_PATH_EN")
    data_path_zh = os.environ.get("TEST_DATA_PATH_ZH")
    en_model_path = os.environ.get("EN_MODEL_PATH")
    zh_model_path = os.environ.get("ZH_MODEL_PATH")
    zh_model_name = os.environ.get("ZH_MODEL_NAME")
    en_model_name = os.environ.get("EN_MODEL_NAME")
    sql_file_path = os.environ.get("SQL_FILE_PATH")
    qa_session = UniversalRetrievalConversation(
        files_en=[{"data_path": data_path_en, "name": "company information", "separator": "\n"}],
        files_zh=[{"data_path": data_path_zh, "name": "company information", "separator": "\n"}],
        zh_model_path=zh_model_path,
        en_model_path=en_model_path,
        zh_model_name=zh_model_name,
        en_model_name=en_model_name,
        sql_file_path=sql_file_path,
    )
    ans = qa_session.run("which company runs business in hotel industry?", which_language="en")
    print(ans)


def test_zh_retrievalQA():
    data_path_en = os.environ.get("TEST_DATA_PATH_EN")
    data_path_zh = os.environ.get("TEST_DATA_PATH_ZH")
    en_model_path = os.environ.get("EN_MODEL_PATH")
    zh_model_path = os.environ.get("ZH_MODEL_PATH")
    zh_model_name = os.environ.get("ZH_MODEL_NAME")
    en_model_name = os.environ.get("EN_MODEL_NAME")
    sql_file_path = os.environ.get("SQL_FILE_PATH")
    qa_session = UniversalRetrievalConversation(
        files_en=[{"data_path": data_path_en, "name": "company information", "separator": "\n"}],
        files_zh=[{"data_path": data_path_zh, "name": "company information", "separator": "\n"}],
        zh_model_path=zh_model_path,
        en_model_path=en_model_path,
        zh_model_name=zh_model_name,
        en_model_name=en_model_name,
        sql_file_path=sql_file_path,
    )
    ans = qa_session.run("哪家公司在经营酒店业务？", which_language="zh")
    print(ans)


if __name__ == "__main__":
    test_en_retrievalQA()
    test_zh_retrievalQA()
