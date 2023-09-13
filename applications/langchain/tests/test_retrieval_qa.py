from colossalqa.retrieval_conversation_universal import UniversalRetrievalConversation
import os

def test_en_retrievalQA():
    os.environ["ZH_MODEL_PATH"] = "/data3/data/model_eval_for_non_commercial_use/chatglm2-6b"
    os.environ["ZH_MODEL_NAME"] = "chatglm"
    os.environ["EN_MODEL_PATH"] = "/data3/data/model_eval_for_commerical_use/Llama-2-7b-hf"
    os.environ["EN_MODEL_NAME"] = "llama"
    os.environ["TEST_DATA_PATH_EN"] = "/home/lcyab/data3/ColossalAI/applications/langchain/data/test_data/companies.txt"
    os.environ["TEST_DATA_PATH_ZH"] = "/home/lcyab/data3/ColossalAI/applications/langchain/data/test_data/companies_zh.txt"
    data_path_en = os.environ.get('TEST_DATA_PATH_EN')
    data_path_zh = os.environ.get('TEST_DATA_PATH_ZH')
    qa_session = UniversalRetrievalConversation(files_en=[[data_path_en, 'company information']], files_zh=[[data_path_zh, 'company information']])
    ans = qa_session.run("which company runs business in hotel industry?", which_language='en')
    print(ans)
    # assert 'Marriott' in ans

def test_zh_retrievalQA():
    os.environ["ZH_MODEL_PATH"] = "/data3/data/model_eval_for_non_commercial_use/chatglm2-6b"
    os.environ["ZH_MODEL_NAME"] = "chatglm"
    os.environ["EN_MODEL_PATH"] = "/data3/data/model_eval_for_commerical_use/Llama-2-7b-hf"
    os.environ["EN_MODEL_NAME"] = "llama"
    os.environ["TEST_DATA_PATH_EN"] = "/home/lcyab/data3/ColossalAI/applications/langchain/data/test_data/companies.txt"
    os.environ["TEST_DATA_PATH_ZH"] = "/home/lcyab/data3/ColossalAI/applications/langchain/data/test_data/companies_zh.txt"
    data_path_en = os.environ.get('TEST_DATA_PATH_EN')
    data_path_zh = os.environ.get('TEST_DATA_PATH_ZH')
    qa_session = UniversalRetrievalConversation(files_en=[[data_path_en, 'company information']], files_zh=[[data_path_zh, 'company information']])
    ans = qa_session.run("哪家公司在经营酒店业务？", which_language='zh')
    print(ans)
    # assert '万豪' in ans


if __name__ == "__main__":
    test_en_retrievalQA()
    test_zh_retrievalQA()