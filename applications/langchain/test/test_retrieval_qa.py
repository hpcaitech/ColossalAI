from colossalqa.retrieval_conversation_universal import UniversalRetrievalConversation
import os

def test_en_retrievalQA():
    data_path_en = os.environ.get('TEST_DATA_PATH_EN')
    data_path_zh = os.environ.get('TEST_DATA_PATH_ZH')
    qa_session = UniversalRetrievalConversation(files_en=[[data_path_en, 'company information']], files_zh=[[data_path_zh, 'company information']])
    ans = qa_session.run("which company runs business in hotel industry?", which_language='en')
    print(ans)
    # assert 'Marriott' in ans

def test_zh_retrievalQA():
    data_path_en = os.environ.get('TEST_DATA_PATH_EN')
    data_path_zh = os.environ.get('TEST_DATA_PATH_ZH')
    qa_session = UniversalRetrievalConversation(files_en=[[data_path_en, 'company information']], files_zh=[[data_path_zh, 'company information']])
    ans = qa_session.run("哪家公司在经营酒店业务？", which_language='zh')
    print(ans)
    # assert '万豪' in ans


if __name__ == "__main__":
    test_en_retrievalQA()
    test_zh_retrievalQA()