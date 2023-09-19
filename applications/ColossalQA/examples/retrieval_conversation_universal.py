import argparse
from colossalqa.retrieval_conversation_universal import UniversalRetrievalConversation

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--en_model_path', type=str, default=None)
    parser.add_argument('--zh_model_path', type=str, default=None)
    parser.add_argument('--zh_model_name', type=str, default=None)
    parser.add_argument('--en_model_name', type=str, default=None)
    parser.add_argument('--sql_file_path', type=str, default=None, help='path to the a empty folder for storing sql files for indexing')
    args = parser.parse_args()
    
    # Will ask for documents path in runnning time
    session = UniversalRetrievalConversation(files_en=None, 
                files_zh=None, 
                zh_model_path=args.zh_model_path, en_model_path=args.en_model_path,
                zh_model_name=args.zh_model_name, en_model_name=args.en_model_name,
                sql_file_path=args.sql_file_path
                )
    session.start_test_session()
        