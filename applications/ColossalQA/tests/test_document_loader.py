import os
from colossalqa.data_loader.document_loader import DocumentLoader


def test_add_document():
    PATH = os.environ.get('TEST_DOCUMENT_LOADER_DATA_PATH')
    files = [[PATH, 'all data']]
    document_loader = DocumentLoader(files)
    documents = document_loader.all_data
    all_files = []
    for doc in documents:
        assert isinstance(doc.page_content, str)==True
        if doc.metadata['source'] not in all_files:
            all_files.append(doc.metadata['source'])
    print(all_files)
    assert len(all_files) == 6


if __name__=='__main__':
    test_add_document()

