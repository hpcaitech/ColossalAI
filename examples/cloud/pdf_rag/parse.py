from pathlib import Path
import time
import json
from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Document

def create_indexing_pipeline():
    document_store = InMemoryDocumentStore()
    converter = PyPDFToDocument()
    cleaner = DocumentCleaner()
    splitter = DocumentSplitter(split_by="sentence", split_length=1)
    writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP)
    
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("converter", converter)
    indexing_pipeline.add_component("cleaner", cleaner)
    indexing_pipeline.add_component("splitter", splitter)
    indexing_pipeline.add_component("writer", writer)
    
    indexing_pipeline.connect("converter", "cleaner")
    indexing_pipeline.connect("cleaner", "splitter")
    indexing_pipeline.connect("splitter", "writer")
    
    return indexing_pipeline, document_store

def process_pdfs(pdf_directory, indexing_pipeline):
    papers_dir = Path(pdf_directory)
    pdf_files = list(papers_dir.glob("*.pdf"))
    for pdf_file in pdf_files:    
        try:
            indexing_pipeline.run({"converter": {"sources": [pdf_file]}})
        except:
            pass

def save_to_json(document_store, output_path):
    all_documents = document_store.filter_documents()
    docs_list = [doc.to_dict() for doc in all_documents]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(docs_list, f, ensure_ascii=False, indent=2)

def main():
    PDF_DIRECTORY = "PATH_TO_YOUR_PDF_DIRECTORY"
    OUTPUT_JSON = "PATH_TO_YOUR_JSON"
    
    start_time = time.time()
    indexing_pipeline, document_store = create_indexing_pipeline()
    process_pdfs(PDF_DIRECTORY, indexing_pipeline)
    save_to_json(document_store, OUTPUT_JSON)

if __name__ == "__main__":
    main()
