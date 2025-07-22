# PDF_RAG Workflow Demo

This project demonstrates a document retrieval and vectorization workflow based on Haystack, FlagEmbedding, HNSWLib, and BM25.

## Directory Structure

- `RAG_workflow/parse.py`: Parses and splits PDF documents, and generates the content in JSON format.
- `RAG_workflow/embedding.py`：Vectorizes the document content and produces the embedding vector base.
- `RAG_workflow/HNSW_retrieve.py`：Performs hybrid retrieval and recall using HNSW and BM25.

## Environment Setup

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Workflow Steps

1. **PDF Parsing**
   - Modify `PATH_TO_YOUR_PDF_DIRECTORY` in `parse.py` to your PDF folder path.
   - Update the output JSON path（`PATH_TO_YOUR_JSON`）。
   - Run:
     ```bash
     python PDF-RAG/parse.py
     ```
   - The generated JSON file will be used in the next embedding step.

2. **JSON Vectorization**
   - In `embedding.py`, update the input JSON path (`PATH_TO_YOUR_JSON.json`) and the output embedding file path (`PATH_TO_YOUR_EMBEDDING.npy`).
   - Run:
     ```bash
     python PDF-RAG/embedding.py
     ```

3. **Retrieval and Recall**
   - In `HNSW_retrieve.py`, update the embedding and JSON paths (`PATH_TO_YOUR_JSON.json`).
   - Run:
     ```bash
     python PDF-RAG/HNSW_retrieve.py
     ```
   - The script will output the construction and retrieval times for both HNSW and BM25, along with the merged retrieval results.
   - Adjust HNSW and BM25 parameters according to the descriptions to get desired results.
   - In `hnswlib.Index()`, use `space='l2'` for Squared L2, `'ip'` for Inner Product, and `'cosine'` for Cosine Similarity.

## Dependencies

- numpy
- scikit-learn
- hnswlib
- rank_bm25
- FlagEmbedding
- haystack
- haystack-integrations



---

Feel free to open an issue if you encounter any problems!
