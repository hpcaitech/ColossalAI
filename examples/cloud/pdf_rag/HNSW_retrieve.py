import numpy as np
import json
from FlagEmbedding import FlagAutoModel
import time
from rank_bm25 import BM25Okapi
import hnswlib

def get_list_shape(lst):
    shape = []
    current = lst
    while isinstance(current, list) and len(current) > 0:
        shape.append(len(current))
        current = current[0]
    return tuple(shape)

def load_model():
    return FlagAutoModel.from_finetuned(
        'BAAI/bge-base-en-v1.5',
        query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
        #  devices='cpu', # Uncomment this line if you want to use GPU.
        use_fp16=True
    )

def encode_query(model, query):
    query_vectors = [np.array(model.encode(query)).tolist()]
    print('query_vectors_shape', get_list_shape(query_vectors))
    return query_vectors

def load_data(vectors_path, docs_path):
    vectors = np.load(vectors_path).tolist()
    with open(docs_path, 'r', encoding='utf-8') as file:
        docs = json.load(file)
    return vectors, docs

def build_hnsw_index(vectors):
    # start_time = time.time()
    num_elements = len(vectors)
    p = hnswlib.Index(space='cosine', dim=768)
    p.init_index(max_elements=num_elements, ef_construction=200, M=16)
    # M defines the maximum number of outgoing connections in the graph. Higher M leads to higher accuracy/run_time at fixed ef/efConstruction.
    # ef_construction controls index search speed/build speed tradeoff. Increasing the efConstruction parameter may enhance index quality, but it also tends to lengthen the indexing time.
    p.add_items(np.array(vectors), np.arange(num_elements))
    # HNSW_time = time.time()
    #print('HNSW build time:', HNSW_time - start_time)
    p.set_ef(32)
    # ef controlling query time/accuracy trade-off. Higher ef leads to more accurate but slower search.
    return p

def search_hnsw(index, query_vectors, docs):
    # HNSW_time = time.time()
    labels, distances = index.knn_query(np.array(query_vectors), k=10)
    results = [docs[i]['content'] for i in labels[0]]
    # end_HNSW_time = time.time()
    # print('HNSW search time:', end_HNSW_time - HNSW_time)
    return results

def build_bm25(docs):
    corpus = [doc['content'] for doc in docs]
    tokenized_corpus = [list(text.split()) for text in corpus]
    # bm25_build_start = time.time()
    bm25 = BM25Okapi(tokenized_corpus)
    # bm25_build_end = time.time()
    # print('BM25 build time:', bm25_build_end - bm25_build_start)
    return bm25, corpus

def search_bm25(bm25, corpus, query):
    # bm25_search_start = time.time()
    tokenized_query = list(query.split())
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_n = np.argsort(bm25_scores)[::-1][:10]
    bm25_results = [corpus[i] for i in bm25_top_n]
    # bm25_search_end = time.time()
    # print('BM25 search time:', bm25_search_end - bm25_search_start)
    return bm25_results

def merge_results(results, bm25_results):
    merged_results = []
    for i in range(len(results)):
        merged_results.append(results[i])
    for i in range(len(bm25_results)):
        merged_results.append(bm25_results[i])
    merged_results = list(set(merged_results))
    return merged_results

def main():
    model = load_model()
    query = "This is a test query to find relevant documents."
    query_vectors = encode_query(model, query)
    vectors, docs = load_data('PATH_TO_YOUR_EMBEDDING.npy', 'PATH_TO_YOUR_JSON.json')
    
    hnsw_index   = build_hnsw_index(vectors)
    hnsw_results = search_hnsw(hnsw_index, query_vectors, docs)
    
    bm25, corpus = build_bm25(docs)
    bm25_results = search_bm25(bm25, corpus, query)
    
    merged_results = merge_results(hnsw_results, bm25_results)

    return merged_results
if __name__ == "__main__":
    retrieved_data=main()
