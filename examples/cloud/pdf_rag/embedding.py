import json
import numpy as np
from FlagEmbedding import FlagAutoModel
import time
from sklearn.metrics.pairwise import cosine_similarity
import os

def load_model(model_name='BAAI/bge-base-en-v1.5', use_fp16=True):
    return FlagAutoModel.from_finetuned(
        model_name,
        query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
        # device='cpu', # Uncomment this line if you want to use GPU.
        use_fp16=use_fp16
    )

def load_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def extract_texts(data):
    return [doc.get('content', '').strip() for doc in data]

def generate_embeddings(model, texts):
    return np.array(model.encode(texts))

def save_embeddings(embeddings, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, embeddings)

def load_embeddings(file_path):
    try:
        return np.load(file_path)
    except FileNotFoundError:
        return None


def main():
    config = {
        'model_name': 'BAAI/bge-base-en-v1.5',
        'json_path': 'PATH_TO_YOUR_JSON.json',
        'embedding_path': 'PATH_TO_YOUR_EMBEDDING.npy',
        'use_fp16': True,
        'use_precomputed_embeddings': False
    }
    
    model = load_model(
        model_name=config['model_name'],
        use_fp16=config['use_fp16']
    )
    
    if config['use_precomputed_embeddings']:
        embeddings = load_embeddings(config['embedding_path'])
        if embeddings is None:
            return
    else:
        data = load_data(config['json_path'])
        if not data:
            return
            
        texts = extract_texts(data)
        embeddings = generate_embeddings(model, texts)
        save_embeddings(embeddings, config['embedding_path'])
    
# Test demo with simple KNN cosine_similarity
    # query='This is a test query to find relevant documents.'
    # query_embedding=np.array(model.encode(query))
    # similarity_scores = cosine_similarity([query_embedding], embeddings)
    # indices = np.argsort(-similarity_scores)
    
    return embeddings

if __name__ == '__main__':
    main()
