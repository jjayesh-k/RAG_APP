import os
import pickle
import ollama
from usearch.index import Index
import numpy as np
import hashlib
from config import EMBEDDING_MODEL, INDEX_CACHE, BATCH_SIZE

def to_float16(float_vector):
    return np.array(float_vector, dtype=np.float16)

def get_file_hash(files_content):
    """Fast hash using first/last 1000 chars + length"""
    sample = files_content[:1000] + files_content[-1000:] + str(len(files_content))
    return hashlib.md5(sample.encode('utf-8')).hexdigest()

def build_index_optimized(chunks, file_hash):
    """
    Build index with caching - checks if index already exists.
    """
    index_path = os.path.join(INDEX_CACHE, f"index_{file_hash}.usearch")
    chunk_map_path = os.path.join(INDEX_CACHE, f"chunks_{file_hash}.pkl")
    
    # Try to load existing index
    if os.path.exists(index_path) and os.path.exists(chunk_map_path):
        print("Loading cached index from disk...")
        index = Index.restore(index_path)
        with open(chunk_map_path, 'rb') as f:
            mapping = pickle.load(f)
        return index, mapping
    
    # Build new index
    print("Building new index...")
    dummy = ollama.embed(model=EMBEDDING_MODEL, input="test")['embeddings'][0]
    index = Index(ndim=len(dummy), metric='cos', dtype='f16')
    mapping = {}
    
    # Batch embed chunks
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        
        for j, chunk in enumerate(batch):
            idx = i + j
            embed = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
            index.add(idx, to_float16(embed))
            mapping[idx] = chunk
        
        if i % 100 == 0:
            print(f"  Indexed {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks")
    
    os.makedirs(INDEX_CACHE, exist_ok=True)

    # Save index to disk
    index.save(index_path)
    with open(chunk_map_path, 'wb') as f:
        pickle.dump(mapping, f)
    
    print("✓ Index saved to disk")
    return index, mapping