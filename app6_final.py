from flask import Flask, request, jsonify, render_template, Response, stream_with_context
import ollama
import numpy as np
from usearch.index import Index
import time
import re
import os
import pickle
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
import threading
from werkzeug.utils import secure_filename
from parser import parse_hybrid_pdf
import tempfile

app = Flask(__name__)

# --- CONFIGURATION ---
EMBEDDING_MODEL = 'nomic-embed-text'
# EMBEDDING_MODEL = 'all-minilm' --->> ollama pull all-minilm
# LANGUAGE_MODEL = 'ministral-3:3b'
LANGUAGE_MODEL = 'mistral:7b'
# EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
CACHE_DIR = "cache"
INDEX_CACHE = "cache/index_cache"
MIN_CHUNK_SIZE = 500
MAX_CHUNK_SIZE = 1500
SIMILARITY_THRESHOLD = 0.75
BATCH_SIZE = 50  # Process embeddings in batches

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(INDEX_CACHE, exist_ok=True)

# --- GLOBAL STATE ---
class RAGState:
    def __init__(self):
        self.vector_index = None
        self.chunk_map = {}
        self.is_ready = False
        self.lock = threading.Lock()

state = RAGState()

# --- OPTIMIZED HELPER FUNCTIONS ---
def to_float16(float_vector):
    return np.array(float_vector, dtype=np.float16)

def get_file_hash(files_content):
    """Fast hash using first/last 1000 chars + length"""
    sample = files_content[:1000] + files_content[-1000:] + str(len(files_content))
    return hashlib.md5(sample.encode('utf-8')).hexdigest()

def semantic_chunker_optimized(text):
    """
    Semantic chunker that ensures NO TEXT is dropped.
    """
    text = text.replace('\n', ' ').replace('  ', ' ').strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return []
    
    print(f"Processing {len(sentences)} sentences...")
    
    # BATCH EMBEDDING
    sentence_embeddings = []
    for i in range(0, len(sentences), BATCH_SIZE):
        batch = sentences[i:i + BATCH_SIZE]
        batch_embeds = []
        for sent in batch:  
            embed = ollama.embed(model=EMBEDDING_MODEL, input=sent)['embeddings'][0]
            batch_embeds.append(np.array(embed, dtype=np.float16))
        sentence_embeddings.extend(batch_embeds)
        
        if i % 100 == 0:
            print(f"  Processed {min(i + BATCH_SIZE, len(sentences))}/{len(sentences)} sentences")
    
    # Calculate similarities
    similarities = []
    for i in range(len(sentence_embeddings) - 1):
        vec1, vec2 = sentence_embeddings[i], sentence_embeddings[i+1]
        sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)
        similarities.append(float(sim))
    
    # Create chunks (ZERO LOSS LOGIC)
    chunks = []
    current_chunk = [sentences[0]]
    current_len = len(sentences[0])
    
    for i in range(len(similarities)):
        sent = sentences[i+1]
        sent_len = len(sent)
        
        # Split only if max size reached OR similarity drops low
        should_split = (
            similarities[i] < SIMILARITY_THRESHOLD or 
            current_len + sent_len > MAX_CHUNK_SIZE
        )
        
        # CHANGED: We removed the check `and current_len >= MIN_CHUNK_SIZE`
        # Now, if the topic changes, we split immediately, preserving small specific facts.
        if should_split:
            chunks.append(' '.join(current_chunk).strip())
            current_chunk = [sent]
            current_len = sent_len
        else:
            current_chunk.append(sent)
            current_len += sent_len
    
    # Append the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk).strip())
    
    return chunks
    # print("--------------> chunks", chunks)
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

# --- FLASK ROUTES ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    start_time = time.time()
    
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files received"}), 400

    print("\n=== Processing Upload ===")

    # Creating temp dir for PDFs because OCR needs physical files
    TEMP_DIR = "temp_uploads"
    os.makedirs(TEMP_DIR, exist_ok=True)

    full_text = ""
    for f in files:
        # content = f.read().decode('utf-8', errors='ignore')
        # full_text += content + "\n"
        filename = secure_filename(f.filename)
        file_ext = os.path.splitext(filename)[1].lower()

        if file_ext == '.pdf':
            print(f"PDF detected: {filename}")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_path = temp_file.name
                f.save(temp_path)

            try:
                #calling parser.py func
                pdf_text = parse_hybrid_pdf(temp_path)
                full_text += pdf_text + "\n"
            except Exception as e:
                print(f"Error parsing PDF {filename} : {e}")
            finally:
                #Cleanup: Remove the temp file to save space
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        else:
            #fallback for .txt or other text files
            try:
                content = f.read().decode('utf-8', errors='ignore')
                full_text += content + "\n"
            except Exception as e:
                print(f"Error reading file {filename}: {e}")

    
    read_time = time.time() - start_time
    print(f"✓ Files read in {read_time:.2f}s")

    file_hash = get_file_hash(full_text)
    chunk_cache_path = os.path.join(CACHE_DIR, f"chunks_{file_hash}.pkl")
    
    chunks = []
    chunk_time = 0
    
    if os.path.exists(chunk_cache_path):
        print("✓ Loading chunks from cache...")
        cache_start = time.time()
        with open(chunk_cache_path, 'rb') as f:
            chunks = pickle.load(f)
        chunk_time = time.time() - cache_start
        print(f"✓ Loaded {len(chunks)} chunks in {chunk_time:.2f}s")
    else:
        print("Processing new content (Semantic Only)...")
        chunk_start = time.time()
        
        # ALWAYS use semantic chunker now
        chunks = semantic_chunker_optimized(full_text)
        
        chunk_time = time.time() - chunk_start
        print(f"✓ Created {len(chunks)} chunks in {chunk_time:.2f}s")
        
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(chunk_cache_path, 'wb') as f:
            pickle.dump(chunks, f)
        print("✓ Chunks cached")

    index_start = time.time()
    with state.lock:
        index, mapping = build_index_optimized(chunks, file_hash)
        state.vector_index = index
        state.chunk_map = mapping
        state.is_ready = True
    
    index_time = time.time() - index_start
    print(f"✓ Index ready in {index_time:.2f}s")
    
    total_time = time.time() - start_time
    print(f"✓ TOTAL TIME: {total_time:.2f}s\n")
    
    return jsonify({
        "message": f"Processed {len(chunks)} chunks successfully!",
        "count": len(chunks),
        "time": {
            "read": round(read_time, 2),
            "chunking": round(chunk_time, 2),
            "indexing": round(index_time, 2),
            "total": round(total_time, 2)
        }
    })

@app.route('/chat', methods=['POST'])
def chat():
    if not state.is_ready:
        return jsonify({"error": "System not ready. Upload files first."}), 400

    data = request.json
    query = data.get("message", "")

    print(f"\n--- Query: {query} ---")
    
    # Retrieve (with lock to prevent race conditions)
    with state.lock:
        embed = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
        matches = state.vector_index.search(to_float16(embed), 15)
    
    results = []
    for idx, dist in zip(matches.keys, matches.distances):
        sim = 1 - dist
        if sim > 0.25:
            results.append((state.chunk_map[idx], sim))
    
    results.sort(key=lambda x: x[1], reverse=True)
    top_results = results[:5]

    print(f"Retrieved {len(top_results)} chunks:")
    for i, (txt, score) in enumerate(top_results):
        print(f"  [{i}] {score:.2f} | {txt}...")

    if not top_results:
        return jsonify({"error": "No relevant data found in documents."})

    context_str = "\n\n".join([f"Source {i+1}: {txt}" for i, (txt, _) in enumerate(top_results)])
    
    # Stream response
    def generate():
        context_data = [{"text": txt, "score": float(score)} for txt, score in top_results]
        yield json.dumps({"type": "context", "data": context_data}) + "\n"

        system_instruction = """You are a helpful AI assistant.
1. Use ONLY the provided Context to answer questions.
2. If the answer is not in the context, say "I don't have that information."
3. Be concise and professional."""

        final_prompt = f"""{system_instruction}

Context:
{context_str}

Question: {query}

Answer:"""

        stream = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[{'role': 'user', 'content': final_prompt}],
            stream=True
        )
        
        for chunk in stream:
            content = chunk['message']['content']
            if content:
                yield json.dumps({"type": "token", "content": content}) + "\n"

    return Response(stream_with_context(generate()), mimetype='application/x-ndjson')

@app.route('/clear-cache', methods=['POST'])
def clear_cache():
    """Endpoint to clear all caches"""
    import shutil
    try:
        shutil.rmtree(CACHE_DIR)
        shutil.rmtree(INDEX_CACHE)
        os.makedirs(CACHE_DIR, exist_ok=True)
        os.makedirs(INDEX_CACHE, exist_ok=True)
        
        with state.lock:
            state.vector_index = None
            state.chunk_map = {}
            state.is_ready = False
        
        return jsonify({"message": "Cache cleared successfully!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True)