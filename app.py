from flask import Flask, request, jsonify, render_template, Response, stream_with_context
import ollama
import time
import os
import pickle
import json
import threading
from werkzeug.utils import secure_filename
# from parser import parse_hybrid_pdf
from pymupdf_parser import parse_hybrid_pdf
import tempfile
# from chunker import semantic_chunker_optimized
from recursive_chunker import chunk_markdown
from indexer import build_index_optimized, to_float16, get_file_hash
from config import EMBEDDING_MODEL, LANGUAGE_MODEL, CACHE_DIR, INDEX_CACHE, BATCH_SIZE
import sys
import webbrowser
from threading import Timer

# Check if a settings.json file exists next to the .exe
if getattr(sys, 'frozen', False):
    app_dir = os.path.dirname(sys.executable)
else:
    app_dir = os.path.dirname(os.path.abspath(__file__))

settings_path = os.path.join(app_dir, 'settings.json')

if os.path.exists(settings_path):
    print(f"✓ Loading custom settings from {settings_path}")
    try:
        with open(settings_path, 'r') as f:
            settings = json.load(f)
            EMBEDDING_MODEL = settings.get('EMBEDDING_MODEL', EMBEDDING_MODEL)
            LANGUAGE_MODEL = settings.get('LANGUAGE_MODEL', LANGUAGE_MODEL)
    except Exception as e:
        print(f"⚠️ Error loading settings: {e}")

print(f"Using Models -> Language: {LANGUAGE_MODEL} | Embedding: {EMBEDDING_MODEL}")
app = Flask(__name__)

# --- GLOBAL STATE ---
class RAGState:
    def __init__(self):
        self.vector_index = None
        self.chunk_map = {}
        self.is_ready = False
        self.lock = threading.Lock()

state = RAGState()

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
        chunks = chunk_markdown(full_text)
        
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

import re # Add this at the top of app.py

@app.route('/chat', methods=['POST'])
def chat():
    if not state.is_ready:
        return jsonify({"error": "System not ready. Upload files first."}), 400

    data = request.json
    query = data.get("message", "")

    print(f"\n--- Query: {query} ---")
    
    # 1. Retrieve Candidates (Vector Search)
    with state.lock:
        # USE EMBEDDING_MODEL (Uppercase constant)
        embed = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
        # Fetch Top 30 to cast a wide net
        matches = state.vector_index.search(to_float16(embed), 30) 
    
    # 2. Setup Keywords & Phrases
    # List A: Words to Ignore completely (Stop Words)
    stop_words = {
        'what', 'is', 'explain', 'the', 'a', 'an', 'in', 'on', 'of', 'for', 'to', 'and', 
        'how', 'do', 'does', 'are', 'all', 'me', 'us', 'list', 'show', 'tell', 'about', 'describe'
    }
    
    # List B: Command words to strip ONLY for the "Phrase Match" check
    # We remove "explain", but we KEEP "of" so "Freedom OF Association" stays intact
    command_prefixes = ['explain', 'what is', 'tell me about', 'describe', 'list', 'show me']
    
    query_lower = query.lower()
    
    # --- FIX 1: Smart Phrase Cleaning ---
    # "Explain freedom of association" -> "freedom of association"
    search_phrase = query_lower
    for cmd in command_prefixes:
        if search_phrase.startswith(cmd):
            search_phrase = search_phrase[len(cmd):].strip()
    
    # Create "Query Terms" for individual word matching
    query_terms = set([word for word in query_lower.split() if word not in stop_words])
    
    results = []
    
    # 3. Hybrid Scoring Loop
    for idx, dist in zip(matches.keys, matches.distances):
        if idx == -1: continue # Skip empty Faiss slots
            
        vector_score = 1 - dist
        if vector_score < 0.20: continue
            
        chunk_text = state.chunk_map[idx]
        chunk_lower = chunk_text.lower()
        keyword_bonus = 0
        
        # --- BONUS A: Smart Phrase Match (+0.5) ---
        # Now we check if "freedom of association" is in the text (ignoring "Explain")
        # len > 3 check prevents matching empty strings if user types just "explain"
        if len(search_phrase) > 3 and search_phrase in chunk_lower:
            keyword_bonus += 0.5 
            
        # --- BONUS B: Header Match (+0.3) ---
        first_line = chunk_lower.split('\n')[0]
        if any(term in first_line for term in query_terms):
            keyword_bonus += 0.3
            
        # --- BONUS C: Word Match (max +0.25) ---
        matches_found = 0
        for term in query_terms:
            if re.search(r'\b' + re.escape(term) + r'\b', chunk_lower):
                matches_found += 1
        word_bonus = min(matches_found * 0.05, 0.25)
        keyword_bonus += word_bonus
        
        final_score = vector_score + keyword_bonus
        
        # STORE INDEX TOO: (Index, Text, Score)
        results.append((idx, chunk_text, final_score))

    # --- FIX 2: Context Expansion (Smart Neighbors) ---
    # Sort by score to find the "Anchor" chunks
    results.sort(key=lambda x: x[2], reverse=True)
    
    final_indices = set()
    top_anchors = results[:3] # Pick top 3 winners
    
    for idx, txt, score in top_anchors:
        final_indices.add(idx)
        
        # If score is high (>0.5), it's a direct hit (like a Header or Phrase match).
        # Automatically grab the NEXT 2 chunks to capture the full list/answer.
        if score > 0.5:
            if (idx + 1) in state.chunk_map: 
                final_indices.add(idx + 1)
                print(f"  [Expand] Auto-included Neighbor Chunk {idx+1}")
            if (idx + 2) in state.chunk_map: 
                final_indices.add(idx + 2)
                print(f"  [Expand] Auto-included Neighbor Chunk {idx+2}")

    # Sort indices so text appears in reading order (Chunk 9 -> 10 -> 11)
    sorted_indices = sorted(list(final_indices))
    
    final_context_list = []
    for idx in sorted_indices:
        txt = state.chunk_map[idx]
        final_context_list.append((txt, 0.0)) 

    if not final_context_list:
        return jsonify({"error": "No relevant data found."})

    # 5. Generate Response
    context_str = "\n\n".join([f"Source (Chunk {i}): {txt}" for i, (txt, _) in zip(sorted_indices, final_context_list)])
    
    def generate():
        # Send context metadata to UI
        # We limit to top 5 for UI display to prevent clutter
        context_data = [{"text": txt[:200]+"...", "score": 1.0} for txt, _ in final_context_list[:5]]
        yield json.dumps({"type": "context", "data": context_data}) + "\n"

        system_instruction = "You are a helpful AI assistant. Use ONLY the provided Context to answer."
        final_prompt = f"""{system_instruction}

Context:
{context_str}

Question: {query}

Answer:"""

        stream = ollama.chat(
            model=LANGUAGE_MODEL, # Uppercase constant
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

# if __name__ == '__main__':
#     app.run(debug=True, port=5000, threaded=True)
if __name__ == '__main__':
    # Automatically open the browser after 1.5 seconds
    def open_browser():
        webbrowser.open_new('http://127.0.0.1:5000/')

    print("Starting RAG Engine...")
    Timer(1.5, open_browser).start()
    
    # Disable debug mode (Debug mode crashes PyInstaller apps)
    app.run(port=5000, debug=False)