import ollama
import re
import numpy as np


BATCH_SIZE = 50
EMBEDDING_MODEL = 'nomic-embed-text'
# MIN_CHUNK_SIZE = 500
MAX_CHUNK_SIZE = 1500
SIMILARITY_THRESHOLD = 0.75

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