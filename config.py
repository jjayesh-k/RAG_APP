import os
import sys
import json

# --- 1. Define Defaults (Hardcoded Fallbacks) ---
DEFAULT_EMBED_MODEL = 'nomic-embed-text'
DEFAULT_LANG_MODEL = 'mistral:7b'
BATCH_SIZE = 50
# --- 2. Determine App Directory ---
# This handles the tricky part: finding the folder whether running as Python or .exe
if getattr(sys, 'frozen', False):
    # Running as compiled .exe
    APP_DIR = os.path.dirname(sys.executable)
else:
    # Running as script
    APP_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 3. Define Paths ---
CACHE_DIR = os.path.join(APP_DIR, "cache")
INDEX_CACHE = os.path.join(APP_DIR, "index_cache")
SETTINGS_PATH = os.path.join(APP_DIR, 'settings.json')

# --- 4. Dynamic Config Loader ---
final_embed_model = DEFAULT_EMBED_MODEL
final_lang_model = DEFAULT_LANG_MODEL

if os.path.exists(SETTINGS_PATH):
    try:
        with open(SETTINGS_PATH, 'r') as f:
            settings = json.load(f)
            # Only overwrite if the key exists in json
            final_embed_model = settings.get('embedding_model', final_embed_model)
            final_lang_model = settings.get('language_model', final_lang_model)
        print(f"✓ Loaded custom config: {final_lang_model} / {final_embed_model}")
    except Exception as e:
        print(f"⚠️ Error reading settings.json: {e}")

# --- 5. Export Constants (This is what app.py imports) ---
EMBEDDING_MODEL = final_embed_model
LANGUAGE_MODEL = final_lang_model