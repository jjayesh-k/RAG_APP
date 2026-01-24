import pickle
import os

# --- CONFIGURATION ---
# Replace this with the actual path to your .pkl file
PKL_FILE_PATH = "cache/chunks_dfcf00ae04aed4dfad4e5fc0bc9d76ef.pkl" 
OUTPUT_TXT_FILE = "decoded_chunks.txt"

def decode_pickle_to_txt(pkl_path, output_path):
    # 1. Check if file exists
    if not os.path.exists(pkl_path):
        print(f"Error: The file '{pkl_path}' was not found.")
        return

    print(f"Reading pickle file: {pkl_path}...")

    try:
        # 2. Load the data from the pickle file
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        # 3. Check if it's a list (which your chunks usually are)
        if not isinstance(data, list):
            print(f"Warning: Expected a list, but got {type(data)}. Trying to process anyway...")

        # 4. Write to a text file
        with open(output_path, 'w', encoding='utf-8') as f_out:
            f_out.write(f"=== DECODED CONTENT OF {os.path.basename(pkl_path)} ===\n")
            f_out.write(f"Total Chunks Found: {len(data)}\n\n")

            for i, chunk in enumerate(data):
                header = f"--- CHUNK {i} ---"
                f_out.write(header + "\n")
                f_out.write(str(chunk)) # Convert to string just in case
                f_out.write("\n\n" + "="*40 + "\n\n") # Separator
        
        print(f"Success! Decoded content saved to: {output_path}")
        print(f"Found {len(data)} chunks.")

    except Exception as e:
        print(f"Failed to decode pickle file: {e}")

# --- RUN THE FUNCTION ---
if __name__ == "__main__":
    # Create a dummy file for testing if you don't have one yet
    # (Uncomment the lines below to create a test file if needed)
    # test_data = ["This is chunk 1.", "This is chunk 2."]
    # with open("test_chunks.pkl", "wb") as f:
    #     pickle.dump(test_data, f)
    # decode_pickle_to_txt("test_chunks.pkl", "output.txt")
    
    decode_pickle_to_txt(PKL_FILE_PATH, OUTPUT_TXT_FILE)