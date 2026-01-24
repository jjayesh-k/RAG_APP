from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_markdown(md_text):
    """
    Optimized chunker for Markdown files.
    Respects Headers (#, ##) and Lists (-) to keep context intact.
    """
    
    # 1. Define separators specific to Markdown
    # This list tells the code: "Try to split by H1 first. If that's too big, try H2..."
    markdown_separators = [
        "\n# ",      # Level 1 Header
        "\n## ",     # Level 2 Header
        "\n### ",    # Level 3 Header
        "\n#### ",   # Level 4 Header
        "\n- ",      # Bullet points (Keep lists together if possible)
        "\n\n",      # Standard Paragraph break
        "\n",        # Line break
        ". ",        # Sentence break
        " "          # Word break
    ]
    
    # 2. Initialize the Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,       # Target size (characters)
        chunk_overlap=200,     # Context overlap (prevents cutoff issues)
        separators=markdown_separators,
        is_separator_regex=False
    )
    
    # 3. Create Chunks
    chunks = text_splitter.create_documents([md_text])
    
    return [chunk.page_content for chunk in chunks]

# --- Test Block ---
if __name__ == "__main__":
    import os
    
    # Point this to your generated .md file
    input_file = "universal_skipped.md" 
    
    if os.path.exists(input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            raw_md = f.read()
            
        print(f"Reading {len(raw_md)} characters...")
        
        chunk_list = chunk_markdown(raw_md)
        
        print(f"✅ Created {len(chunk_list)} chunks.")
        print(f"   Average Chunk Size: {sum(len(c) for c in chunk_list) / len(chunk_list):.0f} chars")
        
        # Save for inspection
        with open("chunked_data.txt", "w", encoding="utf-8") as f:
            for i, c in enumerate(chunk_list):
                f.write(f"--- CHUNK {i+1} ---\n{c}\n\n")
        print("   Saved preview to 'chunked_data.txt'")
    else:
        print(f"❌ File '{input_file}' not found. Run your parser first!")