import fitz  # PyMuPDF
import pymupdf4llm # The Universal Markdown Parser

def parse_hybrid_pdf(pdf_path):
    print(f"Analyzing PDF (Universal + Skip Scanned): {pdf_path}")
    
    try:
        valid_page_indices = []
        
        # --- Step 1: Pre-Scan to find "Good" Pages ---
        # We open the file quickly to check each page's text content
        with fitz.open(pdf_path) as doc:
            total_pages = len(doc)
            for i, page in enumerate(doc):
                # Get raw text (fastest method)
                text_sample = page.get_text()
                
                # HEURISTIC: If a page has < 50 characters, we assume it is 
                # a scanned image or empty. We SKIP it.
                if len(text_sample.strip()) >= 50:
                    valid_page_indices.append(i)
                else:
                    print(f"  [!] Page {i + 1} skipped (Detected as Scanned/Empty).")
        
        # --- Step 2: Handle Edge Case (All Scanned) ---
        if not valid_page_indices:
            return "[ERROR] This entire document appears to be scanned images. No text found."

        print(f"  Processing {len(valid_page_indices)} of {total_pages} pages...")

        # --- Step 3: Run Universal Conversion on Valid Pages Only ---
        # pymupdf4llm accepts a 'pages' list to selectively convert content.
        # This gives you the Markdown tables & headers for ONLY the text pages.
        md_text = pymupdf4llm.to_markdown(pdf_path, pages=valid_page_indices)
        
        return md_text

    except Exception as e:
        return f"Error parsing PDF: {e}"

# --- Test Block ---
if __name__ == "__main__":
    import os
    pdf_file = r"D:\JK\RAG\RAG_APP\Tata Code Of Conduct.pdf"
    
    if os.path.exists(pdf_file):
        text = parse_hybrid_pdf(pdf_file)
        
        # Save as Markdown (.md)
        with open("universal_skipped.md", "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\n✅ Done! Saved to 'universal_skipped.md'")