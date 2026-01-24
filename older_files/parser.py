import pdfplumber
import pytesseract
from pdf2image import convert_from_path
import os

# --- PATH CONFIGURATION (Your specific paths) ---
POPPLER_BIN_PATH = r"E:\Projects_main\TATA\safety\Release-24.02.0-0\poppler-24.02.0\Library\bin"
# Update this if using a portable Tesseract
TESSERACT_CMD_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe" 

# Setup Tesseract
if os.path.exists(TESSERACT_CMD_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD_PATH

def parse_hybrid_pdf(pdf_path):
    print(f"Analyzing Hybrid PDF: {pdf_path}")
    full_text = ""
    
    # We open the file with pdfplumber to loop through pages
    with pdfplumber.open(pdf_path) as pdf:
        # print("------------>", type(pdf.pages))
        total_pages = len(pdf.pages)
        print(f"Document has {total_pages} pages.")
        
        for i, page in enumerate(pdf.pages):
            page_num = i + 1
            # print("---------------->", page)
            # 1. Try Digital Extraction first
            extracted = page.extract_text(layout=True)
            
            # 2. Decision Logic: Is this page Scanned?
            # We assume it's scanned if extracted text is None or very short (< 15 chars)
            if not extracted or len(extracted.strip()) < 15:
                print(f"Page {page_num}: No digital text found. Running OCR...")
                
                try:
                    # Convert JUST this specific page to an image (Save RAM)
                    # first_page and last_page are 1-based indices
                    images = convert_from_path(
                        pdf_path, 
                        first_page=page_num, 
                        last_page=page_num,
                        poppler_path=POPPLER_BIN_PATH
                    )
                    
                    # Run OCR on the image
                    if images:
                        ocr_text = pytesseract.image_to_string(images[0])
                        full_text += f"\n--- Page {page_num} (OCR) ---\n{ocr_text}\n"
                    
                except Exception as e:
                    print(f"OCR Failed on page {page_num}: {e}")
            
            else:
                # It is a digital page
                # print(f"=Page {page_num}: Digital text extracted.")
                full_text += f"\n--- Page {page_num} ---\n{extracted}\n"

    return full_text

# --- RUN IT ---
if __name__ == "__main__":
    pdf_file = r"D:\JK\RAG\RAG_APP\Tata Code Of Conduct.pdf"
    
    if os.path.exists(pdf_file):
        text = parse_hybrid_pdf(pdf_file)
        
        # Save to file
        with open("hybrid_output.txt", "w", encoding="utf-8") as f:
            f.write(text)
        print("\n Done! Saved to 'hybrid_output.txt'")