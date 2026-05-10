import sys
from pathlib import Path
import json

# Add project root to sys.path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ingest.pdf_loader import load_pdfs
from src.ingest.chunker import chunk_records
from src.ingest.html_scraper import load_sectoral_faqs

def main():
    # Step 1: Load PDF pages
    print("Loading PDF pages...")
    pdf_pages = load_pdfs("data/raw")
    print(f"Loaded {len(pdf_pages)} PDF page records.")

    # Step 2: Chunk PDF pages
    print("Chunking PDF pages...")
    pdf_chunks = chunk_records(pdf_pages)
    print(f"Chunked into {len(pdf_chunks)} PDF chunks.")

    # Step 3: Load HTML sectoral FAQs
    print("Loading sectoral FAQ Q&A...")
    faq_records = load_sectoral_faqs("data/raw/faqs/sectoral_faqs.html")
    print(f"Loaded {len(faq_records)} sectoral FAQ Q&A records.")

    # Step 4: Combine all records
    all_chunks = pdf_chunks + faq_records
    print(f"\nTotal combined chunks: {len(all_chunks)}")

    # Step 5: Save as JSON
    output_path = Path("data/processed/chunks.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved {len(all_chunks)} chunks to {output_path}")
    print(f"  - PDF chunks: {len(pdf_chunks)}")
    print(f"  - Sectoral FAQ Q&A: {len(faq_records)}")
    print(f"File size: {size_mb:.1f} MB")

if __name__ == "__main__":
    main()
