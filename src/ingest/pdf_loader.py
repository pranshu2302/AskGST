from pathlib import Path
from pypdf import PdfReader
import sys
import random
import logging
logging.getLogger("pypdf").setLevel(logging.ERROR)


def load_pdfs(raw_dir):
    """
    Walk raw_dir recursively, extract text from all PDFs page-by-page,
    return a list of page-level records.
    """
    raw_dir = Path(raw_dir)
    pdf_paths = sorted(raw_dir.rglob("*.pdf"))
    pages = []
    errors = 0
    total_pdfs = 0
    total_pages = 0
    for pdf_path in pdf_paths:
        try:
            reader = PdfReader(str(pdf_path))
            num_pages = len(reader.pages)
            file_pages = 0
            for i, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text is None or text.strip() == "":
                    continue
                record = {
                    "id": f"{pdf_path.stem}_p{i}",
                    "source": pdf_path.name,
                    "source_type": pdf_path.parent.name,
                    "page": i,
                    "content": text,
                    "char_count": len(text)
                }
                pages.append(record)
                file_pages += 1
            print(f"Processed: {pdf_path.name} — {file_pages}/{num_pages} pages")
            total_pdfs += 1
            total_pages += file_pages
        except Exception as e:
            print(f"ERROR processing {pdf_path.name}: {type(e).__name__}: {e}", file=sys.stderr)
            errors += 1
    print(f"\nTotal: {len(pdf_paths)} PDFs found, "
      f"{len(pdf_paths) - errors} processed successfully, "
      f"{len(pages)} pages extracted, "
      f"{errors} error{'s' if errors != 1 else ''}")
    return pages


if __name__ == "__main__":
    pages = load_pdfs("data/raw")
    sample = random.sample(pages, 5)
    for r in sample:
        print(f"\n--- {r['id']} ({r['source_type']}) ---")
        print(r['content'][:300])
