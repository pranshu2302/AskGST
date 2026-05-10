from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_records(page_records):
    """
    Split each page record's content into smaller chunks.
    Returns a new list of chunk records.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = []
    for page_record in page_records:
        chunk_texts = splitter.split_text(page_record["content"])
        for chunk_index, chunk_text in enumerate(chunk_texts):
            if not chunk_text or len(chunk_text.strip()) < 50:
                continue
            chunk = {
                "id": f"{page_record['id']}_c{chunk_index}",
                "source": page_record["source"],
                "source_type": page_record["source_type"],
                "page": page_record["page"],
                "content": chunk_text,
                "char_count": len(chunk_text)
            }
            chunks.append(chunk)
    return chunks

if __name__ == "__main__":
    from pdf_loader import load_pdfs

    print("Loading PDFs...")
    pages = load_pdfs("data/raw")

    print(f"\nChunking {len(pages)} pages...")
    chunks = chunk_records(pages)

    chunk_sizes = [c["char_count"] for c in chunks]
    print(f"\nTotal: {len(pages)} page records → {len(chunks)} chunks")
    if chunk_sizes:
        print(f"Avg chunk size: {sum(chunk_sizes)//len(chunk_sizes)} chars")
        print(f"Min chunk size: {min(chunk_sizes)} chars")
        print(f"Max chunk size: {max(chunk_sizes)} chars")
    if chunks:
        print(f"\nFirst chunk:")
        print(chunks[0])
        print(f"\nMiddle chunk:")
        print(chunks[len(chunks)//2])
        print(f"\nLast chunk:")
        print(chunks[-1])
    tiny_chunks = [c for c in chunks if c["char_count"] < 50]
    print(f"\nChunks under 50 chars: {len(tiny_chunks)}")
    for c in tiny_chunks[:10]:
        print(f"  [{c['char_count']} chars] {c['id']}: {repr(c['content'])}")
