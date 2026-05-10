from bs4 import BeautifulSoup
import sys

def load_sectoral_faqs(html_path):
    """
    Parse the CBIC sectoral FAQ HTML, extract one record per Q&A pair.
    Returns a list of dicts.
    """
    records = []
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
        table = soup.find("table", id="myTable")
        if not table:
            print("ERROR: Table with id='myTable' not found.", file=sys.stderr)
            return records
        rows = table.find_all("tr")
        for row in rows:
            try:
                position = row.get("position", "")
                is_titlehead = row.get("id") == "titlehead"
                if not position.startswith("cat") or is_titlehead:
                    continue
                cells = row.find_all("td")
                if len(cells) < 3:
                    continue
                q_num = cells[0].get_text(strip=True)
                question = cells[1].get_text(strip=True)
                answer = cells[2].get_text(strip=True)
                if not q_num.isdigit() or not question or not answer:
                    continue
                category = position
                content = f"Q: {question}\nA: {answer}"
                record = {
                    "id": f"sectoral_faq_{category}_q{q_num}",
                    "source": "sectoral_faqs.html",
                    "source_type": "sectoral_faq",
                    "page": None,
                    "content": content,
                    "char_count": len(content),
                    "category": category
                }
                records.append(record)
            except Exception as e:
                print(f"ERROR parsing row: {type(e).__name__}: {e}", file=sys.stderr)
                continue
    except Exception as e:
        print(f"ERROR opening/parsing HTML: {type(e).__name__}: {e}", file=sys.stderr)
    return records


if __name__ == "__main__":
    html_path = "data/raw/faqs/sectoral_faqs.html"
    records = load_sectoral_faqs(html_path)
    print(f"Total Q&A records: {len(records)}")
    if records:
        print("\nFirst record:")
        print(records[0])
        print("\nMiddle record:")
        print(records[len(records)//2])
        print("\nLast record:")
        print(records[-1])
    print(f"\nTotal: {len(records)} Q&A records extracted from {html_path}")
