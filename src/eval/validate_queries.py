"""
Validate eval/queries.json against chunks.json.
"""
import json
from collections import Counter

with open("data/processed/chunks.json") as f:
    chunks = json.load(f)
chunk_ids = {c["id"] for c in chunks}

with open("src/eval/queries.json") as f:
    queries = json.load(f)

print(f"Total queries: {len(queries)}")
print("Categories:", Counter(q["category"] for q in queries))
print("Phrasing styles:", Counter(q["phrasing_style"] for q in queries))

invalid = []
for q in queries:
    for cid in q.get("gold_chunk_ids", []):
        if cid not in chunk_ids:
            invalid.append((q["id"], cid))

if invalid:
    print(f"\n⚠️ INVALID chunk IDs ({len(invalid)}):")
    for qid, cid in invalid:
        print(f"  {qid} -> {cid}")
else:
    print("\n✅ All chunk IDs valid.")

no_gold = [q["id"] for q in queries if not q.get("gold_chunk_ids")]
print(f"\nQueries with no gold: {len(no_gold)} (expected: 5)")
print(f"  → {no_gold}")