"""
Failure analysis for eval results.

Loads data/eval/results.json, prints:
  - Recall@5 per config
  - Failures per config (query IDs)
  - Hybrid failures with retrieved chunk IDs
  - Reranker regressions (hybrid passed, hybrid_rerank failed)
  - Reranker wins (hybrid failed, hybrid_rerank passed)
  - Failure breakdown by category and phrasing_style
  - What expected-failure queries actually retrieved (for context)

Usage: python src/eval/analyze.py
"""
import json
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = REPO_ROOT / "data" / "eval" / "results.json"

CONFIGS = ["vector_only", "bm25_only", "hybrid", "hybrid_rerank"]


def main() -> None:
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    summary = results["summary"]
    per_query = results["per_query_results"]
    with_gold = [q for q in per_query if q["gold_chunk_ids"]]
    no_gold = [q for q in per_query if not q["gold_chunk_ids"]]

    # --- Header ---
    print("=" * 70)
    print("EVAL ANALYSIS")
    print("=" * 70)
    print(f"Total queries:         {summary['total_queries']}")
    print(f"Queries with gold:     {summary['queries_with_gold']}")
    print(f"Expected-failure:      {len(no_gold)} (excluded from recall)")

    # --- Recall@5 per config ---
    print("\nRecall@5 by config:")
    for cfg in CONFIGS:
        m = summary["metrics_per_config"][cfg]
        print(f"  {cfg:15s}: {m['recall_at_5']:.2%} ({m['passed']}/{m['total_with_gold']})")

    # --- Pass/fail matrix ---
    print("\n" + "=" * 70)
    print("PASS/FAIL MATRIX (queries with gold)")
    print("=" * 70)
    print(f"{'qid':5s} {'category':12s} {'phrasing':12s} {'V':>3s} {'B':>3s} {'H':>3s} {'HR':>3s}  query")
    print("-" * 70)
    for q in with_gold:
        flags = ["✓" if q["retrieved"][c]["passed"] else "✗" for c in CONFIGS]
        preview = q["query"][:40] + ("..." if len(q["query"]) > 40 else "")
        print(
            f"{q['query_id']:5s} {q['category']:12s} {q['phrasing_style']:12s} "
            f"{flags[0]:>3s} {flags[1]:>3s} {flags[2]:>3s} {flags[3]:>3s}  {preview}"
        )

    # --- Per-config failure lists ---
    print("\n" + "=" * 70)
    print("FAILURES PER CONFIG (query IDs)")
    print("=" * 70)
    for cfg in CONFIGS:
        fails = [q["query_id"] for q in with_gold if not q["retrieved"][cfg]["passed"]]
        print(f"  {cfg:15s} ({len(fails)}): {fails}")

    # --- Hybrid failures: full detail ---
    hybrid_fails = [q for q in with_gold if not q["retrieved"]["hybrid"]["passed"]]
    print("\n" + "=" * 70)
    print(f"HYBRID FAILURES ({len(hybrid_fails)} queries)")
    print("=" * 70)
    for q in hybrid_fails:
        print(f"\n[{q['query_id']}] {q['query']}")
        print(f"  category={q['category']} | phrasing={q['phrasing_style']}")
        print(f"  gold:     {q['gold_chunk_ids']}")
        print(f"  hybrid retrieved:")
        for i, cid in enumerate(q["retrieved"]["hybrid"]["chunk_ids"], 1):
            print(f"    {i}. {cid}")
        # Did any other config get this right?
        rescuers = [c for c in CONFIGS if c != "hybrid" and q["retrieved"][c]["passed"]]
        if rescuers:
            print(f"  ✓ Rescued by: {rescuers}")
        else:
            print(f"  ✗ No config retrieved gold (all-fail)")

    # --- Reranker regressions ---
    regressions = [
        q for q in with_gold
        if q["retrieved"]["hybrid"]["passed"]
        and not q["retrieved"]["hybrid_rerank"]["passed"]
    ]
    print("\n" + "=" * 70)
    print(f"RERANKER REGRESSIONS ({len(regressions)} queries — hybrid passed, hybrid_rerank failed)")
    print("=" * 70)
    for q in regressions:
        print(f"\n[{q['query_id']}] {q['query']}")
        print(f"  category={q['category']} | phrasing={q['phrasing_style']}")
        print(f"  gold: {q['gold_chunk_ids']}")
        print(f"  hybrid top 5:")
        for i, cid in enumerate(q["retrieved"]["hybrid"]["chunk_ids"], 1):
            mark = "  ← GOLD" if cid in q["gold_chunk_ids"] else ""
            print(f"    {i}. {cid}{mark}")
        print(f"  hybrid_rerank top 5:")
        for i, cid in enumerate(q["retrieved"]["hybrid_rerank"]["chunk_ids"], 1):
            mark = "  ← GOLD" if cid in q["gold_chunk_ids"] else ""
            print(f"    {i}. {cid}{mark}")

    # --- Reranker wins ---
    wins = [
        q for q in with_gold
        if not q["retrieved"]["hybrid"]["passed"]
        and q["retrieved"]["hybrid_rerank"]["passed"]
    ]
    print("\n" + "=" * 70)
    print(f"RERANKER WINS ({len(wins)} queries — hybrid failed, hybrid_rerank passed)")
    print("=" * 70)
    for q in wins:
        print(f"\n[{q['query_id']}] {q['query']}")
        print(f"  category={q['category']} | phrasing={q['phrasing_style']}")
        print(f"  gold: {q['gold_chunk_ids']}")
        gold_rank = next(
            (i for i, cid in enumerate(q["retrieved"]["hybrid_rerank"]["chunk_ids"], 1)
             if cid in q["gold_chunk_ids"]),
            None,
        )
        print(f"  rerank surfaced gold at rank: {gold_rank}")

    # --- Breakdown by category / phrasing for hybrid failures ---
    print("\n" + "=" * 70)
    print("HYBRID FAILURE BREAKDOWN")
    print("=" * 70)
    by_cat = Counter(q["category"] for q in hybrid_fails)
    by_phr = Counter(q["phrasing_style"] for q in hybrid_fails)
    cat_total = Counter(q["category"] for q in with_gold)
    phr_total = Counter(q["phrasing_style"] for q in with_gold)

    print("\nBy category (failures / total):")
    for cat, n in sorted(cat_total.items(), key=lambda kv: -kv[1]):
        f = by_cat.get(cat, 0)
        print(f"  {cat:15s}: {f}/{n}")

    print("\nBy phrasing style (failures / total):")
    for phr, n in sorted(phr_total.items(), key=lambda kv: -kv[1]):
        f = by_phr.get(phr, 0)
        print(f"  {phr:15s}: {f}/{n}")

    # --- Expected-failure queries: what did they retrieve? ---
    print("\n" + "=" * 70)
    print(f"EXPECTED-FAILURE QUERIES — what hybrid retrieved")
    print("=" * 70)
    print("(These have no gold; included for diagnostic context.)")
    for q in no_gold:
        print(f"\n[{q['query_id']}] {q['query']}")
        print(f"  hybrid top 5:")
        for i, cid in enumerate(q["retrieved"]["hybrid"]["chunk_ids"], 1):
            print(f"    {i}. {cid}")


if __name__ == "__main__":
    main()
