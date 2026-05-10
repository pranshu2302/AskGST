# AskGST — Eval Findings (Day 7)

## Test Set

- **30 hand-written queries**, 25 with gold chunk IDs and 5 deliberate expected-failure cases.
- **8 GST categories**: registration (6), rcm (6), other (4), itc (3), returns (3), composition (2), eway_bill (1), rates (3 — all expected-failure).
- **5 phrasing styles**: direct (13), casual (5), scenario (5), paraphrase (2), expected_failure (5).
- The expected-failure queries are excluded from recall calculation; they exist to surface known dataset weaknesses (fragmented rate-schedule PDFs, missing notifications/circulars, noisy form-heavy chunks).

## Headline Numbers (Recall@5)

| Configuration            | Recall@5  | Passed    |
|--------------------------|-----------|-----------|
| BM25 only                | 48.0%     | 12/25     |
| Vector only (BGE)        | 56.0%     | 14/25     |
| **Hybrid (RRF)**         | **72.0%** | **18/25** |
| Hybrid + Reranker        | 64.0%     | 16/25     |

## Findings

### 1. Hybrid retrieval substantially outperformed either retriever alone

Hybrid beat vector-only by +16 points and BM25-only by +24 points. The configurations have visibly complementary failure modes:

- **BM25 alone** missed 13 queries — most of them with conceptual or paraphrased phrasing (q5 "what does reverse charge mean", q12 "time of supply", q14 "conditions for ITC", q16 "GTA recipient", q20 "how often GSTR-3B"). All five of these were rescued by hybrid.
- **Vector alone** missed 11 queries — many with strong lexical anchors that BM25 catches (q1 "rent-a-cab", q10 "aggregate turnover" verbatim, q13 "motor vehicles", q22 "1 crore turnover"). Four of these were rescued by hybrid.
- **Hybrid uniquely caught** 6 queries that *both* single retrievers missed (q5, q12, q14, q16, q20, q22). This is RRF doing exactly what it's supposed to do: a query needs only to land in either retriever's top 20 to surface in the merged top 5.

Hybrid's only "loss" against a single retriever was q10 ("aggregate turnover"), where BM25 alone passed but hybrid didn't — vector noise pushed the gold chunk out of the merged top 5. One regression out of 25 queries is acceptable for a +16-point net gain.

### 2. The cross-encoder reranker net-regressed hybrid by 8 points

Hybrid+reranker scored 64% vs hybrid's 72%. Breakdown:

- **3 regressions**: q1 (rent-a-cab), q15 (Bangalore freelancer), q23 (TDS rate).
- **1 win**: q10 (aggregate turnover).
- **Net: −2 queries**, or −8 percentage points.

The regressions follow a consistent pattern. On q1, hybrid placed both gold chunks at ranks 2 and 4; the reranker pushed both out and replaced them with chunks containing the phrase "rent-a-cab" but in restrictive contexts (Section 17(5) blocked-credit lists rather than the FAQ Q&A). On q23, the gold was hybrid's rank-2 result and the reranker demoted it past rank 5 in favor of a tangentially-related sectoral FAQ. On q15, the reranker promoted closely-related cat18 sectoral FAQs but specifically dropped the one matching the gold label.

Pattern: the cross-encoder is *over-confident on lexical overlap with restrictive or near-duplicate chunks*. It re-orders RRF's good results based on surface similarity, which on this corpus often means picking a more-statutory or more-narrow chunk over the FAQ that actually answers the question.

**Caveat on statistical confidence.** With 25 ground-truth queries, a 2-query swing is suggestive but not conclusive — confidence intervals on either configuration are wide. The decision below is made on observed regressions plus the per-query mechanism analysis, not just the headline delta.

**Decision: drop the reranker from the V1 pipeline.** Re-evaluate on a 50–75 query set in V2 before reintroducing.

### 3. The 7 hybrid failures cluster into three patterns

**Pattern A — Gold labeling was too narrow (3 queries: q10, q11, q21).** On q21 ("annual return — who has to file"), hybrid surfaced FAQ chunks at p104 of cbic_faqs_v1/v2 that explicitly answer the question ("All taxpayers filing return in GSTR-1 to GSTR-3, other than ISDs, casual/non-resident taxpayers..."). My gold was Rule 80 statutory text from cgst_rules_2017_part_a_p84_c3, which is technically the source of truth but reads less naturally as an answer. q11 ("who registers regardless of turnover") has the same shape — the gold is Section 24 statutory text; hybrid retrieved sectoral_faq_cat18_q7 which directly references and quotes Section 24. **These aren't true retrieval failures — they're eval-set design failures.** The fix is to allow multiple acceptable gold chunks per query, including FAQ paraphrases of the same statutory rule.

**Pattern B — Genuine retrieval misses (3 queries: q2, q17, q19).** These are real failures.
- **q2** ("just started a small consultancy, 12 lakhs"): The gold is `gst_overview_2019_p12_c0`, which states the 20-lakh services threshold cleanly. Hybrid retrieved 5 sectoral/CBIC FAQ chunks, none of which contained the gold or its content. The overview chunk is short, well-formatted, and *should* have ranked — likely a vocabulary-mismatch issue ("yearly income" / "consultancy" vs the chunk's "aggregate turnover" / "suppliers of services" formal phrasing).
- **q17** ("when do I need to generate an e-way bill"): Gold is Rule 138 (`cgst_rules_2017_part_a_p138_c0`). Hybrid retrieved 4 chunks from the *same chapter* (pages 140, 149, 150, plus an EWB-01 form page from part_b) but missed page 138 itself. The chunk title "Information to be furnished prior to commencement of movement of goods" doesn't lexically match "when do I need to generate," and BGE embeddings didn't bridge the gap. RRF over-weighted the keyword-y near-misses.
- **q19** ("sell goods to a customer outside India"): Gold is `igst_act_2017_p5_c2`. Inspecting the chunk: the goods-export definition is the *last partial sentence* of an otherwise stopover/customs-frontier chunk (text ends mid-word: "(5) "export of goods" with its grammatical variat..."). This is a chunking-boundary failure, not a retrieval failure. The right answer literally falls across a chunk break.

**Pattern C — Multi-concept query, single gold (1 query: q29).** "What is the difference between exempt, nil-rated, and zero-rated supply?" The query needs to surface three definitions that live in different chunks (exempt in CGST Act, zero-rated in IGST Act, nil-rated implicitly across rate schedule). Single gold chunk was structurally insufficient; this is also an eval-set design issue.

### 4. Expected-failure queries diagnosed real chunking issues, not retrieval issues

Of the 5 expected-failure queries, 4 retrieved their *correct region* of the corpus in the top 5. Hybrid for q24 ("GST rate on biscuits") returned 4 rate-schedule pages in the top 5; for q26 ("HSN code for live horses") it returned the live-animals rate page at rank 1; for q25 (legal services rate) it returned the services rate schedule and reverse-charge services pages.

The reason these queries can't be answered isn't retrieval — it's that the rate-schedule chunks have **column-fragmented content from PDF extraction** (header rows separated from data rows, multi-column tables flattened into linearized text). Retrieval points at the right page; the page itself is unreadable for downstream LLM consumption.

This is an important finding: **Day 1's PDF extraction broke the rate schedules, not the retrieval pipeline.** Fixing the rate-schedule queries needs table-aware chunking, not retrieval tuning.

### 5. Per-category and per-phrasing breakdowns

**Hybrid failures by category** (failures / total in category):

| Category      | Failures   | Notes |
|---------------|------------|-------|
| registration  | 3 / 6      | All three are eval-design or vocabulary-mismatch issues |
| other         | 2 / 4      | q19 (chunk boundary), q29 (multi-concept) |
| eway_bill     | 1 / 1      | q17 — keyword/phrasing mismatch |
| returns       | 1 / 3      | q21 — gold-label too narrow |
| rcm           | 0 / 6      | Sectoral FAQ cat9 is high-quality; retrieval handles RCM well |
| itc           | 0 / 3      | All passed |
| composition   | 0 / 2      | Both passed |

RCM and ITC are the strongest categories — both are well-served by clean sectoral and CBIC FAQ chunks with explicit Q&A structure. Registration is the weakest, partly because the topic is genuinely fragmented across multiple sources (Section 22, Section 24, FAQ paraphrases, overview document) and partly because my gold labels favored statutory text over FAQ paraphrases.

**Hybrid failures by phrasing style:**

| Phrasing       | Failures   | Notes |
|----------------|------------|-------|
| direct         | 3 / 13     | 23% — lowest failure rate |
| casual         | 2 / 5      | 40% — q17 e-way bill, q19 export-of-goods |
| scenario       | 1 / 5      | 20% — q2 consultancy |
| paraphrase     | 1 / 2      | 50% — q29 multi-concept |

Casual and paraphrase queries fail more often, as expected. Scenario queries fared better than I anticipated — q9 (Amazon seller), q15 (freelance designer), q22 (1 crore trader), q30 (kirana shop) all passed, suggesting the BGE encoder handles natural-language scenarios reasonably well when the chunk contains an explicit Q&A.

## What This Means for V2

Three priorities, ranked by expected impact:

1. **Table-aware chunking for rate schedules and rule-form PDFs.** The expected-failure queries (q24–q26, q28) all retrieve the right region — the chunks themselves are unusable. Use `pdfplumber.extract_tables` to detect tables, then convert each row into a natural-language sentence ("HSN 0101: live asses, mules and hinnies — 12% GST"). Estimated impact: would unlock the 4 rate-related queries plus an unknown count of currently-passing-but-low-quality answers.

2. **Allow multiple acceptable gold chunks per query in the eval set.** Three of the seven hybrid failures (q10, q11, q21) are queries where retrieval found a defensible alternative answer that wasn't on my gold list. The fix is to expand each gold list to include FAQ paraphrases of statutory rules. Re-running the eval after this would likely move hybrid recall from 72% to ~84%. This is also the cheapest fix — no code changes, just relabeling.

3. **Larger eval set (50–75 queries) to revisit the reranker decision.** The 3-regression vs 1-win pattern is suggestive but the sample is small. A 50–75 query set with broader category coverage would either confirm the reranker is structurally net-negative on this corpus or reveal it as noise. Until then, V1 ships without it.

A fourth, lower-priority item: **index recent CBIC notifications and circulars** (deferred from Day 1). Would address q27-style queries directly. Lower priority because notifications change frequently and would need a refresh pipeline.

## What This Means for V1

- **Final pipeline: BGE + Qdrant + BM25 + RRF (no reranker).** 72% recall@5 on 25 ground-truth queries.
- **Documented limitations**: rate-schedule queries (PDF chunking, not retrieval), niche statutory text where queries don't share vocabulary with the source, multi-concept comparative queries.
- **Eval set is part of the deliverable.** queries.json, harness.py, and analyze.py all live in the repo. Re-running on a new ingest is one command.
- **Honest framing for the resume bullet**: "72% recall@5 on a 25-query ground-truth eval set; +16 points over vector-only baseline; cross-encoder reranker evaluated and deferred to V2 pending larger sample."
