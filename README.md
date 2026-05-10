# AskGST — RAG Q&A System for Indian GST

> 🚧 **Work in progress.** Currently in active development. Live demo and full documentation coming soon.

## What It Does

AskGST is a Retrieval-Augmented Generation (RAG) system that answers plain-English questions about Indian Goods and Services Tax (GST), grounded in official documents published by the Central Board of Indirect Taxes and Customs (CBIC).

Built for small business owners who want quick, sourced answers to GST questions without wading through hundreds of pages of legal text.

### Example Queries

- "Do I need to register for GST if my turnover is ₹15 lakhs?"
- "What's the time of supply for services under reverse charge?"
- "Can I claim input tax credit on rent-a-cab services?"
- "When is GSTR-3B due for monthly filers?"

## Architecture

User question
→ Hybrid retrieval (BM25 + dense vectors over Qdrant)
→ Cross-encoder re-ranker (BAAI/bge-reranker-base)
→ LLM (Google Gemini 2.5 Flash) with retrieved context
→ Answer with source citations

## Tech Stack

- **Language:** Python 3.11+
- **LLM:** Google Gemini 2.5 Flash
- **Embeddings:** BAAI/bge-small-en-v1.5
- **Vector DB:** Qdrant (local Docker)
- **Framework:** LangChain
- **UI:** Streamlit
- **Hosting:** Hugging Face Spaces (planned)

## Data Sources

All source documents are official publications from CBIC and the GST Council:

- CGST Act, 2017 (with amendments)
- CGST Rules, 2017
- IGST Act, 2017
- E-way Bill Rules
- Anti-Profiteering Rules
- CBIC FAQs (1st & 2nd editions)
- CBIC Sectoral FAQs
- GST Rate Schedules (Goods & Services)

## Setup (Development)

> Setup instructions will be finalized when V1 is complete.

```bash
# Clone the repo
git clone https://github.com/<username>/askgst.git
cd askgst

# Create venv and install deps (instructions TBD)
```

## Status

| Phase | Status |
|---|---|
| Data acquisition | ✅ Complete |
| Ingestion pipeline | 🚧 In progress |
| Vector indexing | ⏳ Planned |
| RAG pipeline (V1) | ⏳ Planned |
| Hybrid retrieval + Re-ranker (V2) | ⏳ Planned |
| Evaluation harness | ⏳ Planned |
| Deployment | ⏳ Planned |

## Limitations

- Data current as of [date TBD]; not updated for notifications issued after that
- Not a substitute for professional tax advice
- V1 focuses on procedural and conceptual queries; rate-specific queries may have lower accuracy

## License

MIT (planned)

## Author

Pranshu Singhal