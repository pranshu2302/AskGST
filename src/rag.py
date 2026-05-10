
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from langchain_core.messages import SystemMessage, HumanMessage
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.client import get_llm
from src.llm.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid import HybridRetriever


COLLECTION_NAME = "askgst_chunks"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
TOP_K = 5
CHUNKS_PATH = "data/processed/chunks.json"


# Cache for expensive resources — initialized on first call
_model = None
_client = None
_llm = None
_retriever = None  # NEW: hybrid retriever (wraps bm25 + vector + qdrant)


def _get_components():
    """Lazy-load and cache the model, Qdrant client, LLM, and hybrid retriever."""
    global _model, _client, _llm, _retriever
    if _model is None:
        print("Loading BGE model...")
        _model = SentenceTransformer(MODEL_NAME)
    if _client is None:
        _client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    if _llm is None:
        _llm = get_llm()
    if _retriever is None:
        print("Building BM25 index...")
        bm25_store = BM25Store(CHUNKS_PATH)
        _retriever = HybridRetriever(
            bm25_store=bm25_store,
            vector_model=_model,
            qdrant_client=_client
        )
    return _retriever, _llm


# (Deleted: retrieve_chunks — replaced by hybrid retriever)


def format_context(hits):
    """
    Turn hybrid retrieval results (list of (chunk_dict, rrf_score) tuples) 
    into a numbered context string for the prompt.
    """
    parts = []
    for i, (chunk, _score) in enumerate(hits, start=1):
        source = chunk.get("source", "unknown")
        page = chunk.get("page")
        content = chunk.get("content", "")
        if page is not None:
            label = f"[Source {i}] {source}, page {page}"
        else:
            category = chunk.get("category", "")
            if category:
                label = f"[Source {i}] {source} ({category})"
            else:
                label = f"[Source {i}] {source}"
        parts.append(f"{label}:\n{content}")
    return "\n\n".join(parts)


def _build_sources(hits):
    """Build the structured sources list (shared by streaming + non-streaming paths)."""
    sources = []
    for i, (chunk, rrf_score) in enumerate(hits, start=1):
        sources.append({
            "index": i,
            "source": chunk.get("source"),
            "source_type": chunk.get("source_type"),  # for UI badge
            "page": chunk.get("page"),
            "id": chunk.get("id"),
            "score": round(rrf_score, 4),
            "content_preview": chunk.get("content", "")[:200]
        })
    return sources


def answer_question(question):
    """
    Full RAG pipeline: hybrid retrieve → format → prompt → LLM → return structured answer.
    """
    retriever, llm = _get_components()
    # Step 1+2: hybrid retrieval (top 5 from BM25+vector merged via RRF)
    hits = retriever.search(question, top_k=TOP_K, candidates_per_retriever=20)
    # Step 3: format context for the prompt
    context_str = format_context(hits)
    # Step 4: build messages
    user_message = USER_PROMPT_TEMPLATE.format(
        context=context_str,
        question=question
    )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message)
    ]
    # Step 5: call LLM
    response = llm.invoke(messages)
    answer_text = response.content
    # Build sources list
    sources = _build_sources(hits)
    return {
        "question": question,
        "answer": answer_text,
        "sources": sources
    }


def answer_question_stream(question):
    """
    Streaming version of answer_question. Yields events:
      {"event": "sources", "sources": [...]}   — once, before any LLM tokens
      {"event": "token", "text": "..."}         — many times, as LLM streams
      {"event": "done"}                         — once, at the end

    The original answer_question is untouched and still used by the eval harness.
    """
    retriever, llm = _get_components()

    # Step 1+2: hybrid retrieval (same as answer_question)
    hits = retriever.search(question, top_k=TOP_K, candidates_per_retriever=20)

    # Yield sources FIRST so the UI can render them while the LLM warms up
    yield {"event": "sources", "sources": _build_sources(hits)}

    # Step 3: format context
    context_str = format_context(hits)

    # Step 4: build messages
    user_message = USER_PROMPT_TEMPLATE.format(
        context=context_str,
        question=question
    )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message)
    ]

    # Step 5: stream LLM tokens
    for chunk in llm.stream(messages):
        # Some chunks have empty .content (e.g. role markers); skip them
        if chunk.content:
            yield {"event": "token", "text": chunk.content}

    yield {"event": "done"}


if __name__ == "__main__":
    test_questions = [
        "What is the GST registration threshold for small businesses?",
        "Can I claim input tax credit on rent-a-cab services?",
        "What is reverse charge mechanism?"
    ]
    for q in test_questions:
        print("=" * 80)
        print(f"Q: {q}")
        print("=" * 80)
        result = answer_question(q)
        print(f"\nA: {result['answer']}")
        print(f"\nSources used:")
        for src in result["sources"]:
            print(f"  [{src['index']}] {src['source']} (page {src['page']}) — score {src['score']}")
        print()
