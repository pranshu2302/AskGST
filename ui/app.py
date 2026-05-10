"""
Streamlit UI for AskGST — minimal demo interface with streaming answers,
suggested questions, polished source display, and visible pre-warming.
"""
import sys
from pathlib import Path
import streamlit as st

# Add project root to path so we can import src.rag
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.rag import answer_question_stream, _get_components

st.set_page_config(
    page_title="AskGST",
    page_icon="🧾",
    layout="centered"
)


# --- Title row with status badge ---
st.title("AskGST")
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("AI-powered Q&A on Indian GST")
with col2:
    if st.session_state.get("components_loaded"):
        st.markdown("🟢 **Ready**")
    else:
        st.markdown("🟡 *Loading...*")


# --- Pre-warming with visible feedback ---
@st.cache_resource
def initialize_components():
    """Lazy-load model, Qdrant client, and LLM. Cached across reruns."""
    return _get_components()


if "components_loaded" not in st.session_state:
    with st.status("🔄 Initializing AskGST...", expanded=True) as status:
        st.write("Loading BGE embedding model...")
        st.write("Building BM25 index over 4,392 chunks...")
        st.write("Connecting to Qdrant vector database...")
        st.write("Connecting to Gemini API...")
        initialize_components()
        status.update(label="✅ AskGST ready", state="complete", expanded=False)
    st.session_state.components_loaded = True
    st.rerun()  # rerun so the title-row badge flips to 🟢 Ready
else:
    initialize_components()  # cached, instant


# --- Source-type badges (used in the sources expander) ---
SOURCE_TYPE_LABELS = {
    "acts": "📜 Act",
    "faqs": "❓ FAQ",
    "sectoral_faq": "❓ Sectoral FAQ",
    "rates": "💰 Rate Schedule",
    "overview": "📘 Overview",
}


# --- Suggested questions (deliberately picked from passing eval queries) ---
EXAMPLE_QUERIES = [
    "What is the GST registration threshold?",
    "Can I claim ITC on rent-a-cab services?",
    "What is reverse charge mechanism?",
    "How does the composition scheme work?",
]

st.markdown("**Try a sample question:**")
selected_example = None
cols = st.columns(len(EXAMPLE_QUERIES))
for i, eq in enumerate(EXAMPLE_QUERIES):
    if cols[i].button(eq, key=f"example_{i}", use_container_width=True):
        selected_example = eq


# --- Text input ---
typed_query = st.text_input(
    "Ask a question about GST:",
    placeholder="e.g., What is the GST registration threshold?"
)

ask_clicked = st.button("Ask")

# Decide whether to run: either a sample button was clicked, or the user typed
# something and pressed Ask. The sample-button path uses its own query string.
query = selected_example or typed_query
should_run = bool(selected_example) or (ask_clicked and bool(typed_query))


if should_run and query:
    sources = None
    answer_buffer = ""

    # Phase 1: kick off retrieval, pull the sources event
    with st.spinner("Searching GST documents..."):
        gen = answer_question_stream(query)
        first_event = next(gen)
        if first_event["event"] == "sources":
            sources = first_event["sources"]

    # Phase 2: render answer area, stream tokens into it
    st.markdown("### Answer")
    answer_placeholder = st.empty()
    for event in gen:
        if event["event"] == "token":
            answer_buffer += event["text"]
            answer_placeholder.markdown(answer_buffer)
        elif event["event"] == "done":
            break

    # Phase 3: render sources after the answer is complete
    if sources:
        with st.expander(f"📚 Sources used ({len(sources)})", expanded=False):
            for src in sources:
                with st.container(border=True):
                    source_type = src.get("source_type") or ""
                    badge = SOURCE_TYPE_LABELS.get(source_type, source_type or "Source")
                    page_str = f"Page {src['page']}" if src["page"] else ""

                    # Header row: badge + filename on the left, score on the right
                    col_h1, col_h2 = st.columns([4, 1])
                    with col_h1:
                        header = f"**[{src['index']}]** {badge} · `{src['source']}`"
                        if page_str:
                            header += f" · {page_str}"
                        st.markdown(header)
                    with col_h2:
                        st.caption(f"Score: {src['score']}")

                    # Preview, italicized
                    preview = src["content_preview"]
                    if len(preview) >= 200:
                        preview = preview.rsplit(" ", 1)[0] + "..."
                    st.markdown(f"*{preview}*")


st.markdown("---")
st.caption(
    "AskGST is a RAG-based Q&A demo over official Indian GST documents. "
    "Not a substitute for professional tax advice. "
    "[GitHub](https://github.com/[your-username]/askgst)"
)