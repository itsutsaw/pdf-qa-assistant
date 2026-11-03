"""Run with: streamlit run app.py"""

from hashlib import sha256
import os
from pathlib import Path

import streamlit as st
from providers import (CHAT_MODEL, EMBEDDING_MODEL, GeminiEmbeddings, INDEX_VERSION, error_message,
                       make_model, list_available_models)

from rag import PDFError, PDFIndex, answer_question, read_pdf, split_pages
from usage import DAILY_LIMITS, DEMO_MAX_CHUNKS, SESSION_LIMITS, DailyLimit

st.set_page_config(page_title="PageCite · PDF Assistant", page_icon="📖", layout="wide")
st.markdown("""<style>
.block-container {max-width: 1100px; padding-top: 2.4rem;}
[data-testid="stSidebar"] {background-color: #f1f5f3;}
h1 {letter-spacing: -1.5px;}
</style>""", unsafe_allow_html=True)


def reset_document():
    for key in ("index", "parsed", "messages", "indexed_identity"):
        st.session_state.pop(key, None)


def show_error(exc):
    if isinstance(exc, (PDFError, ValueError)):
        st.error(str(exc))
    else:
        st.error(error_message(exc))


def demo_api_key() -> str:
    """The site owner's key from Streamlit secrets or the environment; never shown to visitors."""
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:  # No secrets file exists, e.g. when running locally.
        key = ""
    return (key or os.environ.get("GEMINI_API_KEY", "")).strip()


@st.cache_resource
def daily_limit() -> DailyLimit:
    # cache_resource returns one shared object to every visitor on this server.
    return DailyLimit(DAILY_LIMITS)


def use_demo_quota(kind: str) -> str | None:
    """Count one use of the demo key; return a message instead if a limit is reached."""
    used = st.session_state.setdefault("demo_usage", {"pdfs": 0, "questions": 0})
    if used[kind] >= SESSION_LIMITS[kind]:
        label = "PDFs" if kind == "pdfs" else "questions"
        return (f"You've reached this demo's limit of {SESSION_LIMITS[kind]} {label} per visit. "
                "Add your own Gemini API key in the sidebar to continue.")
    if not daily_limit().take(kind):
        return ("The demo has reached today's shared usage limit. Try again tomorrow, "
                "or add your own Gemini API key in the sidebar.")
    used[kind] += 1
    return None


def check_demo_pdf(passages: int):
    """Raise PDFError if the demo key should not embed this PDF. Each passage is one API request."""
    if passages > DEMO_MAX_CHUNKS:
        raise PDFError(f"This PDF has {passages} passages; the demo allows up to {DEMO_MAX_CHUNKS} "
                       "(about 40–60 pages of notes). Use a shorter PDF, or add your own Gemini API key.")
    if limit_message := use_demo_quota("pdfs"):
        raise PDFError(limit_message)


def own_key_input() -> str:
    entered = st.text_input("Gemini API key", type="password", help="Used only in this session.")
    st.link_button("Get a Gemini API key", "https://aistudio.google.com/apikey")
    st.caption("For no-charge use, choose a Free tier project in AI Studio and leave paid billing disabled.")
    return entered.strip()


def render_answer(message):
    # Render provider text as plain text to avoid document-supplied remote images/HTML.
    st.text(message["text"])
    if message.get("sources"):
        page_numbers = sorted({source["page"] for source in message["sources"]})
        st.caption("Source pages: " + ", ".join(map(str, page_numbers)))
        for number, source in enumerate(message["sources"], start=1):
            with st.expander(f"Source {number} · Page {source['page']}"):
                st.text(source["quote"])
                st.caption("Retrieved passage")
                st.text(source["passage"])


with st.sidebar:
    st.markdown("### 📖 PageCite")
    st.caption("A little clarity for long documents.")
    st.divider()
    st.markdown("**1 · Connect your model**")
    demo_key = demo_api_key()
    if demo_key:
        st.caption("Ready to use: this demo includes a Gemini key with shared daily limits. "
                   "Please don't upload private or confidential documents.")
        with st.expander("Use your own API key instead"):
            own_key = own_key_input()
    else:
        own_key = own_key_input()
    using_demo = bool(demo_key) and not own_key
    api_key = own_key or demo_key
    st.caption("PDF text and questions are sent to Google. Free-tier limits apply; paid projects may incur charges. Free-tier content may be used to improve Google products.")
    key_id = sha256(api_key.encode()).hexdigest() if api_key else None
    if st.session_state.get("model_key_id") != key_id:
        for name in ("available_models", "chat_model_choice", "embedding_model_choice"):
            st.session_state.pop(name, None)
        st.session_state["model_key_id"] = key_id
        reset_document()
    chat_model = embedding_model = None
    if using_demo:
        # Fixed models avoid a model-list request per visitor.
        chat_model, embedding_model = CHAT_MODEL, EMBEDDING_MODEL
    else:
        if st.button("Load available models", disabled=not api_key):
            # Clear stale options even when refreshing fails.
            st.session_state.pop("available_models", None)
            try:
                with st.spinner("Checking models listed by Google…"):
                    st.session_state["available_models"] = list_available_models(api_key)
                for name in ("chat_model_choice", "embedding_model_choice"):
                    st.session_state.pop(name, None)
            except Exception as exc:
                show_error(exc)
        available = st.session_state.get("available_models")
        if available:
            chat_model = st.selectbox("Answer model", available["chat"], key="chat_model_choice")
            embedding_model = st.selectbox("Embedding model", available["embedding"], key="embedding_model_choice")
            st.caption("These models are listed by Google for your key. Check free-tier quota in AI Studio; listing does not verify billing or quota.")
    can_ask = bool(api_key and chat_model and embedding_model)
    st.divider()
    st.markdown("**2 · Choose a document**")
    sample = st.toggle("Try the sample handbook")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"], disabled=sample,
                                help="Selectable text · Up to 10 MB and 100 pages")
    st.caption("Scanned PDFs need OCR first. Page references use PDF page order.")
    sample_path = Path(__file__).parent / "examples" / "sample_handbook.pdf"
    if sample_path.exists():
        st.download_button("Download sample PDF", sample_path.read_bytes(),
                           file_name="sample_handbook.pdf", mime="application/pdf")
    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state["messages"] = []
    st.caption("The index and conversation stay in this session's memory. Changing or removing the PDF clears them.")

st.caption("READ LESS. UNDERSTAND MORE.")
st.title("Your PDF, in conversation.")
st.write("Ask a question. Find an answer. See exactly which pages it came from.")

data, filename = None, None
if sample and sample_path.exists():
    data, filename = sample_path.read_bytes(), "sample_handbook.pdf"
elif not sample and uploaded is not None:
    data, filename = uploaded.getvalue(), uploaded.name

identity = (sha256(data).hexdigest(), filename, INDEX_VERSION, embedding_model) if data is not None else None
if st.session_state.get("selected_identity") != identity:
    reset_document()
    st.session_state["selected_identity"] = identity

if data is None:
    st.info("Upload a PDF in the sidebar, or switch on the sample handbook to get started.")
    left, middle, right = st.columns(3)
    with left:
        st.markdown("#### 01 · Add your PDF")
        st.write("Lecture notes, a handbook, or a research paper with selectable text.")
    with middle:
        st.markdown("#### 02 · Ask a question")
        st.write("Ask a specific question about information inside the document.")
    with right:
        st.markdown("#### 03 · Check the source")
        st.write("Open the quoted passages and page references to verify the answer.")
    st.stop()

st.markdown("**Selected document**")
st.text(filename)
ready = st.session_state.get("indexed_identity") == identity
if not ready:
    if not can_ask:
        st.info("Enter your Gemini API key, then click Load available models in the sidebar.")
    if st.button("Prepare PDF", type="primary", disabled=not can_ask):
        try:
            with st.status("Preparing your document…", expanded=True) as status:
                st.write("Extracting text and preserving page numbers…")
                parsed = read_pdf(data, filename)
                chunks = split_pages(parsed.pages)
                if using_demo:
                    check_demo_pdf(len(chunks))
                st.write(f"Creating a searchable index of {len(chunks)} passages…")
                progress = st.progress(0.0)
                note = st.empty()

                def show_progress(done, total):
                    note.empty()
                    progress.progress(done / total, text=f"Embedding passages {done}/{total}…")

                def show_wait(seconds):
                    note.info(f"Gemini's per-minute limit was reached. Waiting {seconds:.0f} seconds, "
                              "then continuing where it stopped…")

                embeddings = GeminiEmbeddings(api_key, model=embedding_model,
                                              on_progress=show_progress, on_wait=show_wait)
                try:
                    index = PDFIndex.build(chunks, embeddings)
                finally:
                    embeddings.close()
                st.session_state.update(index=index, parsed=parsed, messages=[], indexed_identity=identity)
                status.update(label="Ready to answer questions", state="complete", expanded=False)
            st.rerun()
        except Exception as exc:
            if not isinstance(exc, PDFError):
                st.error(f"PDF preparation failed using {embedding_model}.")
            show_error(exc)
    st.stop()

parsed = st.session_state["parsed"]
index = st.session_state["index"]
first, second, third = st.columns(3)
first.metric("PDF pages", parsed.total_pages)
second.metric("Searchable passages", len(index.chunks))
third.metric("Status", "Ready")
if parsed.skipped_pages:
    st.warning("Pages without selectable text were skipped: " + ", ".join(map(str, parsed.skipped_pages)))
st.caption("Ask each question in full; previous messages are displayed for reference but are not sent to the model.")
if sample:
    st.info('Try: “How many days of annual leave are available?” or “How do I report a lost laptop?”')

for message in st.session_state.get("messages", []):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_answer(message)
        else:
            st.text(message["text"])

question = st.chat_input("Ask something about this PDF…", disabled=not can_ask, max_chars=2000)
if not api_key:
    st.info("Add your API key again to ask questions.")
limit_message = None
if question and question.strip() and using_demo:
    limit_message = use_demo_quota("questions")
    if limit_message:
        st.warning(limit_message)
if question and question.strip() and not limit_message:
    with st.chat_message("user"):
        st.text(question)
    try:
        with st.chat_message("assistant"):
            with st.spinner("Finding the relevant passages…"):
                embeddings = GeminiEmbeddings(api_key, model=embedding_model)
                try:
                    model = make_model(api_key, model=chat_model)
                    answer = answer_question(question, index, embeddings, model)
                finally:
                    embeddings.close()
            message = {"role": "assistant", "text": answer.text, "sources": answer.sources}
            render_answer(message)
        st.session_state["messages"].extend([{"role": "user", "text": question}, message])
    except Exception as exc:
        st.error(f"Question failed using embeddings: {embedding_model}; answers: {chat_model}.")
        show_error(exc)

st.caption("Answers can be mistaken. Page references and source quotes help you check them.")
