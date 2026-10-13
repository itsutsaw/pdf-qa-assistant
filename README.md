# Pagewise — PDF Question-Answering Assistant

A small resume project: upload a PDF, ask a question, and read an answer with verified source quotations and page references.

**Stack:** Python, Streamlit, LangChain, FAISS, pypdf, and the Gemini Developer API.

## Run locally

Use Python 3.11 or later (this project was tested with Python 3.14).

```bash
cd /home/utsaw/Desktop/Machine_Learning/ML_Challenge/pdf_qa_assistant
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 127.0.0.1
```

Open http://localhost:8501. Enter a Gemini API key in the password field in the sidebar, click **Load available models**, choose the listed answer and embedding models, then choose a PDF or the sample handbook and click **Prepare PDF**. Then ask a complete question. You can instead provide your key through the `GEMINI_API_KEY` environment variable. Do not put a real key in source files or commit one to Git.

The local virtual environment may already be installed if you are using the workspace where this project was built. In that case, activate it and run Streamlit directly.

## Get your Gemini key and stay on the free tier

1. Open [Google AI Studio API keys](https://aistudio.google.com/apikey) and sign in with your Google account.
2. Complete the initial setup, then create an API key in a project. If prompted, create a new project.
3. Confirm that project's plan is **Free tier**. Leave paid billing disabled if you want no-charge usage.
4. Copy the key into the app's **Gemini API key** password field. Do not send it in chat or commit it to Git.
5. Click **Load available models**. The dropdowns contain compatible models returned by Google for your key. Prefer a Flash-Lite answer model and `gemini-embedding-001` when listed; confirm their free-tier quota in AI Studio.
6. Switch on **Try the sample handbook**, click **Prepare PDF**, and ask “How many days of annual leave are available?”

The app now discovers models through Google's model-list endpoint instead of relying on fixed model names. It lists compatible Flash answer models and supported Gemini embedding models. Model listing does not guarantee inference access, free-tier pricing, or remaining quota. If a model still returns 404, reload the list and select another model. The error includes whether PDF preparation or answering failed and the selected model names. Changing the embedding model or API key clears the index so vectors from different models cannot be mixed.

The app cannot determine your billing tier or prevent Google from charging a billing-enabled project. There is no automatic paid-model or OpenAI fallback.

Free-tier usage is limited, not unlimited. A 429/quota message means you should check your limits in AI Studio and wait for them to reset; a shorter PDF uses fewer embedding requests. The app makes one embedding request per text passage, then one query-embedding request and one answer request per question. For a first test, use the three-page sample. Failed preparation can use quota for already-completed passages; do not repeatedly retry a large PDF.

This is not an offline app. Preparing a PDF sends its extracted text to Google. Questions and retrieved passages are also sent to Google. Google's free-tier content may be used to improve its products, so use public or sample PDFs. No API calls are made just by opening the app or selecting a PDF. **Load available models** sends only a model-list request, not PDF content.

## Try it

Switch on **Try the sample handbook**, prepare it, and ask:

- How many days of paid annual leave do employees receive? (Page 1: 20 days.)
- How do I report a lost laptop? (Page 2: contact the IT help desk immediately.)
- What is the meal reimbursement limit? (Page 3: INR 500 per day during approved business travel.)
- Who won the football World Cup? (Not found in the document.)

The sample is fictional and contains no personal/company data. `examples/evaluation_questions.json` contains a small manual evaluation checklist, not measured benchmark results.

## How it works

1. **Extraction:** pypdf reads selectable text, recording each physical PDF page number.
2. **Chunking:** LangChain splits each page into 1,000-character passages with 150-character overlap. Chunks never cross a page boundary.
3. **Embedding:** The selected embedding model converts each passage into its own 768-dimensional vector. Embedding 001 uses retrieval task types; Embedding 2 uses document/query prefixes. Inputs are sent separately to avoid aggregation.
4. **Retrieval:** an in-memory FAISS index searches normalized vectors using cosine similarity and selects up to four passages.
5. **Generation:** The selected Gemini Flash model returns a structured answer and supporting quotations through LangChain.
6. **Validation:** the app checks that source IDs exist and quotes occur in the retrieved text, then maps them to actual page numbers. Unsupported answers or invalid citations produce “Not found in the document.”

The question and document are treated as data in the model prompt. Previous chat messages are displayed but are not sent to the model; ask each question in full. No agents, tools, accounts, or database are needed.

## Project files

```text
app.py                         Streamlit interface
providers.py                   Gemini embeddings, chat client, and API error handling
rag.py                         PDF parsing, chunking, FAISS search, answer validation
examples/sample_handbook.pdf    Three-page fictional demonstration document
examples/evaluation_questions.json
tests/                         Offline pipeline and UI tests
requirements.txt               Runtime dependencies
requirements-dev.txt           Test dependencies
.streamlit/config.toml          Theme and upload limits
```

## Test

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests use real PDF parsing and FAISS search, plus deterministic embedding/model test doubles and real Gemini adapters with mocked SDK responses. They do **not** establish live model accuracy, latency, or cost. They cover invalid/encrypted/blank PDFs, page preservation, retrieval, invalid citations, unsupported answers, and the Streamlit question/clear/document-removal flow. No API key or paid calls are needed for tests.

For manual evaluation, run the included questions with a real API key, record whether the answer and source page are correct, and add questions from another PDF. Only report measured results on your resume.

## Scope and limitations

- One PDF per session, maximum 10 MB, 100 pages, 150,000 extracted characters, and 220 chunks.
- Scanned/image-only pages require OCR elsewhere. Empty pages are skipped with a notice. PDF page order can differ from printed page labels.
- Tables, formulas, and multi-column layouts may extract poorly.
- Top-four retrieval can miss relevant context; “Not found” can mean retrieval or citation validation failed, not that the fact is definitely absent from the entire PDF.
- Quote validation checks provenance; it cannot prove that every generated claim logically follows from its quote. Inspect the displayed sources.
- Indexes, chat history, and entered keys remain in session memory; the app does not write uploaded PDFs or indexes to disk or share them through a global cache. Switching/removing the document clears the old index and chat.
- This is a local portfolio app, not a production service with authentication, persistent storage, or a hardened untrusted-file processing sandbox.

## Resume description

After running and reviewing the app yourself:

**PDF Question-Answering Assistant | Python, Streamlit, LangChain, FAISS**

- Built a RAG application that answers questions from uploaded PDFs using semantic retrieval and an LLM.
- Implemented page-preserving text chunking, source quotations, page references, and fallback handling for unsupported answers.

## References

- [Streamlit chat elements](https://docs.streamlit.io/develop/api-reference/chat)
- [Gemini API keys](https://ai.google.dev/gemini-api/docs/api-key)
- [Gemini pricing and free tier](https://ai.google.dev/gemini-api/docs/pricing)
- [Gemini embeddings](https://ai.google.dev/gemini-api/docs/embeddings)
- [LangChain documentation](https://docs.langchain.com/oss/python/langchain/overview)
