# Document Q&A

A Retrieval-Augmented Generation (RAG) web app that lets you upload documents and ask questions about them using natural language. Answers are grounded in the uploaded content, with source citations.

## Features

- Upload **PDF, DOCX, Markdown, or plain-text** files
- Automatic chunking, embedding, and FAISS + BM25 indexing
- **Three retrieval modes** — Semantic (FAISS), Keyword (BM25), Hybrid (RRF fusion)
- **Role-based answers** — General, Product Manager, or Sales framing
- **Document chunk preview** — see the exact text passage used for each answer
- **Query match score** per source — shows how closely each cited page matched your question
- Powered by **Groq** (`llama-3.3-70b-versatile`) for fast LLM inference
- **Streamlit** chat UI with expandable source citations
- Persistent chat history within a session
- Multi-document support with diversity-aware retrieval

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/ai-doc-assistant.git
cd ai-doc-assistant
```

### 2. Create a virtual environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env and add your Groq API key
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

### 5. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Usage

1. Set your Groq API key in `.env` (or it will be read from the environment).
2. Upload one or more documents using the file uploader.
3. Click **Analyze Documents** — the app will chunk and embed your documents.
4. Choose a **Retrieval Algorithm** in the sidebar: Semantic / Keyword (BM25) / Hybrid (RRF).
5. Choose an **Answer Style**: General / Product Manager / Sales.
6. Type a question in the chat box and press Enter.
7. Expand **Sources** under any answer to see cited pages, their query match score, and the matched text chunk.

## Project Structure

```
ai-doc-assistant/
├── app.py                  # Streamlit UI
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── .streamlit/
│   └── config.toml         # Hides Streamlit Deploy button (toolbarMode = "minimal")
└── src/
    ├── __init__.py
    ├── document_loader.py  # PDF / DOCX / MD / TXT parsing
    ├── chunker.py          # 1000-char sliding-window chunker
    ├── embeddings.py       # sentence-transformers (all-MiniLM-L6-v2)
    ├── vector_store.py     # FAISS IndexFlatL2 wrapper
    └── qa_chain.py         # RAG prompt + Groq API call
```

## Tech Stack

| Component | Library |
|-----------|---------|
| UI | Streamlit 1.35 |
| PDF parsing | PyMuPDF |
| DOCX parsing | python-docx |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector search | FAISS (faiss-cpu) |
| Keyword search | rank-bm25 (BM25Okapi) |
| LLM | Groq (`llama-3.3-70b-versatile`) |

## Architecture Overview

```
User uploads documents
        ↓
document_loader.py  — parse PDF / DOCX / MD / TXT into {text, source, page} dicts
        ↓
chunker.py          — sliding-window split (1000 chars, 200 overlap) into chunks
        ↓
embeddings.py       — encode chunks with sentence-transformers (all-MiniLM-L6-v2)
        ↓
vector_store.py     — store embeddings in FAISS IndexFlatL2 + BM25Okapi (both in-memory)
        ↓
User asks a question  +  selects retrieval mode  +  selects role
        ↓
vector_store.py     — retrieve top-5 chunks via selected mode:
                        • Semantic  → FAISS dense vector search
                        • Keyword   → BM25 term-frequency search
                        • Hybrid    → Reciprocal Rank Fusion of both
                      (multi-doc: search_diverse ensures per-source representation)
        ↓
qa_chain.py         — build prompt with retrieved context + role-specific system prompt,
                      call Groq LLM (llama-3.3-70b-versatile, temp=0.2, max 1024 tokens)
        ↓
app.py              — render answer + citations + query match score + chunk preview
```

## Known Limitations

- **In-memory index only** — the FAISS and BM25 indices are rebuilt on every upload and lost on page refresh or server restart. There is no persistence between sessions.
- **Keyword search limitations** — BM25 requires exact token matches; stemming and synonyms are not handled.
- **Single-page DOCX** — Word documents are treated as one page regardless of length, so page citations are always "page 1" for DOCX files.
- **Context window cap** — the prompt is truncated at 12 000 characters of context, so very long documents may have relevant sections silently excluded.
- **No conversation memory** — each question is answered independently; the LLM has no memory of previous turns in the chat.
- **API key required** — the app depends on the Groq cloud API; it cannot run fully offline.
- **Query match score is relative** — the displayed score shows how well a source matched *compared to other retrieved chunks*, not an absolute measure of answer correctness.

## Design Trade-offs

| Decision | Choice made | Alternative considered | Reason |
|---|---|---|---|
| **Vector store** | FAISS (in-memory) | Chroma / Pinecone (persistent) | Zero infrastructure, instant setup; persistence not required for this scope |
| **Embedding model** | `all-MiniLM-L6-v2` (local) | OpenAI `text-embedding-ada-002` (API) | Runs offline, no cost per query, fast enough for small doc sets |
| **LLM** | Groq `llama-3.3-70b-versatile` | OpenAI GPT-4o | Groq offers free tier with very low latency; GPT-4o adds cost and latency |
| **Chunking** | Character sliding-window (1000/200) | Sentence-aware / recursive splitter | Simple and predictable; sentence splitters add complexity for marginal gain at this scale |
| **Multi-doc retrieval** | `search_diverse` (top-k per source) | Plain top-k across all sources | Prevents one large document from dominating results and silencing smaller docs |
| **UI framework** | Streamlit | FastAPI + React | Streamlit ships a full chat UI in ~200 lines; React adds weeks of frontend work |
| **Hybrid retrieval** | RRF (Reciprocal Rank Fusion) | Score normalisation + weighted sum | RRF is rank-based so no score normalisation needed across different scales |
| **Confidence score** | Query match (retrieval similarity) | LLM self-rating (extra API call) | Avoids a second LLM call per query; score is honest about what it measures |

### Scaling to 100 000 users

The prototype is intentionally simple. The table below shows what each component would need to change to support production-scale traffic.

| Component | Prototype (now) | Production at 100k users |
|---|---|---|
| **Vector store** | FAISS in-memory, rebuilt per upload, lost on restart | Hosted vector DB (Pinecone / Weaviate / Qdrant) — persistent, multi-tenant, horizontally scalable |
| **Embedding** | `all-MiniLM-L6-v2` loaded in-process on app server | Dedicated embedding microservice (GPU-backed) or managed API (OpenAI / Cohere) with batching + caching |
| **LLM inference** | Single Groq API key, synchronous call | Load-balanced API key pool, async task queue (Celery + Redis), or self-hosted vLLM cluster for cost control |
| **Document indexing** | Blocking — indexing happens in the request thread, freezes the UI | Async pipeline: upload → S3 → Celery worker indexes in background → notifies UI via webhook/polling |
| **File storage** | Temp files on local disk, deleted after indexing | Object storage (S3 / GCS) for raw uploads; allows re-indexing, audit, multi-region access |
| **Session state** | Streamlit in-memory session (lost on server restart) | Redis-backed sessions — survives restarts, shared across multiple app replicas |
| **App server** | Single Streamlit process on one EC2 instance | Containerised (Docker) + orchestrated (ECS / Kubernetes) behind an ALB; auto-scales on CPU/request metrics |
| **Authentication** | None — anyone with the URL can use it | OAuth 2.0 / SSO (Google, GitHub) with per-user document namespacing in the vector DB |
| **Rate limiting** | None | API Gateway or Nginx rate limiting; per-user quotas to prevent abuse and control LLM costs |
| **Observability** | Print logs to stdout | Structured logging (CloudWatch / Datadog), request tracing, LLM cost dashboard, uptime alerting |

## Deployment (EC2)

> Tested on **Amazon Linux 2023**, Python 3.11, port 8501.

### Instance requirements

- **AMI**: Amazon Linux 2023
- **Instance type**: `t2.small` (2 GB RAM) — tested and working with 30 GB storage
- **Storage**: 30 GB (needed for `sentence-transformers` model ~500 MB + OS + packages)
- **Security Group inbound rules**:

  | Type | Port | Source |
  |------|------|--------|
  | SSH | 22 | Your IP |
  | Custom TCP | 8501 | 0.0.0.0/0 |

### Connect

```bash
chmod 400 your-key.pem
ssh -i your-key.pem ec2-user@<YOUR_EC2_IP>
```

### Full setup (copy-paste)

```bash
# System dependencies
sudo dnf update -y
sudo dnf install python3.11 python3.11-pip python3.11-devel git -y

# Virtual environment
python3.11 -m venv /home/ec2-user/venv
source /home/ec2-user/venv/bin/activate
pip install --upgrade pip

# Clone and install
cd /home/ec2-user
git clone <your-repo-url> ai-doc-assistant
cd ai-doc-assistant
pip install -r requirements.txt

# Set API key
echo "GROQ_API_KEY=<your_key_here>" > .env

# Start the app (background)
nohup streamlit run app.py --server.port 8501 --server.address 0.0.0.0 \
  &> /home/ec2-user/streamlit.log &

echo "Done. Open http://$(curl -s ifconfig.me):8501"
```

### Quick restart script

```bash
#!/bin/bash
# Save as ~/restart_app.sh then: chmod +x ~/restart_app.sh
source /home/ec2-user/venv/bin/activate
pkill -f streamlit
sleep 2
nohup streamlit run /home/ec2-user/ai-doc-assistant/app.py \
  --server.port 8501 --server.address 0.0.0.0 \
  &> /home/ec2-user/streamlit.log &
echo "Restarted. Open http://$(curl -s ifconfig.me):8501"
```

### Known deployment issues

**`TypeError: Client.__init__() got an unexpected keyword argument 'proxies'`**
`httpx >= 0.28.0` removed a parameter used internally by the Groq SDK. Fixed by pinning `httpx<0.28.0` — already included in `requirements.txt`.

**`ModuleNotFoundError: No module named 'dotenv'`**
Run `pip install python-dotenv` (not `dotenv`).

**App not accessible in browser**
Confirm port 8501 is open in your Security Group and use `http://` not `https://`.

**App crashes / out of memory**
`t2.micro` (1 GB RAM) is not enough. `t2.small` (2 GB RAM) with 30 GB storage works fine.

## License

MIT
