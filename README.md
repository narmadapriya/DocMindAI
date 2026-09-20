# 🧠 DocMindAI — Agentic Multi-Document Multimodal RAG System

<p align="center">
  <strong>Local-first AI document intelligence platform for grounded chat, summarization, comparison, retrieval, verification, and citations.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Phase-15%20Complete-6D28D9?style=for-the-badge" alt="Phase 15 Complete" />
  <img src="https://img.shields.io/badge/Python-3.10.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10.11" />
  <img src="https://img.shields.io/badge/React-18.3.1-61DAFB?style=for-the-badge&logo=react&logoColor=111827" alt="React 18.3.1" />
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/LangGraph-Agentic%20RAG-7C3AED?style=for-the-badge" alt="LangGraph" />
  <img src="https://img.shields.io/badge/Ollama-Qwen2.5--VL%203B-111827?style=for-the-badge&logo=ollama&logoColor=white" alt="Ollama Qwen2.5-VL 3B" />
</p>

---

## 📌 Project Overview

**DocMindAI** is a full-stack **Agentic Multi-Document Multimodal Retrieval-Augmented Generation (RAG)** application designed for secure document intelligence.

The system allows users to upload supported documents, index their content into a local vector database, and interact with them through:

- 💬 **Chat & Ask**
- 📝 **File Summary**
- ⚡ **Executive Summary**
- 🔄 **Cross-Document Comparison**
- 🔎 **Evidence-grounded retrieval**
- ✅ **Evidence validation and answer verification**
- 🔗 **Source citations**
- 📊 **Usage and system analytics**

DocMindAI is built as a **local-first AI system**. Core RAG inference uses **Ollama + Qwen2.5-VL 3B**, while document embeddings use **nomic-embed-text**. No cloud LLM API key is required for the core RAG workflow.

---

## ✨ Key Features

### 📚 Multi-Document Knowledge Repository

- Upload **PDF, DOCX, TXT, CSV, and XLSX**
- Maximum **10 files per upload batch**
- Maximum **25 MB per file**
- Per-user storage quota support
- Persistent document metadata in PostgreSQL
- Vector indexing in ChromaDB
- Document ownership isolation

### 🧩 Multimodal Document Processing

DocMindAI processes multiple content types from supported files:

- 📝 Text
- 📊 Tables
- 🖼️ Images / visual document content
- 📑 Page and document metadata

The ingestion pipeline parses, chunks, enriches, embeds, and indexes document content before making it available to the RAG workflow.

### 💬 Chat & Ask

Ask questions against one or more selected RAG-ready documents.

The answer pipeline uses retrieved evidence rather than relying only on model knowledge.

### 📝 Summaries

Two summary modes are available:

- **File Summary**
- **Executive Summary**

The Summaries workflow accepts **exactly one document at a time**.

Selecting another document replaces the currently selected document.

### 🔄 Cross-Document Comparison

Compare information across **2–5 documents** using one or more requested metrics.

The comparison workflow retains independent multi-select behavior.

### 🔗 Citations

Generated answers can include source citations tied back to retrieved document evidence.

### 🔐 Authentication & User Isolation

- JWT-based authentication
- User registration and login
- Protected routes
- Password hashing
- Password recovery flow
- User-owned document access control

### 📈 Dashboard & Analytics

The dashboard provides visibility into:

- Document activity
- Storage usage
- RAG service health
- PostgreSQL status
- ChromaDB status
- Ollama/model status
- Embedding-model status
- LangGraph status

---

## 🏗️ Agentic RAG Architecture

```mermaid
flowchart TD
    A[👤 User] --> B[⚛️ React Frontend]
    B --> C[⚡ FastAPI API]
    C --> D[🔐 Authentication & Document Scope]

    D --> E[🧭 Router Agent]

    E --> F1[💬 Chat]
    E --> F2[📝 Summary]
    E --> F3[🔄 Comparison]

    F1 --> G[🔎 Retrieval Agent]
    F2 --> G
    F3 --> G

    G --> H[🧠 Embedding Model]
    H --> I[(🟣 ChromaDB)]

    I --> J[✅ Evidence Validation Agent]
    J --> K[🤖 Reasoning Agent]
    K --> L[🛡️ Verification Agent]
    L --> M[🔗 Citation Agent]
    M --> N[📨 Grounded Response]

    C --> P[(🐘 PostgreSQL)]
    C --> Q[🦙 Ollama]
    Q --> R[Qwen2.5-VL 3B]
```

---

## 🔁 RAG Workflow

```text
User Request
     │
     ▼
Router Agent
     │
     ├── CHAT
     ├── SUMMARY
     └── COMPARISON
     │
     ▼
Retrieval Agent
     │
     ▼
ChromaDB Vector Retrieval
     │
     ▼
Evidence Validation
     │
     ▼
Reasoning Agent
     │
     ▼
Verification Agent
     │
     ▼
Citation Agent
     │
     ▼
Grounded Final Response
```

---

## 🛠️ Technology Stack

### Languages

| Technology | Version / Standard | Purpose |
|---|---:|---|
| 🐍 Python | **3.10.11** | Backend, RAG, ingestion, AI services |
| 🟨 JavaScript / JSX | ES Modules | React frontend |
| 🎨 CSS | CSS3 + Tailwind CSS | Styling |
| 🗄️ SQL | PostgreSQL | Relational persistence |

### Backend & AI

| Technology | Version / Constraint | Purpose |
|---|---:|---|
| ⚡ FastAPI | Repository requirement, version not pinned | REST API |
| 🦄 Uvicorn | Repository requirement, version not pinned | ASGI server |
| 🧠 LangGraph | `>=0.2,<1.0` | Agentic workflow orchestration |
| 🟣 ChromaDB | `>=0.5,<2.0` | Vector database |
| 🦙 Ollama Python client | `>=0.4,<1.0` | Local model communication |
| 🤖 Qwen2.5-VL | **3B** (`qwen2.5vl:3b`) | Local reasoning / multimodal LLM |
| 🔢 nomic-embed-text | Ollama model | Local embeddings |
| 🐘 PostgreSQL | Version not pinned | Users, documents, chats, citations, analytics |
| 🧱 SQLAlchemy | Repository requirement, version not pinned | ORM |
| 🔄 Alembic | Repository requirement, version not pinned | Database migrations |
| 🔑 python-jose | Repository requirement, version not pinned | JWT handling |
| 🔒 Passlib + bcrypt | bcrypt `>=4,<5` | Password security |
| ✉️ email-validator | `>=2,<3` | Authentication/email validation |
| 📄 PyMuPDF | Repository requirement, version not pinned | PDF processing |
| 📄 pdfplumber | Repository requirement, version not pinned | PDF table/text extraction |
| 📝 python-docx | Repository requirement, version not pinned | DOCX parsing |
| 📊 openpyxl | Repository requirement, version not pinned | XLSX processing |
| 🖼️ Pillow | Repository requirement, version not pinned | Image processing |
| 🔤 pytesseract | Repository requirement, version not pinned | OCR integration |

### Frontend

| Technology | Version | Purpose |
|---|---:|---|
| ⚛️ React | **18.3.1** | UI framework |
| ⚛️ React DOM | **18.3.1** | React DOM rendering |
| ⚡ Vite | **6.4.3** | Development/build tooling |
| 🎨 Tailwind CSS | **3.4.19** | Styling |
| 🐻 Zustand | **5.0.15** | Client state management |
| 🌐 Axios | **1.19.0** | HTTP client |
| 🧭 React Router DOM | **6.30.6** | Routing |
| 📊 Recharts | **2.15.4** | Dashboard charts |
| 📄 React Markdown | **9.1.0** | Markdown answer rendering |
| 📝 remark-gfm | **4.0.1** | GitHub-Flavored Markdown support |
| 📤 React Dropzone | **14.4.1** | File upload UI |
| 📑 jsPDF | **3.0.2** | PDF exports |
| 🎯 Lucide React | **0.468.0** | UI icons |
| 🧩 clsx | **2.1.1** | Conditional classes |
| 🎨 tailwind-merge | **2.6.1** | Tailwind class merging |

> Backend base packages such as FastAPI, SQLAlchemy and Alembic are intentionally not pinned to exact versions in the current `requirements.txt`. Phase 15-specific packages use the constraints shown above.

---

## 📂 Supported File Types

| Type | Extension | Text | Tables | Visual Content |
|---|---|:---:|:---:|:---:|
| 📕 PDF | `.pdf` | ✅ | ✅ | ✅ |
| 📘 Word | `.docx` | ✅ | ✅ | ✅ |
| 📄 Text | `.txt` | ✅ | — | — |
| 📗 CSV | `.csv` | ✅ | ✅ | — |
| 📗 Excel | `.xlsx` | ✅ | ✅ | ✅ |

---

## 🗂️ Project Structure

```text
DocMindAI/
│
├── backend/
│   ├── alembic/
│   ├── app/
│   │   ├── api/
│   │   ├── auth/
│   │   ├── core/
│   │   ├── database/
│   │   ├── models/
│   │   ├── rag/
│   │   │   ├── agents/
│   │   │   ├── chunking/
│   │   │   ├── embeddings/
│   │   │   ├── llm/
│   │   │   ├── ocr/
│   │   │   ├── parser/
│   │   │   ├── pipelines/
│   │   │   ├── prompts/
│   │   │   ├── utils/
│   │   │   ├── vectordb/
│   │   │   └── vision/
│   │   ├── schemas/
│   │   ├── services/
│   │   └── utils/
│   │
│   ├── .env.example
│   ├── .password_reset.env.example
│   ├── alembic.ini
│   ├── main.py
│   ├── requirements.txt
│   └── requirements.phase15.txt
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   ├── store/
│   │   ├── styles/
│   │   ├── utils/
│   │   ├── App.jsx
│   │   └── main.jsx
│   │
│   ├── .env.example
│   ├── index.html
│   ├── package.json
│   ├── package-lock.json
│   ├── postcss.config.js
│   ├── tailwind.config.js
│   └── vite.config.js
│
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚙️ Prerequisites

Install the following before running DocMindAI:

- 🐍 **Python 3.10.11**
- 🟢 **Node.js 18+**
- 🐘 **PostgreSQL**
- 🦙 **Ollama**
- 🔧 **Git**
- 🔤 **Tesseract OCR** if OCR processing is enabled on your machine

---

## 🦙 Local AI Model Setup

Install Ollama and pull the models used by DocMindAI:

```powershell
ollama pull qwen2.5vl:3b
ollama pull nomic-embed-text
```

Verify:

```powershell
ollama list
```

Expected models include:

```text
qwen2.5vl:3b
nomic-embed-text
```

---

## 🐘 PostgreSQL Setup

Create a PostgreSQL database for DocMindAI.

Example database name:

```text
docmindai
```

Configure the connection in:

```text
backend/.env
```

Example:

```env
DATABASE_URL=postgresql+psycopg://docmindai:YOUR_PASSWORD@127.0.0.1:5432/docmindai
```

Never commit your real `.env` file.

---

## 🚀 Backend Setup

### 1. Open the backend directory

```powershell
cd backend
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

### 3. Activate it

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.phase15.txt
```

`requirements.phase15.txt` includes the base `requirements.txt`.

### 5. Create the environment file

```powershell
Copy-Item .env.example .env
```

Update the values in `.env`, especially:

```env
DATABASE_URL=postgresql+psycopg://docmindai:YOUR_PASSWORD@127.0.0.1:5432/docmindai
SECRET_KEY=YOUR_LONG_RANDOM_SECRET

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5vl:3b
EMBEDDING_MODEL=nomic-embed-text
```

### 6. Apply database migrations

```powershell
alembic upgrade head
```

### 7. Start FastAPI

```powershell
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Backend:

```text
http://127.0.0.1:8000
```

Swagger API documentation:

```text
http://127.0.0.1:8000/docs
```

---

## ⚛️ Frontend Setup

Open another terminal:

```powershell
cd frontend
```

Install dependencies:

```powershell
npm ci
```

Start the Vite development server:

```powershell
npm run dev
```

Frontend:

```text
http://127.0.0.1:5173
```

The Vite development server proxies `/api` requests to the FastAPI backend running on port `8000`.

---

## 🔐 Environment Configuration

Important backend configuration includes:

```env
PROJECT_NAME=DocMindAI
API_VERSION=v1

DATABASE_URL=postgresql+psycopg://...

SECRET_KEY=...
ALGORITHM=HS256

UPLOAD_DIR=uploads
CHROMA_DB_DIR=app/chroma_db

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5vl:3b
EMBEDDING_MODEL=nomic-embed-text
```

Password-recovery SMTP settings are also supported.

Use:

```text
backend/.env.example
```

as the public configuration template.

Do not commit:

```text
backend/.env
```

---

## 🧠 Core Agent Roles

| Agent | Responsibility |
|---|---|
| 🧭 Router Agent | Determines Chat, Summary, or Comparison intent |
| 🔎 Retrieval Agent | Retrieves relevant document evidence |
| ✅ Evidence Validation Agent | Validates retrieved evidence |
| 🤖 Reasoning Agent | Produces evidence-grounded reasoning |
| 🛡️ Verification Agent | Checks response consistency |
| 🔗 Citation Agent | Connects final answers to sources |
| 💬 Chat Agent | Handles document Q&A |
| 📝 Summary Agent | Handles summary requests |
| 🔄 Comparison Agent | Handles multi-document comparison |

---

## 🗃️ Data Storage

DocMindAI separates relational data from vector data:

### PostgreSQL

Stores application records such as:

- Users
- Documents
- Chunks
- Chats
- Messages
- Citations
- Settings
- Analytics

### ChromaDB

Stores embedded document chunks for semantic retrieval.

### Local Upload Storage

Uploaded source documents are stored locally under the configured upload directory.

---

## 🔒 Security

DocMindAI includes:

- JWT authentication
- Password hashing
- Protected API routes
- User/document ownership verification
- Secure document scoping during retrieval
- File validation
- Upload limits
- Storage quota handling
- Environment-based secret configuration

> Never commit `.env`, database credentials, JWT secrets, SMTP passwords, uploaded user documents, or ChromaDB runtime data.

---

## 📌 Application Workflows

### Chat & Ask

```text
Select document(s)
      ↓
Ask a question
      ↓
Retrieve evidence
      ↓
Validate evidence
      ↓
Reason + verify
      ↓
Generate citations
      ↓
Grounded answer
```

### Summary

```text
Select exactly 1 document
      ↓
Choose File Summary or Executive
      ↓
Retrieve document evidence
      ↓
Generate grounded summary
      ↓
Return citations
```

### Comparison

```text
Select 2–5 documents
      ↓
Enter comparison metric(s)
      ↓
Retrieve evidence from each document
      ↓
Cross-document reasoning
      ↓
Comparison result + citations
```

---

## 🖥️ Main Application Pages

- 🏠 Dashboard
- 💬 Chat & Ask
- 📚 Documents
- 📝 Summaries
- 🔄 Comparisons
- 🔗 Citations
- ⚙️ Settings
- 👤 Profile
- 🔐 Login / Registration / Password Recovery

---

## 🧪 Build Check

To verify the frontend production build:

```powershell
cd frontend
npm run build
```

For the backend, a basic startup check is:

```powershell
cd backend
uvicorn main:app --host 127.0.0.1 --port 8000
```

---

## 📊 Project Status

**DocMindAI Phase 15 — Completed**

The completed system includes:

- ✅ Full-stack React + FastAPI application
- ✅ PostgreSQL persistence
- ✅ JWT authentication
- ✅ Multi-format document ingestion
- ✅ Multimodal extraction
- ✅ ChromaDB vector storage
- ✅ Local Ollama inference
- ✅ Agentic LangGraph RAG
- ✅ Chat & Ask
- ✅ Single-document summaries
- ✅ 2–5 document comparisons
- ✅ Evidence validation
- ✅ Verification
- ✅ Citations
- ✅ Analytics dashboard
- ✅ Light and dark UI themes
- ✅ PDF export
- ✅ Password recovery workflow

---

## 🎯 Project Goal

DocMindAI demonstrates practical AI engineering skills across:

- Agentic AI
- Retrieval-Augmented Generation
- Multimodal document processing
- Local LLM deployment
- Vector databases
- Backend API design
- Relational databases
- Authentication and authorization
- Full-stack React development
- Evidence grounding and citations

It is designed as a portfolio project for **AI Engineer / AI-ML Engineer / Generative AI Engineer** roles.

---

## 📜 Version

```text
DocMindAI: Phase 15 Complete
Frontend package version: 2.0.0
```

---

<p align="center">
  <strong>🧠 DocMindAI — Ask. Retrieve. Verify. Cite.</strong>
</p>
