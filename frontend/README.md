# DocMindAI Phase 12 — React JavaScript Frontend

Frozen baseline: `frontend(4).zip` + the supplied Settings / Chat initial / Chat active screenshots.

## Stack
- Programming Language: JavaScript (JSX)
- Framework: React.js
- Styling: Tailwind CSS + custom `src/App.css`
- Build Tool: Vite

## Backend contract preserved
No backend files are included or changed. The frontend keeps the existing FastAPI contract, including OAuth2 email login, document APIs, and frozen RAG routes:
- POST `/api/auth/login`
- POST `/api/auth/register`
- GET `/api/users/me`
- POST `/api/upload` with compatibility fallback to `/api/documents/upload`
- document list/status/download/delete routes
- POST `/api/v1/chat`
- POST `/api/v1/summary`
- POST `/api/v1/compare`

## Requested UI changes implemented
1. Show/Hide password controls on Login, Register/Create Account, and Settings password inputs.
2. Document uploader supports PDF/DOCX/TXT/CSV/XLSX and enforces maximum 10 files per batch.
3. Settings defaults to Account & Password UI matching the supplied reference, with AI & RAG Parameters as the second tab.
4. Responsive desktop/tablet/mobile layout and mobile navigation drawer.
5. Smooth sidebar collapse/expand with icon-only collapsed mode.
6. Chat starts in source-selection state. Selecting a source shows RAG ACTIVE and enables the composer. Clear Chat clears messages and selected source.
7. Navigation/actions are wired to frontend state or existing backend API services.

## Local run 
Backend:
```powershell
cd C:\Users\Admin\Desktop\PROJECTS\DocMindAI\backend
uvicorn main:app --reload --host 127.0.0.1 --port 8000
http://127.0.0.1:8000
```

Frontend:
```powershell
cd C:\Users\Admin\Desktop\PROJECTS\DocMindAI\frontend
npm install
npm run test
npm run build
npm run dev
```

Vite proxies `/api` to `http://127.0.0.1:8000`.
