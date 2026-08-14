# Mobilis Telecom Chatbot — Streamlit UI

Run the Streamlit interface locally:

```bash
pip install -r requirements.txt   # ensure dependencies including streamlit are installed
streamlit run streamlit_app.py
```

The sidebar has buttons to build or ensure the knowledge base is loaded. Use the text box to enter customer messages and click "Send" to get answers from the RAG pipeline.

## Developer: React frontend + FastAPI backend

Quick instructions to run the new React frontend and the FastAPI wrapper:

1. Install Python dependencies (adds `fastapi` + `uvicorn`):

```bash
pip install -r requirements.txt
```

2. Run the backend API (from repo root):

```bash
uvicorn api:app --reload --port 8000
```

3. Start the frontend (from `frontend/`):

```bash
cd frontend
npm install
npm run dev
```

Open the Vite dev URL (usually http://localhost:5173) to use the React interface. The frontend talks to the backend at `http://localhost:8000/chat`.

