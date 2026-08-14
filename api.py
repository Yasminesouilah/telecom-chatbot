from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from rag.pipeline import chat, setup, stream_chat
from rag.classifier import _get_classifier
from rag.embedding import get_embedding_fn


class ChatRequest(BaseModel):
    message: str
    # optional conversation history: list of message objects {from: 'user'|'bot', text: str}
    history: Optional[List[Dict[str, Any]]] = None


app = FastAPI(title="Telecom Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    # Ensure the knowledge base is ready. This is idempotent and fast
    # after the first run because the collection is persisted.
    try:
        setup(rebuild=False)
        # Warm up expensive components so the first real request is fast:
        try:
            _get_classifier()
        except Exception:
            # non-fatal: classifier may be missing in lightweight setups
            pass
        try:
            get_embedding_fn()
        except Exception:
            pass
    except Exception as e:
        # If setup fails on startup, we still allow the app to run and
        # surface errors on requests.
        print("Warning: setup failed on startup:", e)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="message is required")
    try:
        # pass conversation history through to the RAG pipeline so the LLM
        # can use conversational context (e.g. pronoun references, follow-ups)
        answer = chat(req.message, history=req.history)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="message is required")

    def event_generator():
        import json

        try:
            # Use the new stream_chat which yields tokens. Wrap each token
            # as a JSON-framed SSE `data:` event so clients (and proxies)
            # can reliably parse incremental updates.
            for token in stream_chat(req.message, history=req.history):
                payload = json.dumps({"type": "token", "text": token})
                yield (f"data: {payload}\n\n").encode("utf-8")

            # Signal completion
            done_payload = json.dumps({"type": "done"})
            yield (f"data: {done_payload}\n\n").encode("utf-8")
        except Exception as e:
            err_payload = json.dumps({"type": "error", "error": str(e)})
            yield (f"data: {err_payload}\n\n").encode("utf-8")

    # SSE-compatible MIME type
    return StreamingResponse(event_generator(), media_type="text/event-stream; charset=utf-8")
