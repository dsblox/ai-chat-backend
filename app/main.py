from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.chat_service import ChatService
from app.llm import StubLLMClient
from app.llm_factory import make_llm
from app.schemas import ChatRequest, ChatResponse

# Load .env from backend project root (parent of app/), regardless of cwd
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    # Per-request LLM selection: default to stub, even if OpenAI is configured.
    llm_name = (request.metadata or {}).get("llm")
    if llm_name == "openai":
      llm = make_llm()
    else:
      llm = StubLLMClient()

    service = ChatService(llm=llm)
    return service.chat(request)
