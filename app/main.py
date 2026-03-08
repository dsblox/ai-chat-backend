from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

# Load .env from backend project root (parent of app/), regardless of cwd
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
from fastapi.middleware.cors import CORSMiddleware

from app.chat_service import ChatService
from app.llm_factory import make_llm
from app.schemas import ChatRequest, ChatResponse

app = FastAPI()
chat_service = ChatService(llm=make_llm())

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
    return chat_service.chat(request)
