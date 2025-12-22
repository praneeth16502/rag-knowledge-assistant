from fastapi import FastAPI
from pydantic import BaseModel
from app.ingest import load_and_split_pdfs
from app.rag import build_vector_store, answer_question

app = FastAPI(title="LLM-Powered Knowledge Assistant (RAG)")

vectorstore = None


class QuestionRequest(BaseModel):
    query: str


@app.on_event("startup")
def startup():
    global vectorstore
    documents = load_and_split_pdfs("data/docs")
    vectorstore = build_vector_store(documents)


@app.post("/ask")
def ask(request: QuestionRequest):
    return answer_question(vectorstore, request.query)
