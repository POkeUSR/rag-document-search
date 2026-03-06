from fastapi import FastAPI
from pydantic import BaseModel
from rag import answer

app = FastAPI()

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask_question(q: Question):
    result = answer(q.question)
    return {
        "question": q.question,
        "answer": result
    }
