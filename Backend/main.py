import sys
sys.path.append(".")
from fastapi import FastAPI
from pydantic import BaseModel
# from Backend.chatbot2 import query_titanic
from Backend.chatbot import query_titanic
import uvicorn

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Titanic Chatbot API is running"}


class QuestionRequest(BaseModel):
    question: str

@app.post("/ask/")
def ask(question_request: QuestionRequest):
    question = question_request.question
    print(question)
    response = query_titanic(question)
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
