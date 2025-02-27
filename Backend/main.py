import sys
sys.path.append(".")
from fastapi import FastAPI
from pydantic import BaseModel
# from Backend.chatbot2 import query_titanic
from Backend.chatbot3 import query_titanic
import uvicorn

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Titanic Chatbot API is running"}


class QuestionRequest(BaseModel):
    question: str

@app.post("/ask/")
def ask(question_request: QuestionRequest):
    # question = question_request.question
    # print(question)
    query = "How many passengers survived?"
    # query = "How many passengers embarked from each port?"  
    # query = "Show me a histogram of passenger ages"   
    # query = "is there any null value " 
    # query = "What was the average ticket fare?" 
    # query = "What is the average age of passengers?"
    # query = "How many rows are in the Titanic dataset?"
    response = query_titanic(query)
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
