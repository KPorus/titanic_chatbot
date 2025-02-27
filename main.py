import sys
sys.path.append(".")
from fastapi import FastAPI
from threading import Thread
from pydantic import BaseModel
# from Backend.chatbot2 import query_titanic
from Backend.chatbot3 import query_titanic
import uvicorn
import streamlit as st
import requests


app = FastAPI()

@app.get("/")
def home():
    return {"message": "Titanic Chatbot API is running"}


class QuestionRequest(BaseModel):
    prompt: str

@app.post("/ask/")
def ask(question_request: QuestionRequest):
    # question = question_request.question
    # print(question)
    # query = "How many passengers survived?"
    # query = "How many passengers embarked from each port?"  
    # query = "Show me a histogram of passenger ages"   
    # query = "is there any null value " 
    # query = "What was the average ticket fare?" 
    # query = "What is the average age of passengers?"
    # query = "How many rows are in the Titanic dataset?"
    response = query_titanic(question_request.prompt)
    return {"response": response}

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)


# Start FastAPI server in a separate thread
def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if "fastapi_thread" not in st.session_state:
    thread = Thread(target=run_fastapi, daemon=True)
    thread.start()
    st.session_state["fastapi_thread"] = True

# Frontend (Streamlit)
st.title("Titanic Dataset Chatbot")
prompt = st.chat_input("Ask a question about the Titanic dataset (e.g., 'How many passengers survived?')")
if prompt:
    with st.spinner("Processing..."):
        try:
            response = requests.post(
            "http://localhost:8000/ask/",
            json={"prompt": prompt}
            ).json()["response"]
            st.write(f"User: {prompt}")
            st.write(f"Chatbot: {response}")
        except requests.RequestException as e:
            st.error(f"Error connecting to the API: {e}")