import streamlit as st
import requests

st.title("Titanic Dataset Chatbot")

prompt = st.chat_input("Ask a question about the Titanic dataset (e.g., 'How many passengers survived?')")
if prompt:
    st.write(f"User: {prompt}")
    
    # API call to the backend
    try:
        response = requests.post(
            "http://127.0.0.1:8000/ask/",
            json={"prompt": prompt}
        )
        response.raise_for_status()  # Raise an error for bad status codes
        result = response.json()["response"]
        st.write(f"Chatbot: {result}")
    except requests.RequestException as e:
        st.error(f"Error connecting to the API: {e}")