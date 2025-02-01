# Load model directly
import torch
import tensorflow as tf
import streamlit as st
from transformers import   pipeline

 
pipe = pipeline("question-answering", model="Docty/question_and_answer") 


 

def main():
    st.title("Question Answering with Hugging Face Transformers")

    # Input box for the context
    context = st.text_area("Enter the context:")

    # Input box for the question
    question = st.text_input("Ask a question:")

    if st.button("Get Answer"):
        answer = pipe(question, context)   
        st.write(f"Question: {question}")
        st.write(f"Answer: {answer['answer']}")

if __name__ == "__main__":
    main()
