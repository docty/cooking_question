# Load model directly
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("Docty/question_and_answer")

model = AutoModelForQuestionAnswering.from_pretrained("Docty/question_and_answer", from_tf=True) 


def get_answer(question, context):
    # Tokenize the question and context
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")

    # Get the model's outputs
    outputs = model(**inputs)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    # Find the start and end indices of the answer
    start_index = torch.argmax(start_logits)
    end_index = torch.argmax(end_logits)

    # Convert token indices to answer text
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index:end_index + 1]))

    return answer

def main():
    st.title("Question Answering with Hugging Face Transformers")

    # Input box for the context
    context = st.text_area("Enter the context:")

    # Input box for the question
    question = st.text_input("Ask a question:")

    if st.button("Get Answer"):
        answer = get_answer(question, context)
        st.write(f"Question: {question}")
        st.write(f"Answer: {answer}")

if __name__ == "__main__":
    main()
