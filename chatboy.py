import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import streamlit as st

# Load the pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Set up the Streamlit app
st.title("Improved Dialogue GPT")

# Initialize the chat history
if 'chat_history_ids' not in st.session_state:
    st.session_state.chat_history_ids = None

# Function to generate a response
def generate_response(input_text):
    # Encode the new user input
    input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors="pt")
    
    # Append the new input to the chat history
    if st.session_state.chat_history_ids is not None:
        chat_history_ids = torch.cat([st.session_state.chat_history_ids, input_ids], dim=-1)
    else:
        chat_history_ids = input_ids
    
    # Generate the response
    chat_history_ids = model.generate(chat_history_ids,
                                     max_length=1000,
                                     do_sample=True,
                                     top_k=50,
                                     top_p=0.95,
                                     num_return_sequences=1)
    response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    # Update the chat history in session state
    st.session_state.chat_history_ids = chat_history_ids
    
    return response

# Main user interface
input_text = st.text_input("Enter your message")
if input_text:
    response = generate_response(input_text)
    st.text(response)
