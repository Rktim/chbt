from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import streamlit as st


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

st.title("Dialogue GPT")


if 'chat_history_ids' not in st.session_state:
    st.session_state.chat_history_ids = None

# Encode user input and generate a response
input_text = st.text_input("Enter your message")

if input_text:
    # Encode the new user input
    input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors="pt")

    # Append the new input to the chat history
    if st.session_state.chat_history_ids is not None:
        chat_history_ids = torch.cat([st.session_state.chat_history_ids, input_ids], dim=-1)
    else:
        chat_history_ids = input_ids

    # Generate response
    chat_history_ids = model.generate(chat_history_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Update the chat history in session state
    st.session_state.chat_history_ids = chat_history_ids

    # Display the response
    st.text(response)