import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

@st.cache_resource

def loadmodel():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return model, tokenizer

model, tokenizer = loadmodel()

st.title("GPT-2 ChatBot")
st.write("How can I assist you ?")
