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


if "history" not in st.session_state:
    st.session_state["history"] = []

user_input = st.text_input("You: ","")

if user_input:
    inputs = tokenizer.encode(user_input, return_tensors="pt")
    outputs = model.generate(inputs, max_length = 50, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens = True)

st.session_state["history"].append(f"You: {user_input}")
st.session_state["history"].append(f"Bot: {response}")

st.text_area("Conversation: ", "\n".join(st.session_state["history"]), height = 400)
