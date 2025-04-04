import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    return tokenizer, model

tokenizer, model = load_model()

# Streamlit UI
st.title("🤖 GPT-2 Text Generator")
st.markdown("Enter a prompt below and generate text using GPT-2")

prompt = st.text_area("Enter your prompt:", value="My name is Nawneet Raj", height=100)
max_len = st.slider("Max length of generated text:", min_value=20, max_value=200, value=50)
top_k = st.slider("Top-K Sampling:", min_value=10, max_value=100, value=50)
top_p = st.slider("Top-P (nucleus) Sampling:", min_value=0.1, max_value=1.0, value=0.95)

if st.button("Generate"):
    with st.spinner("Generating text..."):
        inputs = tokenizer(prompt, return_tensors="pt")
        output = model.generate(
            inputs["input_ids"],
            max_length=max_len,
            num_return_sequences=1,
            do_sample=True,
            top_k=top_k,
            top_p=top_p
        )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        st.success("Generated Text:")
        st.write(generated_text)
