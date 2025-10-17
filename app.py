import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🕵️",
    layout="wide"
)

# --- Model and Tokenizer Loading ---
@st.cache_resource
def load_model():
    model_path = './fake_news_model'
    if not os.path.exists(model_path):
        return None, None
    try:
        model = BertForSequenceClassification.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)
        return model, tokenizer
    except Exception as e:
        return None, None

model, tokenizer = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if model:
    model.to(device)

# --- Web App Interface ---
st.title("Advanced Fake News Detector 🕵️")
st.markdown("This application uses a fine-tuned **BERT** model to classify news articles.")
st.markdown("---")

if not model or not tokenizer:
    st.error("Error: Model files not found. Please upload/unzip the 'fake_news_bert_model' folder.")
else:
    user_input = st.text_area("Enter article text:", height=250)
    if st.button("Analyze News", type="primary"):
        if user_input:
            with st.spinner('Analyzing...'):
                inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
                inputs = {key: val.to(device) for key, val in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs)
                    prediction = torch.argmax(outputs.logits, dim=-1).item()

                st.markdown("---")
                st.subheader("Analysis Result")
                if prediction == 1:
                    st.success("✅ Real News")
                else:
                    st.error("❌ Fake News")
        else:
            st.warning("Please enter text to analyze.")