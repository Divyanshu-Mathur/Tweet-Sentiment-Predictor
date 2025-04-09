import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pickle

model = load_model('model.h5',compile=True)
model.summary()

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("encoder.pkl", "rb") as f:
    label_enc = pickle.load(f)

MAX_LEN = 100

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|@\w+|#\w+|[^a-zA-Z\s]", "", text)
    return text.strip()

# Streamlit UI
st.title("Tweet Sentiment Predictor ")
user_input = st.text_area("Enter a tweet to analyze its sentiment:")
print("User:",user_input)

if st.button("Predict"):
    cleaned = clean_text(user_input)
    print("Cleaned:",clean_text)
    seq = tokenizer.texts_to_sequences([cleaned])
    print("Seq: ",seq)
    if not seq or not seq[0]:
        st.error("The input doesn't contain any recognizable words. Please try a valid sentence.")
    else:
        padded = pad_sequences(seq, maxlen=MAX_LEN)
        print("Padded:",padded)

        probs = model.predict(padded)
        pred_idx = np.argmax(probs)
        pred_label = label_enc.inverse_transform([pred_idx])[0]

        st.success(f"Predicted Sentiment: **{pred_label}**")
