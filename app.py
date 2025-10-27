import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load model & tokenizer (gunakan cache biar gak reload terus)
@st.cache_resource
def load_model_and_tokenizer():
    model = load_model("gru_sentiment_model.h5")
    with open("tokenizer-2.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

st.title("🎮 Hey Gamer!")
st.markdown("**How are you feeling about this game?**")
st.markdown("Tell me your honest thoughts, and I'll analyze whether you loved it or not! 😊")
st.markdown("---")

# Helpful examples
with st.expander("🤔 Not sure what to write? Check these examples!"):
    col1, col2 = st.columns(2)
    with col1:
        st.success("**Loved it? Say something like:**")
        st.write("*'This game is absolutely amazing! The graphics are stunning and the storyline kept me hooked for hours!'*")
    with col2:
        st.error("**Didn't enjoy it? Try something like:**")
        st.write("*'Huge disappointment. The game is full of bugs and the controls are terrible.'*")

# User Input
user_input = st.text_area("Enter your text:", "")

# Process input & make prediction if submitted
if st.button("Analyze Sentiment"):
    if user_input.strip():
        # Convert text input to tokens using the tokenizer
        tokenized_input = tokenizer.texts_to_sequences([user_input])

        # Padding input to match the expected input length for the model
        max_len = 150
        padded_input = pad_sequences(
            tokenized_input, maxlen=max_len, padding='post', truncating='post'
        )

        # Predict using the model
        prediction = model.predict(padded_input)

        # Determine sentiment
        sentiment = "Positive" if prediction[0] > 0.5 else "Negative"

        # Color style
        if sentiment == "Positive":
            sentiment_color = "color:green; font-size:24px;"
        else:
            sentiment_color = "color:red; font-size:24px;"

        # Display sentiment
        st.markdown(
            f"<p style='{sentiment_color}'>Predicted Sentiment: {sentiment}</p>",
            unsafe_allow_html=True
        )
    else:
        st.error("Please enter some text.")