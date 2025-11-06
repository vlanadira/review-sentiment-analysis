import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load model & tokenizer (pake cache biar gak reload terus)
@st.cache_resource
def load_model_and_tokenizer():
    model = load_model("gru_sentiment_model.h5")
    with open("tokenizer-3.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# Page Configuration
st.set_page_config(
    page_title="Sentiment Analysis on Game Review",
    page_icon="🤖",
    layout="centered"
)

# Title
st.title("🤖 Prototype of a Game Review Sentiment Analysis Tool")

# Media
st.image("Screenshot 2025-11-06 at 15.08.15.png", 
         caption="The focus of this tool is to analyze the review texts, not the games themselves.")

# Text Context
with st.container(border=True):
    st.header("Welcome, Gamers!")
    st.write("This is **not** a *game*. It is a tool that I have trained to read game review text (like on Steam) and it'll automatically predict whether the review is **Positive** or **Negative**.")
    
    st.subheader("Your Testing Scenario:")
    st.markdown("""
    Imagine you are a *gamer* or a *reviewer* who wants to test how accurate this tool is.

    **Your Task:**
    1. Try entering a game review text in the box below.
    2. You can write your own, or *copy-paste* a review directly from Steam.
    3. Click the "Analyze Sentiment" button and see the result.
    """)

# Helpful Examples
with st.expander("🤔 Not sure what to write? Check these examples!"):
    col1, col2 = st.columns(2)
    with col1:
        st.success("**Loved the game? Say something like:**")
        st.write("'This game is absolutely amazing! The graphics are stunning and the storyline kept me hooked for hours!'")
    with col2:
        st.error("**Didn't enjoy it? Try something like:**")
        st.write("'Huge disappointment. The game is full of bugs and the controls are terrible.'")


st.divider()

user_input = st.text_area("Enter your text here:", "", height=150)

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
        sentiment = "Positif" if prediction[0] > 0.5 else "Negatif"

        # Color style
        if sentiment == "Positif":
            sentiment_color = "color:green; font-size:28px; font-weight:bold;"
            st.markdown(
                f"<p style='{sentiment_color}'>🟢 Predicted Sentiment: Positive </p>",
                unsafe_allow_html=True
            )
        else:
            sentiment_color = "color:red; font-size:28px; font-weight:bold;"
            st.markdown(
                f"<p style='{sentiment_color}'>🔴 Predicted Sentiment: Negative </p>",
                unsafe_allow_html=True
            )
    else:
        st.error("Please enter some text.")
