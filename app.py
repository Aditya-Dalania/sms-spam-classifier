import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already downloaded
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# Load stopwords once and store in a set (for faster lookup)
stop_words = set(stopwords.words("english"))

# Function for text cleaning
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\W", " ", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords efficiently
    return " ".join(words)

# Load the vectorizer and model
try:
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    model = pickle.load(open("model.pkl", "rb"))
except FileNotFoundError:
    st.error("Error: Model files not found! Ensure 'vectorizer.pkl' and 'model.pkl' are in the correct directory.")
    st.stop()

# Streamlit UI
st.title("📧 Email/SMS Spam Detector")
st.write("Enter a message below to check if it's spam or not.")

# Input text box
user_input = st.text_area("Type your email content here...", "")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Clean and transform the input
        cleaned_input = clean_text(user_input)
        transformed_text = vectorizer.transform([cleaned_input])
        prediction = model.predict(transformed_text)[0]

        # Display result
        if prediction == 1:
            st.error("🚨 This email is SPAM!")
        else:
            st.success("✅ This email is NOT spam.")
