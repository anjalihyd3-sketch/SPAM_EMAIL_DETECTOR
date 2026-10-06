
import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

st.title("📧 Spam Email Detector")
st.write("Check whether an email is spam or normal.")

@st.cache_resource
def train_model():
    data = pd.read_csv("spam.csv")

    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", MultinomialNB())
    ])

    model.fit(data["message"], data["label"])
    return model

model = train_model()

email = st.text_area("Enter your email message")

if st.button("Check Email"):
    if email.strip():
        result = model.predict([email])[0]

        if result == "spam":
            st.error("🚨 This message is SPAM!")
        else:
            st.success("✅ This message is NOT SPAM!")
    else:
        st.warning("Please enter an email message.")