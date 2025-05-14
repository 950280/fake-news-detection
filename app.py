import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords (only first time)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Clean the text like before
def clean_text(text):
    text = re.sub(r'\W', ' ', text)              # Remove special characters
    text = text.lower()                          # Convert to lowercase
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Setup the web page
st.set_page_config(page_title=" Fake News Detector", layout="centered")
st.title("📰 Fake News Detection App")
st.subheader("Enter a news headline to find out if it's Real or Fake!")

# One-time note
st.info("ℹ️ Note: This model is trained on specific datasets. It may not be accurate for unknown or unrelated topics.")

# User input field
user_input = st.text_input("Type your news headline here:")

# When user clicks the button
if st.button("Check News"):
    if user_input:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0].strip().lower()  #clean up the output

        #print for debugging 
        print("Prediction:", prediction)
        
        if prediction == "fake":
            st.error("This news is **Fake**.")
        elif prediction == "real":
            st.success("This news is **Real**.")
        else:
            st.warning(f"Unable to determine. Model returned:{prediction}")    
    
