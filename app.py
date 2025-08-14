import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def preprocess(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ''.join([char for char in text if not char.isdigit()])
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Streamlit UI
st.title("📩 SMS Spam Classifier")
input_sms = st.text_area("Enter the message:")

if st.button("Predict"):
    processed_sms = preprocess(input_sms)
    vector_input = vectorizer.transform([processed_sms])
    result = model.predict(vector_input)[0]

    if result == 1:
        st.error("🚨 This is a SPAM message!")
    else:
        st.success("✅ This is NOT spam.")
