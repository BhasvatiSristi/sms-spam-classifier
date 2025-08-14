import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure all required NLTK data is available
nltk.download('punkt')
nltk.download('punkt_tab')  # NEW: required for NLTK 3.9+
nltk.download('stopwords')

ps = PorterStemmer()

def transforming(message):
    # Lowercase
    message = message.lower()
    # Tokenize
    message = nltk.word_tokenize(message)

    y = []
    for i in message:
        if i.isalnum():  # Keep only alphanumeric
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # Stemming
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit UI
st.title("SMS Spam Classifier 📩")

mesg = st.text_area("Enter the message:")

if st.button('Predict'):
    if mesg.strip():
        transformed_mesg = transforming(mesg)
        vector_mesg = tfidf.transform([transformed_mesg])
        pred = model.predict(vector_mesg)[0]

        if pred == 1:
            st.error("🚨 Spam Message Detected!")
        else:
            st.success("✅ This is a Not Spam (Ham) Message.")
    else:
        st.warning("Please enter a message before predicting.")

# Footer
st.markdown("""
    <style>
        .footer {
            position: fixed;
            bottom: 10px;
            right: 10px;
            color: #aaa;
            font-size: 13px;
        }
    </style>
    <div class="footer">
        Made by <b>Bhasvati Sristi</b> @ IIITDM Kancheepuram
    </div>
""", unsafe_allow_html=True)
