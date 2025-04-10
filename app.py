import pandas as pd
import pickle as pk
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Clean the input like you did in training
def clean_review(review):
    return ' '.join(word for word in review.split() if word.lower() not in stop_words)

# Load the model and vectorizer
model = pk.load(open('model.pkl', 'rb'))
vectorizer = pk.load(open('vectorizer.pkl', 'rb'))

# Streamlit UI
st.title('Movie Review Sentiment Analysis')
review = st.text_input('Enter Movie Review')

if st.button('Predict'):
    cleaned = clean_review(review)
    review_vector = vectorizer.transform([cleaned]).toarray()
    result = model.predict(review_vector)
    if result[0] == 0:
        st.write('❌ Negative Review')
    else:
        st.write('✅ Positive Review')
