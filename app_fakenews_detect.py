import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Load the saved model and vectorizer
model = joblib.load('news_classifier_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Streamlit UI
st.title('Fake News Detection')

user_input = st.text_area("Enter news text:")

if st.button('Predict'):
    if user_input:
        cleaned_text = preprocess_text(user_input)
        transformed_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(transformed_text)
        result = "True News" if prediction[0] == 1 else "Fake News"
        st.write(f"Prediction: {result}")
    else:
        st.write("Please enter some text to predict.")