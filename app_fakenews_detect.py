import nltk
nltk.download('stopwords')
nltk.download('punkt')


import streamlit as st
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure nltk resources are available
try:
    import nltk
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except Exception as e:
    st.error(f"Error downloading NLTK resources: {e}")

# Load the saved model and vectorizer using pickle
with open('news_classifier_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
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
