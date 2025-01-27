from faastapi import FaastAPI
import streamlit as st
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


app = FaastAPI()

# Load the model, vectorizer, and scaler
model = pickle.load(open('imdb_model.sav', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))


# Preprocessing function
def preprocess_review(review: str):
    # Remove special characters and extra spaces
    review = ''.join([char if char.isalpha() else ' ' for char in review])  # Keeps only alphabetic characters
    review = ' '.join(review.split())  # Remove extra spaces
    review = review.lower()  # Convert to lowercase

    # Tokenization
    tokens = word_tokenize(review)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    # Return preprocessed review
    return " ".join(tokens)


# Streamlit UI
st.title("🎥 IMDB Sentiment Analysis")
st.write("Predict if a movie review is **Positive** or **Negative**.")

# Input text box for the user
user_review = st.text_area("Enter your movie review:", placeholder="Type your review here...")

if st.button("Predict Sentiment"):
    if user_review.strip() == "":
        st.error("Please enter a valid review!")
    else:
        try:
            # Preprocess the input
            preprocessed_review = preprocess_review(user_review)

            # Vectorize and scale the input
            review_tfidf = vectorizer.transform([preprocessed_review])
            review_scaled = scaler.transform(review_tfidf)

            # Predict sentiment
            prediction = model.predict(review_scaled)[0]
            sentiment = "Positive" if prediction == 1 else "Negative"

            # Display result
            st.success(f"The sentiment of this review is **{sentiment}**!")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
