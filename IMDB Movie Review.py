import pandas as pd
import string
import nltk
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.neural_network import MLPClassifier
import pickle

# nltk downloads (if needed)
# nltk.download('punkt')
# nltk.download('stopwords')

# Load dataset
df = pd.read_csv(r"C:\Users\ashis\OneDrive\Desktop\Python\Dataset\IMDB Dataset.csv", sep=',', on_bad_lines='skip')

# Remove HTML tags
def remove_html_tag(text):
    try:
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()
    except:
        return text  # Return text as is in case of error

df['review'] = df['review'].apply(remove_html_tag)

# Remove special characters and extra spaces
df['review'] = df['review'].str.replace(r"[^a-zA-Z]", ' ', regex=True)
df['review'] = df['review'].str.replace(r"\s+", ' ', regex=True).str.strip()

# Convert to lowercase
df["review"] = df["review"].str.lower()

# Tokenization
df['review'] = df['review'].apply(word_tokenize)

# Remove stopwords
stop_words = set(stopwords.words('english'))
df['review'] = df['review'].apply(lambda tokens: [word for word in tokens if word not in stop_words])

# Stemming
stemmer = PorterStemmer()
df["review"] = df["review"].apply(lambda tokens: [stemmer.stem(token) for token in tokens])

# Prepare data for TF-IDF
df["joined_review"] = df["review"].apply(" ".join)
text_data = df["joined_review"]
tfidf_vectorizer = TfidfVectorizer(max_features=2000)
x_tfidf = tfidf_vectorizer.fit_transform(text_data)

# Convert sentiment to numeric
df["sentiment_numeric"] = df["sentiment"].map({"positive": 1, "negative": 0})

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x_tfidf, df["sentiment_numeric"], test_size=0.3, random_state=1)

# Scale data
scaler = MaxAbsScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Train MLP model
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp_model.fit(x_train_scaled, y_train)

# Save the model, vectorizer, and scaler
with open("imdb_model.sav", "wb") as model_file:
    pickle.dump(mlp_model, model_file)

with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)

with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model, vectorizer, and scaler saved successfully!")
