# spam_detection.py
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import joblib
import os
from sklearn.model_selection import train_test_split
import csv

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')  # Add this line

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
    return ' '.join(filtered_tokens)

def train_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, 'data', 'spam.csv')
    training_data = []
    with open(csv_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader, None)  # Skip the header row
        for row in csv_reader:
            if len(row) >= 2:
                label, message = row[0], row[1]
                training_data.append((message, label))
    X = [preprocess_text(text) for text, _ in training_data]
    y = [label for _, label in training_data]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train_tfidf, y_train)
    joblib.dump(model, 'models/spam_model.pkl')
    joblib.dump(vectorizer, 'models/vectorizer.pkl')

if not os.path.exists('models/spam_model.pkl'):
    train_model()
_model = None
_vectorizer = None

def get_model():
    global _model
    if _model is None:
        _model = joblib.load('models/spam_model.pkl')
    return _model

def get_vectorizer():
    global _vectorizer
    if _vectorizer is None:
        _vectorizer = joblib.load('models/vectorizer.pkl')
    return _vectorizer

def classify_message(message):
    model = get_model()
    vectorizer = get_vectorizer()
    preprocessed_message = preprocess_text(message)
    tfidf_feature_vector = vectorizer.transform([preprocessed_message])
    spam_probability = model.predict_proba(tfidf_feature_vector)[0][1]
    return spam_probability