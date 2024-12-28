import requests
import json
from flask import Flask, request
from collections import deque, defaultdict
from time import time
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
import os
from scipy.sparse import issparse
import threading
import joblib
from concurrent.futures import ThreadPoolExecutor

model_lock = threading.Lock()
current_model = None
current_vectorizer = None

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

BOT_ID = "7ac505c1ac85af8a1dedb27c1d"
BOT_NAME = "Nike-Zeus"
API_ROOT = 'https://api.groupme.com/v3/'
POST_URL = "https://api.groupme.com/v3/bots/post"
REMOVE_MEMBER_URL = "https://api.groupme.com/v3/groups/{group_id}/members/{member_id}?token={access_token}"
DELETE_MESSAGE_URL = "https://api.groupme.com/v3/groups/{group_id}/messages/{message_id}?token={access_token}"
access_token = "lQ9eg2l2TU9hexiRp69jlSkIXrv5u7h3AJkylPGn"  

app = Flask(__name__)

MESSAGE_CACHE_SIZE = 50
message_cache = defaultdict(lambda: deque(maxlen=MESSAGE_CACHE_SIZE))

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

selling_keywords = ['sell', 'selling', 'sale', 'sold', 'vending', 'trading', 'dealing','cheap','price','buying',]
ticket_keywords = ['ticket', 'tickets', 'admission', 'pass', 'entry']
concert_keywords = ['concert', 'show', 'performance', 'gig', 'event']
flagged_words = ['dm', 'messag', 'direct', 'contact','essay writer','student paper assignments','']

def get_flagged_words(message):
    words = re.findall(r'\b\w+\b', message.lower())
    flagged_words = []
    for word in words:
        if word in selling_keywords:
            flagged_words.append(('selling', word))
        elif word in ticket_keywords:
            flagged_words.append(('ticket', word))
        elif word in concert_keywords:
            flagged_words.append(('concert', word))
        elif word in flagged_words:
            flagged_words.append(('flagged', word))
    return flagged_words
keyword_regex = re.compile(r'\b(' + '|'.join(selling_keywords + ticket_keywords + concert_keywords + flagged_words) + r')\b', re.IGNORECASE)

#Change rates later...
RATE_LIMIT_WINDOW = 60  # Time window in seconds
RATE_LIMIT_COUNT = 100  # Maximum number of messages allowed within the time window

user_message_counts = defaultdict(list)

def load_training_data(file_path):
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the full path to the CSV file
    full_path = os.path.join(current_dir, file_path)
    
    training_data = []
    
    try:
        with open(full_path, 'r', encoding='utf-8', errors='replace') as file:
            csv_reader = csv.reader(file)
            next(csv_reader, None)  # Skip the header row
            for row_num, row in enumerate(csv_reader, start=2):  # Start at 2 to account for header
                if not row:  # Skip empty rows
                    print(f"Warning: Empty row found at line {row_num}. Skipping.")
                    continue
                if len(row) < 2:
                    print(f"Warning: Insufficient columns in row at line {row_num}. Skipping. Row content: {row}")
                    continue
                label, message = row[0], row[1]
                training_data.append((message, label))
    except FileNotFoundError:
        print(f"Error: The file {full_path} was not found.")
        return []
    except Exception as e:
        print(f"An error occurred while reading the file: {str(e)}")
        return []
    
    if not training_data:
        print("Warning: No valid data was loaded from the CSV file.")
    
    return training_data
# call func.. don't ask
training_data = load_training_data('spam.csv')


def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
    return ' '.join(filtered_tokens)

# Preprocess training data
X = [preprocess_text(text) for text, _ in training_data]
y = [label for _, label in training_data]

# Split data into training and testing sets, can make adjustments here
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the SVM classifier during startup, don't ask please - Km
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

def train_model():
    global current_model, current_vectorizer
    
    training_data = load_training_data('spam.csv')
    if not training_data:
        print("Error: No training data available. Model training aborted.")
        return
    
    X = [preprocess_text(text) for text, _ in training_data]
    y = [label for _, label in training_data]
#parameter adjustment area, will add a seperate function that globally addresses this later on... -KM
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    new_vectorizer = TfidfVectorizer()
    X_train_tfidf = new_vectorizer.fit_transform(X_train)
    X_test_tfidf = new_vectorizer.transform(X_test)
  
#For tuning SVM parameters
    new_model = SVC(kernel='linear', probability=True)

    new_model.fit(X_train_tfidf, y_train)

    # Use a lock to safely update the global model and vectorizer
    with model_lock:
        current_model = new_model
        current_vectorizer = new_vectorizer
    
    print("Model retrained and updated successfully")


svm_classifier = SVC(kernel='linear', probability=True)
svm_classifier.fit(X_train_tfidf, y_train)
#Model tuning on confidence threshold can also be done here, will impliment a conf__thresh to compare to spam_probability later on
def classify_message(message):
    preprocessed_message = preprocess_text(message)
    
    with model_lock:
        if current_model is None or current_vectorizer is None:
            return 0.0, []  # Return default values if model is not ready
        
        tfidf_feature_vector = current_vectorizer.transform([preprocessed_message])
        spam_probability = current_model.predict_proba(tfidf_feature_vector)[0][1]
        
        feature_names = current_vectorizer.get_feature_names_out()
        coef = current_model.coef_
        if issparse(coef):
            coef = coef.toarray()
        coef = coef.ravel()
        
        top_positive_coefficients = np.argsort(coef)[-10:]
        top_negative_coefficients = np.argsort(coef)[:10]
        top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
        
        word_importance = [(feature_names[i], coef[i]) for i in top_coefficients]
    
    return spam_probability, word_importance

def send_message(message):
    data = {
        "bot_id": BOT_ID,
        "text": message
    }
    print(f"Attempting to send message: {message}")
    try:
        response = requests.post(POST_URL, json=data)
        print(f"Response status code: {response.status_code}")
        print(f"Response content: {response.text}")
        response.raise_for_status()
        print(f"Successfully sent message: {message}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while sending the message: {e}")
        print(f"Response content: {e.response.text if e.response else 'No response'}")

def is_duplicate_message(user_id, message):
    user_cache = message_cache[user_id]
    return any(cached_msg['text'] == message for cached_msg in user_cache)

def add_to_cache(user_id, message):
    user_cache = message_cache[user_id]
    user_cache.append({'text': message, 'time': time()})

#threshhold limit can be adjusted/tuned
def is_spam(user_id, message):
    user_cache = message_cache[user_id]
    spam_count = sum(keyword_regex.search(cached_msg['text']) is not None for cached_msg in user_cache)
    return spam_count > 4 

#changing to 5 for example allows for more leniancy on spam messages
#changing to 2 or 1 makes the spam detection much more focused and stricter 
# (3-4 is the sweet spot in my opinion)

def is_rate_limited(user_id):
    current_time = time()
    user_message_counts[user_id] = [t for t in user_message_counts[user_id] if current_time - t <= RATE_LIMIT_WINDOW]
    if len(user_message_counts[user_id]) >= RATE_LIMIT_COUNT:
        return True
    user_message_counts[user_id].append(current_time)
    return False
def get_memberships(group_id):
    url = f'{API_ROOT}groups/{group_id}'
    params = {'token': access_token}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()['response']['members']
    else:
        print(f"Failed to retrieve memberships for group {group_id}. Status code: {response.status_code}")
        return []

def get_membership_id(group_id, user_id):
    memberships = get_memberships(group_id)
    for membership in memberships:
        if membership['user_id'] == user_id:
            return membership['id']
    return None

def remove_member(group_id, membership_id):
    url = f'{API_ROOT}groups/{group_id}/members/{membership_id}/remove'
    params = {'token': access_token}
    response = requests.post(url, params=params)
    if response.status_code == 200:
        print(f"Successfully removed member {membership_id} from group {group_id}")
        return True
    else:
        print(f"Failed to remove member {membership_id} from group {group_id}. Status code: {response.status_code}")
        return False

def delete_message(group_id, message_id):
    url = f'{API_ROOT}conversations/{group_id}/messages/{message_id}'
    params = {'token': access_token}
    response = requests.delete(url, params=params)
    if response.status_code == 204:
        print(f"Successfully deleted message {message_id} from group {group_id}")
        return True
    else:
        print(f"Failed to delete message {message_id} from group {group_id}. Status code: {response.status_code}")
        return False

def get_group_info(group_id):
    url = f'{API_ROOT}groups/{group_id}'
    params = {'token': access_token}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()['response']
    else:
        print(f"Failed to retrieve group info for group {group_id}. Status code: {response.status_code}")
        return None

def is_admin_or_creator(group_id, user_id):
    group_info = get_group_info(group_id)
    if group_info:
        # Check if the user is the group creator, safety feature #1 because the general members..
        if group_info['creator_user_id'] == user_id:
            return True
        
        # Check if the user is an admin, safety feature #2 because the general members..
        for member in group_info['members']:
            if member['user_id'] == user_id and member.get('roles', []) == ['admin']:
                return True
    
    return False

def kick_user(group_id, user_id):
    if is_admin_or_creator(group_id, user_id):#Safety feature to not look retarded
        print(f"Cannot kick user {user_id} as they are an admin or the group creator")
        return False 
    
    membership_id = get_membership_id(group_id, user_id)
    if membership_id:
        return remove_member(group_id, membership_id)
    else:
        print(f"User {user_id} not found in group {group_id}")
        #User not in group
        return False
    
def update_csv(message, label):
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spam.csv')
    
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        spam_writer = csv.writer(csvfile)
        if not file_exists:
            spam_writer.writerow(['label', 'message'])
        spam_writer.writerow([label, message, '', '', ''])
    
    print(f"Updated spam.csv with new {label} message: {message}")
    
    if label == 'spam':
        #model retraining in a separate thread, only if is spam message
        threading.Thread(target=train_model).start()

def add_ham_message(message):
    update_csv(message, 'ham')
    print("Ham message added to spam.csv without retraining the model")

def add_spam_message(message):
    update_csv(message, 'spam')
    print("Spam message added to spam.csv and model retraining triggered")

def handle_message(message, user_id, group_id, message_id, sender_id):
    print(f"Received message: {message}")
    
    if message.strip() == "A message was deleted":
        print("Ignoring deleted message notification")
        return
    
    print(f"Processing message: {message}")
    
    if is_rate_limited(user_id):
        print("User exceeded rate limit")
        send_message("You're sending messages too quickly. Please slow down.")
        return
    
    if BOT_NAME.lower() in message.lower() and "about" in message.lower():
        print("Responding with bot description")
        send_message("I'm a bot that leverages NLP techniques and machine learning to understand the content of messages and determine if they are related to specific topics (selling, tickets, concerts) or potentially spam/fraudulent. It helps in identifying and flagging messages that might require special attention or assistance.\n\nBy using NLTK, text preprocessing, and a Support Vector Machine (SVM) classifier with TF-IDF features, I can handle variations in word forms, filter out irrelevant words, and classify messages based on their content. The keyword matching and counting mechanism, along with the trained SVM classifier, allows me to determine the relevance and potential spam/fraudulent nature of a message.")
        return
    
    try:
        spam_probability, word_importance = classify_message(message)
        
        print(f"Spam probability: {spam_probability:.2%}")
        print("Top words contributing to classification:")
        for word, importance in word_importance:
            print(f"  - {word}: {importance:.4f}")
        
        if is_spam(user_id, message) or spam_probability > 0.5:
            print(f"Message flagged as spam")
            if sender_id != BOT_ID:
                send_message(f"[ALERT] This message has been flagged as spam or fraudulent with a probability of {spam_probability:.2%}. The user will be removed from the group, and the message will be deleted.")
                kick_user(group_id, user_id)
                delete_message(group_id, message_id)
                add_spam_message(message)  # Add spam message and trigger retraining
            else:
                print("Skipping deletion of bot's own message")
        else:
            print("Message not flagged as spam")
            if not is_duplicate_message(user_id, message):
                add_to_cache(user_id, message)
                add_ham_message(message)  # Add ham message without retraining
            else:
                print("Duplicate message found in user's cache, ignoring")
        
        if keyword_regex.search(message):
            flagged_words = get_flagged_words(message)
            print("Message flagged based on keyword matches:")
            for category, word in flagged_words:
                print(f"  - {category}: {word}")
    
    except Exception as e:
        print(f"An error occurred while processing the message: {str(e)}")
        # Add fallback behavior here later on.... 
        #makes debugging easier

@app.route('/', methods=['POST', 'GET', 'HEAD'])
def root():
    print(f"Received {request.method} request to /")
    
    if request.method == 'POST':
        try:
            data = request.get_json(force=True)
            print(f"Parsed JSON data: {json.dumps(data, indent=2)}")
            
            if data and 'name' in data and 'text' in data and 'user_id' in data and 'group_id' in data and 'id' in data and 'sender_id' in data:
                message_text = data['text']
                sender_name = data['name']
                user_id = data['user_id']
                group_id = data['group_id']
                message_id = data['id']
                sender_id = data['sender_id']
                
                if sender_name.lower() != BOT_NAME.lower():
                    print(f"Processing new message: {message_text}")
                    handle_message(message_text, user_id, group_id, message_id, sender_id)
                else:
                    print("Ignoring bot message")
            else:
                print("Received incomplete data in webhook")
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON data: {e}")
        except Exception as e:
            print(f"An error occurred while processing the webhook: {e}")
    
    return "OK", 200

if __name__ == "__main__":
    print(f"Starting server with BOT_ID: {BOT_ID}")
    
    #start initial trained model
    train_model()
    #send_message("Nike-Zues is starting up!\nThreat Detection System Online...\nThreat Aquisition System Online...\nThreat Response System Online...")
    app.run(debug=True, port=5000)#Yes, ik this is an issue, ill switch to s3 tommorow - Kuria